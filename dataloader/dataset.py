import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from preprocessing.preprocess import (
    pad_to_target, preprocess_mr, preprocess_ct, preprocess_cbct, scan_patients, find_nii
)
from preprocessing.augmentation import (
    apply_augmentation, apply_augmentation_25d
)


def _safe_collate(batch):
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    return default_collate(batch)


class SliceDataset2D(Dataset):
    def __init__(self, cache_dir, patient_list, target_size=320, cfg=None, augment=False):
        self.slices      = []
        self.target_size = target_size
        self.cfg         = cfg or {}
        self.augment     = augment
        for p in patient_list:
            p_dir = os.path.join(cache_dir, p["patient_id"])
            if not os.path.isdir(p_dir):
                continue
            for fname in sorted(os.listdir(p_dir)):
                if fname.endswith(".npy"):
                    self.slices.append(os.path.join(p_dir, fname))
        print(f"2D Dataset: {len(self.slices)} slices from {len(patient_list)} patients.")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        try:
            packed = np.load(self.slices[idx])
            inp  = pad_to_target(packed[0], self.target_size, self.target_size)
            ct   = pad_to_target(packed[1], self.target_size, self.target_size)
            mask = pad_to_target(packed[2], self.target_size, self.target_size)
            if self.augment:
                inp, ct, mask = apply_augmentation(inp, ct, mask, self.cfg)
            inp  = torch.from_numpy(inp).unsqueeze(0).float()
            ct   = torch.from_numpy(ct).unsqueeze(0).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            return inp, ct, mask
        except (OSError, FileNotFoundError) as e:
            print(f"  [WARN] Skipping {self.slices[idx]}: {e}")
            return None


class SliceDataset25D(Dataset):
    def __init__(self, cache_dir, patient_list, n_adjacent=3, target_size=320,
                 cfg=None, augment=False):
        assert n_adjacent % 2 == 1, "n_adjacent must be odd (e.g. 3, 5, 7)"
        self.cache_dir   = cache_dir
        self.target_size = target_size
        self.n_adjacent  = n_adjacent
        self.half        = n_adjacent // 2
        self.cfg         = cfg or {}
        self.augment     = augment
        self.samples     = []

        for p in patient_list:
            p_dir = os.path.join(cache_dir, p["patient_id"])
            if not os.path.isdir(p_dir):
                continue
            fnames = sorted([f for f in os.listdir(p_dir) if f.endswith(".npy")])
            slice_indices = [int(f.replace(".npy", "")) for f in fnames]
            index_set     = set(slice_indices)
            for idx in slice_indices:
                neighbors = [idx + offset for offset in range(-self.half, self.half + 1)]
                if all(n in index_set for n in neighbors):
                    self.samples.append((p_dir, idx, slice_indices))

        print(f"2.5D Dataset (n={n_adjacent}): {len(self.samples)} valid center slices "
              f"from {len(patient_list)} patients.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            p_dir, center_idx, slice_indices = self.samples[idx]
            half = self.half

            mr_slices = []
            for offset in range(-half, half + 1):
                fname  = f"{center_idx + offset:04d}.npy"
                packed = np.load(os.path.join(p_dir, fname))
                mr_sl  = pad_to_target(packed[0], self.target_size, self.target_size)
                mr_slices.append(mr_sl)

            center_packed = np.load(os.path.join(p_dir, f"{center_idx:04d}.npy"))
            ct   = pad_to_target(center_packed[1], self.target_size, self.target_size)
            mask = pad_to_target(center_packed[2], self.target_size, self.target_size)

            if self.augment:
                mr_slices, ct, mask = apply_augmentation_25d(mr_slices, ct, mask, self.cfg)

            inp_tensor  = torch.from_numpy(np.stack(mr_slices, axis=0)).float()
            ct_tensor   = torch.from_numpy(ct).unsqueeze(0).float()
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
            return inp_tensor, ct_tensor, mask_tensor
        except (OSError, FileNotFoundError) as e:
            print(f"  [WARN] Skipping 2.5D sample {idx} (patient {self.samples[idx][0]}): {e}")
            return None


class PatchDataset3D(Dataset):
    def __init__(self, patient_list, stats, cfg,
                 patch_size=96, patches_per_patient=8, augment=False,
                 input_file="mr.nii.gz"):
        self.patient_list        = patient_list
        self.stats               = stats
        self.patch_size          = patch_size
        self.patches_per_patient = patches_per_patient
        self.min_mask_coverage   = cfg["preprocessing"]["min_mask_coverage"]
        self.cfg                 = cfg
        self.augment             = augment
        self.input_file          = input_file
        self.samples             = []

        task = cfg.get("task", "task1")
        self.preprocess_input = preprocess_cbct if task == "task2" else preprocess_mr

        for p in patient_list:
            for _ in range(patches_per_patient):
                self.samples.append(p)

        # Preload all volumes into RAM to avoid repeated NFS reads during training
        print(f"3D Dataset: preloading {len(patient_list)} volumes into RAM...")
        self._vol_cache = {}
        for p in patient_list:
            try:
                self._vol_cache[p["patient_id"]] = self._load_volume(p)
            except (OSError, FileNotFoundError) as e:
                print(f"  [WARN] Skipping {p['patient_id']} at preload: {e}")
        n_loaded = len(self._vol_cache)
        print(f"3D Dataset (patch={patch_size}³): {len(self.samples)} patches "
              f"from {n_loaded} patients loaded.")
        if n_loaded == 0:
            print("  [ERROR] No volumes cached — all patients failed to load. "
                  "Check NFS connectivity or RAM availability.")

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, patient):
        pdir     = patient["path"]
        stem     = self.input_file.replace(".nii.gz", "").replace(".nii", "")
        inp_vol  = nib.load(find_nii(pdir, stem)).get_fdata(dtype=np.float32)
        ct_vol   = nib.load(find_nii(pdir, "ct")).get_fdata(dtype=np.float32)
        mask_vol = nib.load(find_nii(pdir, "mask")).get_fdata(dtype=np.float32)
        if stem == "mr":
            inp_proc = preprocess_mr(inp_vol, mask=mask_vol)
        else:
            inp_proc = preprocess_cbct(inp_vol, self.stats)
        ct_proc   = preprocess_ct(ct_vol, self.stats)
        mask_proc = (mask_vol > 0).astype(np.float32)
        return inp_proc, ct_proc, mask_proc

    def _random_patch(self, inp, ct, mask):
        ps = self.patch_size
        sx, sy, sz = inp.shape

        for _ in range(50):
            x = random.randint(0, max(0, sx - ps))
            y = random.randint(0, max(0, sy - ps))
            z = random.randint(0, max(0, sz - ps))
            m_patch = mask[x:x+ps, y:y+ps, z:z+ps]
            if m_patch.mean() >= self.min_mask_coverage:
                return (inp [x:x+ps, y:y+ps, z:z+ps],
                        ct  [x:x+ps, y:y+ps, z:z+ps],
                        m_patch)

        x = max(0, (sx - ps) // 2)
        y = max(0, (sy - ps) // 2)
        z = max(0, (sz - ps) // 2)
        return (inp [x:x+ps, y:y+ps, z:z+ps],
                ct  [x:x+ps, y:y+ps, z:z+ps],
                mask[x:x+ps, y:y+ps, z:z+ps])

    def _pad_patch(self, arr):
        ps  = self.patch_size
        pad = [(0, max(0, ps - s)) for s in arr.shape]
        return np.pad(arr, pad, mode="constant", constant_values=0)

    def __getitem__(self, idx):
        try:
            patient = self.samples[idx]
            cached  = self._vol_cache.get(patient["patient_id"])
            if cached is None:
                return None
            inp, ct, mask = cached
            inp_p, ct_p, m_p   = self._random_patch(inp, ct, mask)
            if self.augment:
                inp_p, ct_p, m_p = apply_augmentation(inp_p, ct_p, m_p, self.cfg, is_3d=True)
            inp_p = self._pad_patch(inp_p)
            ct_p  = self._pad_patch(ct_p)
            m_p   = self._pad_patch(m_p)
            inp_t  = torch.from_numpy(inp_p).unsqueeze(0).float()
            ct_t   = torch.from_numpy(ct_p).unsqueeze(0).float()
            mask_t = torch.from_numpy(m_p).unsqueeze(0).float()
            return inp_t, ct_t, mask_t
        except (OSError, FileNotFoundError) as e:
            print(f"  [WARN] Skipping 3D patient {self.samples[idx]['patient_id']}: {e}")
            return None


def build_dataloaders(cfg, stats):
    from torch.utils.data import DataLoader

    task       = cfg.get("task", "task1")
    train_root = cfg["paths"]["task2_train"] if task == "task2" else cfg["paths"]["task1_train"]
    cache_dir  = cfg["paths"]["cache_dir"]
    mode       = cfg["data"]["spatial_mode"]
    seed       = cfg["project"]["seed"]
    split      = cfg["data"]["train_val_split"]
    bs         = cfg["data"]["batch_size"]
    nw         = cfg["data"]["num_workers"]
    pin        = cfg["data"]["pin_memory"] and cfg["device"] == "cuda"
    input_file = "cbct.nii.gz" if task == "task2" else "mr.nii.gz"

    patients = scan_patients(train_root, require_ct=True)
    rng      = random.Random(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)
    split_idx     = int(split * len(shuffled))
    tr_patients   = shuffled[:split_idx]
    hold_patients = shuffled[split_idx:]

    target_size = cfg["preprocessing"]["target_size"]
    n_adjacent  = cfg["data"]["n_adjacent"]
    patch_size  = cfg["data"]["patch_size"]
    do_aug      = cfg.get("augmentation", {}).get("enabled", False)

    if mode == "2D":
        tr_ds   = SliceDataset2D(cache_dir, tr_patients,   target_size, cfg=cfg, augment=do_aug)
        hold_ds = SliceDataset2D(cache_dir, hold_patients, target_size, cfg=cfg, augment=False)
    elif mode == "2.5D":
        tr_ds   = SliceDataset25D(cache_dir, tr_patients,   n_adjacent, target_size, cfg=cfg, augment=do_aug)
        hold_ds = SliceDataset25D(cache_dir, hold_patients, n_adjacent, target_size, cfg=cfg, augment=False)
    elif mode == "3D":
        tr_ds   = PatchDataset3D(tr_patients,   stats, cfg, patch_size, augment=do_aug,   input_file=input_file)
        hold_ds = PatchDataset3D(hold_patients, stats, cfg, patch_size, augment=False, input_file=input_file)
    else:
        raise ValueError(f"Unknown spatial_mode: {mode}")

    tr_loader   = DataLoader(tr_ds,   batch_size=bs, shuffle=True,
                             num_workers=nw, pin_memory=pin,
                             collate_fn=_safe_collate)
    hold_loader = DataLoader(hold_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=pin,
                             collate_fn=_safe_collate)

    print(f"Train: {len(tr_loader)} batches | Val: {len(hold_loader)} batches")
    return tr_loader, hold_loader, tr_patients, hold_patients
