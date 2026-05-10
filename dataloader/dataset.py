import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from preprocessing.preprocess import (
    pad_to_target, preprocess_mr, preprocess_ct, scan_patients
)


def _safe_collate(batch):
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    return default_collate(batch)


class SliceDataset2D(Dataset):
    def __init__(self, cache_dir, patient_list, target_size=320):
        self.slices      = []
        self.target_size = target_size
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
            mr   = pad_to_target(packed[0], self.target_size, self.target_size)
            ct   = pad_to_target(packed[1], self.target_size, self.target_size)
            mask = pad_to_target(packed[2], self.target_size, self.target_size)
            mr   = torch.from_numpy(mr).unsqueeze(0)
            ct   = torch.from_numpy(ct).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)
            return mr, ct, mask
        except (OSError, FileNotFoundError) as e:
            print(f"  [WARN] Skipping {self.slices[idx]}: {e}")
            return None


class SliceDataset25D(Dataset):
    def __init__(self, cache_dir, patient_list, n_adjacent=3, target_size=320):
        assert n_adjacent % 2 == 1, "n_adjacent must be odd (e.g. 3, 5, 7)"
        self.cache_dir   = cache_dir
        self.target_size = target_size
        self.n_adjacent  = n_adjacent
        self.half        = n_adjacent // 2
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

            mr_tensor   = torch.from_numpy(np.stack(mr_slices, axis=0)).float()
            ct_tensor   = torch.from_numpy(ct).unsqueeze(0).float()
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
            return mr_tensor, ct_tensor, mask_tensor
        except (OSError, FileNotFoundError) as e:
            print(f"  [WARN] Skipping 2.5D sample {idx} (patient {self.samples[idx][0]}): {e}")
            return None


class PatchDataset3D(Dataset):
    def __init__(self, patient_list, stats, cfg,
                 patch_size=96, patches_per_patient=8):
        self.patient_list      = patient_list
        self.stats             = stats
        self.patch_size        = patch_size
        self.patches_per_patient = patches_per_patient
        self.min_mask_coverage = cfg["preprocessing"]["min_mask_coverage"]
        self.samples           = []

        for p in patient_list:
            for _ in range(patches_per_patient):
                self.samples.append(p)

        print(f"3D Dataset (patch={patch_size}³): {len(self.samples)} patches "
              f"from {len(patient_list)} patients.")

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, patient):
        pdir     = patient["path"]
        mr_vol   = nib.load(os.path.join(pdir, "mr.nii.gz")).get_fdata(dtype=np.float32)
        ct_vol   = nib.load(os.path.join(pdir, "ct.nii.gz")).get_fdata(dtype=np.float32)
        mask_vol = nib.load(os.path.join(pdir, "mask.nii.gz")).get_fdata(dtype=np.float32)
        mr_proc  = preprocess_mr(mr_vol, mask=mask_vol)
        ct_proc  = preprocess_ct(ct_vol, self.stats)
        mask_proc = (mask_vol > 0).astype(np.float32)
        return mr_proc, ct_proc, mask_proc

    def _random_patch(self, mr, ct, mask):
        ps = self.patch_size
        sx, sy, sz = mr.shape

        for _ in range(50):
            x = random.randint(0, max(0, sx - ps))
            y = random.randint(0, max(0, sy - ps))
            z = random.randint(0, max(0, sz - ps))
            m_patch = mask[x:x+ps, y:y+ps, z:z+ps]
            if m_patch.mean() >= self.min_mask_coverage:
                mr_patch   = mr  [x:x+ps, y:y+ps, z:z+ps]
                ct_patch   = ct  [x:x+ps, y:y+ps, z:z+ps]
                return mr_patch, ct_patch, m_patch

        x = max(0, (sx - ps) // 2)
        y = max(0, (sy - ps) // 2)
        z = max(0, (sz - ps) // 2)
        return (mr  [x:x+ps, y:y+ps, z:z+ps],
                ct  [x:x+ps, y:y+ps, z:z+ps],
                mask[x:x+ps, y:y+ps, z:z+ps])

    def _pad_patch(self, arr):
        ps = self.patch_size
        pad = [(0, max(0, ps - s)) for s in arr.shape]
        return np.pad(arr, pad, mode="constant", constant_values=0)

    def __getitem__(self, idx):
        try:
            patient          = self.samples[idx]
            mr, ct, mask     = self._load_volume(patient)
            mr_p, ct_p, m_p = self._random_patch(mr, ct, mask)
            mr_p  = self._pad_patch(mr_p)
            ct_p  = self._pad_patch(ct_p)
            m_p   = self._pad_patch(m_p)
            mr_t  = torch.from_numpy(mr_p).unsqueeze(0).float()
            ct_t  = torch.from_numpy(ct_p).unsqueeze(0).float()
            mask_t = torch.from_numpy(m_p).unsqueeze(0).float()
            return mr_t, ct_t, mask_t
        except (OSError, FileNotFoundError) as e:
            print(f"  [WARN] Skipping 3D patient {self.samples[idx]['patient_id']}: {e}")
            return None


def build_dataloaders(cfg, stats):
    from preprocessing.preprocess import scan_patients
    from torch.utils.data import DataLoader

    train_root = cfg["paths"]["task1_train"]
    cache_dir  = cfg["paths"]["cache_dir"]
    mode       = cfg["data"]["spatial_mode"]
    seed       = cfg["project"]["seed"]
    split      = cfg["data"]["train_val_split"]
    bs         = cfg["data"]["batch_size"]
    nw         = cfg["data"]["num_workers"]
    pin        = cfg["data"]["pin_memory"] and cfg["device"] == "cuda"

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

    if mode == "2D":
        tr_ds   = SliceDataset2D(cache_dir, tr_patients,   target_size)
        hold_ds = SliceDataset2D(cache_dir, hold_patients, target_size)
    elif mode == "2.5D":
        tr_ds   = SliceDataset25D(cache_dir, tr_patients,   n_adjacent, target_size)
        hold_ds = SliceDataset25D(cache_dir, hold_patients, n_adjacent, target_size)
    elif mode == "3D":
        tr_ds   = PatchDataset3D(tr_patients,   stats, cfg, patch_size)
        hold_ds = PatchDataset3D(hold_patients, stats, cfg, patch_size)
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