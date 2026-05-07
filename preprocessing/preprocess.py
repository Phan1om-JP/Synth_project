import json
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm


def compute_and_save_stats(train_root, stats_path, ct_clip_min, ct_clip_max):
    patients = scan_patients(train_root, require_ct=True)
    print(f"Computing stats from {len(patients)} patients...")

    sum_v, sum_sq_v, n_v = 0.0, 0.0, 0
    mr_p01_list, mr_p99_list = [], []

    for p in tqdm(patients, desc="Stats"):
        ct_vol = nib.load(os.path.join(p["path"], "ct.nii.gz")).get_fdata(
            dtype=np.float32, caching="unchanged")
        mr_vol = nib.load(os.path.join(p["path"], "mr.nii.gz")).get_fdata(
            dtype=np.float32, caching="unchanged")
        mask_vol = nib.load(os.path.join(p["path"], "mask.nii.gz")).get_fdata(
            dtype=np.float32, caching="unchanged")

        ct_clipped = np.clip(ct_vol, ct_clip_min, ct_clip_max)
        sum_v    += float(ct_clipped.sum())
        sum_sq_v += float((ct_clipped ** 2).sum())
        n_v      += ct_clipped.size

        brain_voxels = mr_vol[mask_vol > 0]
        if brain_voxels.size == 0:
            brain_voxels = mr_vol[mr_vol > 0]
        mr_p01_list.append(float(np.percentile(brain_voxels, 1)))
        mr_p99_list.append(float(np.percentile(brain_voxels, 99)))

    ct_mean = float(sum_v / n_v)
    ct_std  = float(np.sqrt(sum_sq_v / n_v - ct_mean ** 2))

    stats = {
        "ct_clip_min"    : float(ct_clip_min),
        "ct_clip_max"    : float(ct_clip_max),
        "ct_global_mean" : ct_mean,
        "ct_global_std"  : ct_std,
        "mr_p01_mean"    : float(np.mean(mr_p01_list)),
        "mr_p99_mean"    : float(np.mean(mr_p99_list)),
        "mr_p01_std"     : float(np.std(mr_p01_list)),
        "mr_p99_std"     : float(np.std(mr_p99_list)),
        "n_patients"     : len(patients),
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Stats saved: {stats_path}")
    return stats


def load_or_compute_stats(cfg):
    stats_path = cfg["paths"]["stats_path"]
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    return compute_and_save_stats(
        cfg["paths"]["task1_train"],
        stats_path,
        cfg["preprocessing"]["ct_clip_min"],
        cfg["preprocessing"]["ct_clip_max"],
    )


def preprocess_mr(vol, mask=None):
    brain_voxels = vol[mask > 0] if mask is not None else vol[vol > 0]
    if brain_voxels.size == 0:
        brain_voxels = vol
    p01 = float(np.percentile(brain_voxels, 1))
    p99 = float(np.percentile(brain_voxels, 99))
    vol = np.clip(vol, p01, p99)
    vol = (vol - p01) / (p99 - p01 + 1e-8)
    return vol.astype(np.float32)


def preprocess_ct(vol, stats):
    vol = np.clip(vol, stats["ct_clip_min"], stats["ct_clip_max"])
    vol = (vol - stats["ct_global_mean"]) / (stats["ct_global_std"] + 1e-8)
    return vol.astype(np.float32)


def postprocess_ct(vol, stats):
    vol = vol * stats["ct_global_std"] + stats["ct_global_mean"]
    vol = np.clip(vol, stats["ct_clip_min"], stats["ct_clip_max"])
    return vol.astype(np.float32)


def scan_patients(root, require_ct=True):
    patients = []
    for pid in sorted(os.listdir(root)):
        if pid == "overview":
            continue
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue
        try:
            files = os.listdir(pdir)
        except OSError as e:
            print(f"  [WARN] Skipping {pid}: {e}")
            continue
        if require_ct and "ct.nii.gz" not in files:
            continue
        patients.append({"patient_id": pid, "path": pdir})
    return patients


def pad_to_target(arr, target_h, target_w):
    h, w = arr.shape
    pad_top    = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left   = (target_w - w) // 2
    pad_right  = target_w - w - pad_left
    return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)),
                  mode="constant", constant_values=0)


def cache_patient_slices(patient, cache_dir, stats, cfg):
    pid  = patient["patient_id"]
    pdir = patient["path"]
    p_cache_dir = os.path.join(cache_dir, pid)
    os.makedirs(p_cache_dir, exist_ok=True)

    mr_vol   = nib.load(os.path.join(pdir, "mr.nii.gz")).get_fdata(dtype=np.float32)
    ct_vol   = nib.load(os.path.join(pdir, "ct.nii.gz")).get_fdata(dtype=np.float32)
    mask_vol = nib.load(os.path.join(pdir, "mask.nii.gz")).get_fdata(dtype=np.float32)

    mr_proc   = preprocess_mr(mr_vol, mask=mask_vol)
    ct_proc   = preprocess_ct(ct_vol, stats)
    mask_proc = (mask_vol > 0).astype(np.float32)

    plane            = cfg["data"]["plane"]
    min_mask_coverage = cfg["preprocessing"]["min_mask_coverage"]

    if plane == "axial":
        n_slices  = mr_proc.shape[2]
        get_slice = lambda v, i: v[:, :, i]
    elif plane == "coronal":
        n_slices  = mr_proc.shape[1]
        get_slice = lambda v, i: v[:, i, :]
    else:
        n_slices  = mr_proc.shape[0]
        get_slice = lambda v, i: v[i, :, :]

    saved = 0
    for i in range(n_slices):
        mask_sl = get_slice(mask_proc, i)
        if mask_sl.mean() < min_mask_coverage:
            continue
        mr_sl = get_slice(mr_proc, i)
        ct_sl = get_slice(ct_proc, i)
        packed = np.stack([mr_sl, ct_sl, mask_sl], axis=0).astype(np.float32)
        np.save(os.path.join(p_cache_dir, f"{i:04d}.npy"), packed)
        saved += 1
    return saved


def build_cache(cfg, stats):
    train_root = cfg["paths"]["task1_train"]
    cache_dir  = cfg["paths"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    patients = scan_patients(train_root, require_ct=True)
    total_slices = 0

    for p in tqdm(patients, desc="Caching"):
        pid_cache = os.path.join(cache_dir, p["patient_id"])
        if os.path.isdir(pid_cache) and len(os.listdir(pid_cache)) > 0:
            total_slices += len(os.listdir(pid_cache))
            continue
        total_slices += cache_patient_slices(p, cache_dir, stats, cfg)

    print(f"Cache complete: {total_slices} slices from {len(patients)} patients.")
    return patients