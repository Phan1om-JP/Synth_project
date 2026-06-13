import json
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm


def find_nii(folder: str, stem: str) -> str:
    """Return path to stem.nii.gz or stem.nii, whichever exists."""
    for ext in (".nii.gz", ".nii"):
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{stem}.nii.gz / {stem}.nii not found in {folder}")


def compute_and_save_stats(train_root, stats_path, ct_clip_min, ct_clip_max):
    patients = scan_patients(train_root, require_ct=True)
    print(f"Computing stats from {len(patients)} patients...")

    sum_v, sum_sq_v, n_v = 0.0, 0.0, 0
    mr_p01_list, mr_p99_list = [], []

    for p in tqdm(patients, desc="Stats"):
        ct_vol = nib.load(find_nii(p["path"], "ct")).get_fdata(
            dtype=np.float32, caching="unchanged")
        mr_vol = nib.load(find_nii(p["path"], "mr")).get_fdata(
            dtype=np.float32, caching="unchanged")
        mask_vol = nib.load(find_nii(p["path"], "mask")).get_fdata(
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


def preprocess_cbct(vol, stats):
    vol = np.clip(vol, stats["cbct_clip_min"], stats["cbct_clip_max"])
    vol = (vol - stats["cbct_global_mean"]) / (stats["cbct_global_std"] + 1e-8)
    return vol.astype(np.float32)


def compute_and_save_stats_task2(train_root, stats_path, cbct_clip_min, cbct_clip_max,
                                  ct_clip_min, ct_clip_max):
    patients = scan_patients(train_root, require_ct=True)
    print(f"Computing Task2 stats from {len(patients)} patients...")

    cbct_sum, cbct_sq, cbct_n = 0.0, 0.0, 0
    ct_sum,   ct_sq,   ct_n   = 0.0, 0.0, 0

    for p in tqdm(patients, desc="Task2 stats"):
        cbct_vol = nib.load(find_nii(p["path"], "cbct")).get_fdata(
            dtype=np.float32, caching="unchanged")
        ct_vol   = nib.load(find_nii(p["path"], "ct")).get_fdata(
            dtype=np.float32, caching="unchanged")

        cbct_c = np.clip(cbct_vol, cbct_clip_min, cbct_clip_max)
        cbct_sum += float(cbct_c.sum());  cbct_sq += float((cbct_c**2).sum())
        cbct_n   += cbct_c.size

        ct_c = np.clip(ct_vol, ct_clip_min, ct_clip_max)
        ct_sum += float(ct_c.sum());  ct_sq += float((ct_c**2).sum())
        ct_n   += ct_c.size

    cbct_mean = cbct_sum / cbct_n
    cbct_std  = float(np.sqrt(cbct_sq / cbct_n - cbct_mean**2))
    ct_mean   = ct_sum   / ct_n
    ct_std    = float(np.sqrt(ct_sq   / ct_n   - ct_mean**2))

    stats = {
        "cbct_clip_min"   : float(cbct_clip_min),
        "cbct_clip_max"   : float(cbct_clip_max),
        "cbct_global_mean": float(cbct_mean),
        "cbct_global_std" : float(cbct_std),
        "ct_clip_min"     : float(ct_clip_min),
        "ct_clip_max"     : float(ct_clip_max),
        "ct_global_mean"  : float(ct_mean),
        "ct_global_std"   : float(ct_std),
        "n_patients"      : len(patients),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Task2 stats saved: {stats_path}")
    return stats


def load_or_compute_stats_task2(cfg):
    stats_path = cfg["paths"]["stats_path"]
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    return compute_and_save_stats_task2(
        cfg["paths"]["task2_train"],
        stats_path,
        cfg["preprocessing"]["cbct_clip_min"],
        cfg["preprocessing"]["cbct_clip_max"],
        cfg["preprocessing"]["ct_clip_min"],
        cfg["preprocessing"]["ct_clip_max"],
    )


def cache_patient_slices_task2(patient, cache_dir, stats, cfg):
    pid  = patient["patient_id"]
    pdir = patient["path"]
    p_cache_dir = os.path.join(cache_dir, pid)
    os.makedirs(p_cache_dir, exist_ok=True)

    cbct_vol = nib.load(find_nii(pdir, "cbct")).get_fdata(dtype=np.float32)
    ct_vol   = nib.load(find_nii(pdir, "ct")).get_fdata(dtype=np.float32)
    mask_vol = nib.load(find_nii(pdir, "mask")).get_fdata(dtype=np.float32)

    cbct_proc = preprocess_cbct(cbct_vol, stats)
    ct_proc   = preprocess_ct(ct_vol, stats)
    mask_proc = (mask_vol > 0).astype(np.float32)

    plane             = cfg["data"]["plane"]
    min_mask_coverage = cfg["preprocessing"]["min_mask_coverage"]

    if plane == "axial":
        n_slices  = cbct_proc.shape[2]
        get_slice = lambda v, i: v[:, :, i]
    elif plane == "coronal":
        n_slices  = cbct_proc.shape[1]
        get_slice = lambda v, i: v[:, i, :]
    else:
        n_slices  = cbct_proc.shape[0]
        get_slice = lambda v, i: v[i, :, :]

    saved = 0
    for i in range(n_slices):
        mask_sl = get_slice(mask_proc, i)
        if mask_sl.mean() < min_mask_coverage:
            continue
        cbct_sl = get_slice(cbct_proc, i)
        ct_sl   = get_slice(ct_proc, i)
        packed  = np.stack([cbct_sl, ct_sl, mask_sl], axis=0).astype(np.float16)
        np.save(os.path.join(p_cache_dir, f"{i:04d}.npy"), packed)
        saved += 1
    return saved


def build_cache_task2(cfg, stats):
    train_root = cfg["paths"]["task2_train"]
    cache_dir  = cfg["paths"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    patients     = scan_patients(train_root, require_ct=True)
    total_slices = 0

    for p in tqdm(patients, desc="Caching Task2"):
        pid_cache = os.path.join(cache_dir, p["patient_id"])
        if os.path.isdir(pid_cache) and len(os.listdir(pid_cache)) > 0:
            total_slices += len(os.listdir(pid_cache))
            continue
        total_slices += cache_patient_slices_task2(p, cache_dir, stats, cfg)

    print(f"Task2 cache complete: {total_slices} slices from {len(patients)} patients.")
    return patients


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
        if require_ct and "ct.nii.gz" not in files and "ct.nii" not in files:
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

    mr_vol   = nib.load(find_nii(pdir, "mr")).get_fdata(dtype=np.float32)
    ct_vol   = nib.load(find_nii(pdir, "ct")).get_fdata(dtype=np.float32)
    mask_vol = nib.load(find_nii(pdir, "mask")).get_fdata(dtype=np.float32)

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
        packed = np.stack([mr_sl, ct_sl, mask_sl], axis=0).astype(np.float16)
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