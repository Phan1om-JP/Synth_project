import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def compute_image_metrics(ct_path, sct_path, mask_path):
    ct_arr   = nib.load(ct_path).get_fdata(dtype=np.float32)
    sct_arr  = nib.load(sct_path).get_fdata(dtype=np.float32)
    mask_arr = nib.load(mask_path).get_fdata() > 0

    m    = mask_arr
    mae  = float(np.abs(ct_arr[m] - sct_arr[m]).mean())
    rmse = float(np.sqrt(((ct_arr[m] - sct_arr[m])**2).mean()))
    data_range = float(ct_arr[m].max() - ct_arr[m].min())
    psnr = float(20 * np.log10(data_range / (rmse + 1e-8)))

    ssim_scores = []
    for z in range(ct_arr.shape[2]):
        sl_ct  = ct_arr[:, :, z]
        sl_sct = sct_arr[:, :, z]
        if mask_arr[:, :, z].sum() > 100:
            dr = sl_ct.max() - sl_ct.min()
            if dr > 0:
                ssim_scores.append(ssim_fn(sl_ct, sl_sct, data_range=float(dr)))

    return {
        "mae"  : mae,
        "rmse" : rmse,
        "psnr" : psnr,
        "ssim" : float(np.mean(ssim_scores)) if ssim_scores else 0.0,
    }


def gamma_index_3d(ref, evl, spacing_mm, dd=0.02, dta_mm=2.0, thr=0.1):
    ref_max = ref.max()
    if ref_max == 0:
        return np.full(ref.shape, np.nan), 0.0

    roi    = ref > thr * ref_max
    r_vox  = int(np.ceil(dta_mm / np.min(spacing_mm)))
    zz, yy, xx = np.mgrid[-r_vox:r_vox+1, -r_vox:r_vox+1, -r_vox:r_vox+1]
    dist   = np.sqrt((zz*spacing_mm[2])**2 +
                     (yy*spacing_mm[1])**2 +
                     (xx*spacing_mm[0])**2)
    km     = dist <= dta_mm * 1.5
    zz, yy, xx, dist = zz[km], yy[km], xx[km], dist[km]

    gamma  = np.full(ref.shape, np.nan)
    g_list = []

    for iz, iy, ix in np.argwhere(roi):
        nz = np.clip(iz+zz, 0, ref.shape[0]-1)
        ny = np.clip(iy+yy, 0, ref.shape[1]-1)
        nx = np.clip(ix+xx, 0, ref.shape[2]-1)
        rv = ref[iz, iy, ix]
        ev = evl[nz, ny, nx]
        g  = np.sqrt(((ev-rv)/(dd*ref_max))**2 + (dist/dta_mm)**2)
        gm = float(g.min())
        gamma[iz, iy, ix] = gm
        g_list.append(gm)

    g_arr     = np.array(g_list)
    pass_rate = float((g_arr <= 1.0).mean() * 100) if len(g_arr) > 0 else 0.0
    return gamma, pass_rate


def compute_dvh(dose, struct_mask, n_bins=100):
    d_vals = dose[struct_mask > 0]
    if len(d_vals) == 0:
        return np.array([]), np.array([])
    bins = np.linspace(0, d_vals.max() * 1.05, n_bins)
    dvh  = np.array([(d_vals >= b).mean() * 100 for b in bins])
    return bins, dvh


def compute_dose_metrics(patient_out_dir, modality, cfg, struct_names=None):
    if struct_names is None:
        struct_names = ["PTV_thalamus", "OAR_brainstem",
                        "OAR_cerebellum", "OAR_hippocampus"]

    d_ct_nib = nib.load(
        f"{patient_out_dir}/dose_ct_{modality}.nii.gz")
    d_sct_nib = nib.load(
        f"{patient_out_dir}/dose_sct_{modality}.nii.gz")

    d_ct  = d_ct_nib.get_fdata(dtype=np.float32)
    d_sct = d_sct_nib.get_fdata(dtype=np.float32)
    spacing = np.array(d_ct_nib.header.get_zooms())

    ref_max = d_ct.max()
    d_ct_n  = d_ct  / (ref_max + 1e-8)
    d_sct_n = d_sct / (ref_max + 1e-8)
    diff    = d_sct_n - d_ct_n

    gamma_map, pass_rate = gamma_index_3d(
        d_ct_n, d_sct_n, spacing,
        dd=cfg["dose"]["gamma_dd"],
        dta_mm=cfg["dose"]["gamma_dta_mm"],
        thr=cfg["dose"]["gamma_threshold"],
    )

    dvh_results = {}
    struct_dir  = os.path.join(patient_out_dir, "structures")
    for sname in struct_names:
        spath = os.path.join(struct_dir, f"{sname}.nii.gz")
        if not os.path.exists(spath):
            continue
        smask = nib.load(spath).get_fdata() > 0
        bins_ct,  dvh_ct  = compute_dvh(d_ct_n,  smask)
        bins_sct, dvh_sct = compute_dvh(d_sct_n, smask)
        dvh_results[sname] = {
            "bins_ct": bins_ct, "dvh_ct": dvh_ct,
            "bins_sct": bins_sct, "dvh_sct": dvh_sct,
            "mean_diff": float(diff[smask].mean() * 100) if smask.sum() > 0 else 0.0,
            "max_diff" : float(np.abs(diff[smask]).max() * 100) if smask.sum() > 0 else 0.0,
        }

    return {
        "global_mean_diff_pct": float(diff.mean() * 100),
        "global_max_diff_pct" : float(np.abs(diff).max() * 100),
        "gamma_pass_rate"     : pass_rate,
        "dvh"                 : dvh_results,
    }


import os