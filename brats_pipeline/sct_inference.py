"""
sct_inference.py
Generate synthetic CT (sCT) from BraTS T1c MRI using a pretrained
SynthRAD 2.5D UNet model.

Requires models/unet.py from the SynthRAD repo to be on the Python path.
Upload it to Colab alongside this file.

Usage in Colab:
    from sct_inference import run_patient_sct, verify_sct
"""

import sys
import json
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


# CT normalization stats from SynthRAD training (180 patients, Task 1)
# Source: normalisation_stats.json on the training server
CT_GLOBAL_MEAN = -626.0638
CT_GLOBAL_STD  =  551.8138
CT_CLIP_MIN    = -1000.0
CT_CLIP_MAX    =  1000.0
TARGET_SIZE    = 320       # spatial size the model was trained on


# ---------------------------------------------------------------------------
# Preprocessing / postprocessing
# ---------------------------------------------------------------------------

def preprocess_mr(vol):
    """
    Clip to 1st–99th percentile of brain voxels, then normalize to [0, 1].
    Same logic used during SynthRAD training.
    """
    mask = vol > 0
    brain = vol[mask] if mask.any() else vol.ravel()
    lo = float(np.percentile(brain, 1))
    hi = float(np.percentile(brain, 99))
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-8)
    return vol.astype(np.float32)


def postprocess_ct(vol):
    """Convert model output (z-scored) back to Hounsfield Units."""
    vol = vol * CT_GLOBAL_STD + CT_GLOBAL_MEAN
    return np.clip(vol, CT_CLIP_MIN, CT_CLIP_MAX).astype(np.float32)


def _pad_to_target(vol_3d, target=TARGET_SIZE):
    """Pad H and W axes to target x target. Returns padded vol and pad sizes."""
    h, w = vol_3d.shape[0], vol_3d.shape[1]
    ph = max(0, target - h)
    pw = max(0, target - w)
    ph0, ph1 = ph // 2, ph - ph // 2
    pw0, pw1 = pw // 2, pw - pw // 2
    vol_3d = np.pad(vol_3d, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant")
    return vol_3d, (ph0, ph1, pw0, pw1, h, w)


def _unpad(vol_3d, pad_info):
    ph0, ph1, pw0, pw1, orig_h, orig_w = pad_info
    return vol_3d[ph0: ph0 + orig_h, pw0: pw0 + orig_w, :]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _import_unet():
    """Try several locations for unet.py and return the UNet2D class."""
    import importlib.util

    candidates = [
        # standard: models/unet.py relative to sys.path
        ("models.unet", None),
    ]
    # also search common Colab upload locations
    for root in ["/content", "/content/models", "/content/drive/MyDrive"]:
        for name in ["unet.py", "models/unet.py"]:
            p = Path(root) / name
            if p.exists():
                candidates.append((None, str(p)))

    for mod_name, file_path in candidates:
        try:
            if mod_name:
                mod = importlib.import_module(mod_name)
            else:
                spec = importlib.util.spec_from_file_location("unet", file_path)
                mod  = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            return mod.UNet2D
        except Exception:
            continue

    raise ImportError(
        "Cannot find UNet2D. Upload models/unet.py from the SynthRAD repo "
        "to /content/unet.py or /content/models/unet.py and retry."
    )


def load_model(ckpt_path, device="cuda"):
    """Load the 2.5D UNet from a SynthRAD checkpoint."""
    UNet2D = _import_unet()

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = UNet2D(in_ch=3, out_ch=1, base_ch=64)

    # Strip DataParallel "module." prefix if present
    state = {k.replace("module.", ""): v
             for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)
    model.to(device).eval()

    epoch = ckpt.get("epoch", "n/a")
    mae   = ckpt.get("best_val_mae", "n/a")
    print(f"Loaded model — epoch {epoch}, best val MAE {mae:.2f} HU"
          if isinstance(mae, float) else f"Loaded model — epoch {epoch}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_volume(model, t1c_proc, device="cpu"):
    """
    Run 2.5D slice-by-slice inference on a preprocessed (padded) T1c volume.
    t1c_proc: (H, W, Z) float32 in [0, 1]
    Returns: (H, W, Z) float32 z-scored CT predictions
    """
    H, W, Z = t1c_proc.shape
    sct = np.zeros((H, W, Z), dtype=np.float32)

    with torch.no_grad():
        for z in range(Z):
            z_prev = max(0, z - 1)
            z_next = min(Z - 1, z + 1)

            # Stack 3 adjacent slices as channels: (1, 3, H, W)
            inp = np.stack([
                t1c_proc[:, :, z_prev],
                t1c_proc[:, :, z],
                t1c_proc[:, :, z_next],
            ], axis=0)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
            pred  = model(inp_t).squeeze().cpu().numpy()
            sct[:, :, z] = pred

    return sct


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def verify_sct(t1c_path, sct_path, n_slices=5):
    """
    Show T1c and sCT side by side on several axial slices.
    Check that bone appears bright and soft tissue is in the right HU range.
    """
    t1c = nib.load(str(t1c_path)).get_fdata()
    sct = nib.load(str(sct_path)).get_fdata()

    # Sample slices across the brain (skip top and bottom 10%)
    Z = t1c.shape[2]
    z_samples = np.linspace(int(Z * 0.1), int(Z * 0.9), n_slices, dtype=int)

    fig, axes = plt.subplots(2, n_slices, figsize=(3 * n_slices, 6))
    t1c_vmin, t1c_vmax = np.percentile(t1c, 1), np.percentile(t1c, 99)

    for i, z in enumerate(z_samples):
        axes[0, i].imshow(t1c[:, :, z].T, cmap="gray", origin="lower",
                          vmin=t1c_vmin, vmax=t1c_vmax)
        axes[0, i].set_title(f"T1c  z={z}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(sct[:, :, z].T, cmap="gray", origin="lower",
                          vmin=-200, vmax=800)
        axes[1, i].set_title(f"sCT  z={z}", fontsize=8)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("T1c (input)", fontsize=9)
    axes[1, 0].set_ylabel("sCT (output)", fontsize=9)
    plt.suptitle("T1c vs synthetic CT — bone should appear bright, "
                 "soft tissue mid-grey", fontsize=10)
    plt.tight_layout()
    plt.show()

    head_mask = sct > (CT_CLIP_MIN + 1)   # anything above -1000 is predicted tissue
    print(f"sCT HU range       : [{sct.min():.1f}, {sct.max():.1f}]")
    print(f"sCT mean (head)    : {sct[head_mask].mean():.1f} HU")
    print(f"Expected soft tissue : 20–80 HU")
    print(f"Expected skull bone  : 400–1000 HU")


# ---------------------------------------------------------------------------
# Main per-patient entry point
# ---------------------------------------------------------------------------

def run_patient_sct(patient_id, t1c_path, out_dir, ckpt_path,
                    device="cuda", log_path=None):
    """
    Generate sCT for one BraTS patient.

    Args:
        patient_id  : string ID for output filename
        t1c_path    : path to T1c NIfTI
        out_dir     : output directory
        ckpt_path   : path to .pt checkpoint file
        device      : "cpu" or "cuda"
        log_path    : optional JSON log file

    Returns path to saved sCT NIfTI.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model = load_model(ckpt_path, device=device)

    t1c_img = nib.as_closest_canonical(nib.load(str(t1c_path)))
    t1c_vol = t1c_img.get_fdata().astype(np.float32)

    t1c_proc = preprocess_mr(t1c_vol)

    t1c_padded, pad_info = _pad_to_target(t1c_proc, TARGET_SIZE)

    print(f"Running inference on {patient_id} "
          f"({t1c_vol.shape[2]} slices, device={device}) ...")
    sct_padded = infer_volume(model, t1c_padded, device=device)

    sct_unpadded = _unpad(sct_padded, pad_info)
    sct_hu       = postprocess_ct(sct_unpadded)

    # Build a clean head mask using the largest connected component.
    # Simple threshold + largest-component keeps the actual head and
    # removes fiducial markers, scanner noise, and other small artifacts.
    from scipy import ndimage as _ndi
    rough_mask  = t1c_vol > (float(t1c_vol.max()) * 0.01)
    labeled, _  = _ndi.label(rough_mask)
    counts      = np.bincount(labeled.ravel())
    counts[0]   = 0                          # ignore background label
    head_mask   = labeled == counts.argmax()
    head_mask   = _ndi.binary_fill_holes(head_mask)
    sct_hu[~head_mask] = CT_CLIP_MIN

    out_path = Path(out_dir) / f"{patient_id}_sct.nii.gz"
    nib.save(nib.Nifti1Image(sct_hu, t1c_img.affine), str(out_path))

    brain_mask = head_mask
    result = {
        "patient_id":   patient_id,
        "timestamp":    datetime.utcnow().isoformat(),
        "sct_path":     str(out_path),
        "hu_min":       round(float(sct_hu.min()), 2),
        "hu_max":       round(float(sct_hu.max()), 2),
        "hu_mean_brain": round(float(sct_hu[brain_mask].mean()), 2),
        "checkpoint":   str(ckpt_path),
    }

    print(f"Saved: {out_path}")
    print(f"HU range: [{result['hu_min']}, {result['hu_max']}]")
    print(f"HU mean (brain): {result['hu_mean_brain']} HU")

    if log_path:
        log_path = Path(log_path)
        existing = []
        if log_path.exists():
            with open(log_path) as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    pass
        existing.append(result)
        with open(log_path, "w") as f:
            json.dump(existing, f, indent=2)

    return str(out_path)
