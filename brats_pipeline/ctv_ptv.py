"""
ctv_ptv.py
CTV and PTV generation from GTV mask for BraTS-MEN-RT.

CTV = GTV dilated by ctv_margin_mm  (accounts for microscopic tumor spread)
PTV = CTV dilated by ptv_margin_mm  (accounts for setup uncertainty)

Dilation is performed in mm space using an ellipsoidal kernel so it handles
non-isotropic voxel spacing correctly.

Usage in Colab:
    from ctv_ptv import run_patient_ctv_ptv, verify_ctv_ptv
    Define paths/config in cells, call run_patient_ctv_ptv() per patient.
"""

import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Default margins (mm) — meningioma conventional fractionation
# Adjust per clinical protocol.
# ---------------------------------------------------------------------------
DEFAULT_CTV_MARGIN_MM = 3.0   # GTV → CTV
DEFAULT_PTV_MARGIN_MM = 5.0   # CTV → PTV

# Volume expansion ratio thresholds (from your pipeline QC plan)
EXPANSION_THRESHOLDS = {
    "ctv_gtv": {"normal": (1.1, 1.5), "warning_above": 2.0},
    "ptv_ctv": {"normal": (1.1, 1.4), "warning_above": 1.8},
}


# ---------------------------------------------------------------------------
# Dilation
# ---------------------------------------------------------------------------

def _make_ellipsoid_kernel(margin_mm, spacing_mm):
    """
    Build a 3D ellipsoidal structuring element for dilation.
    Each axis radius = margin_mm / voxel_size_on_that_axis.
    This ensures the dilation corresponds to the same physical distance
    regardless of voxel spacing.
    """
    spacing_mm = np.array(spacing_mm, dtype=float)
    radii = margin_mm / spacing_mm          # radius in voxels per axis
    r_max = int(np.ceil(radii.max()))

    grid = np.mgrid[
        -r_max : r_max + 1,
        -r_max : r_max + 1,
        -r_max : r_max + 1,
    ]
    # Ellipsoid equation: sum((xi / ri)^2) <= 1
    dist_sq = sum((grid[i] / radii[i]) ** 2 for i in range(3))
    return dist_sq <= 1.0


def dilate_mask(binary_mask, spacing_mm, margin_mm):
    """Dilate a binary mask by margin_mm (physical mm, not voxels)."""
    kernel = _make_ellipsoid_kernel(margin_mm, spacing_mm)
    return ndi.binary_dilation(binary_mask, structure=kernel).astype(np.uint8)


# ---------------------------------------------------------------------------
# Volume helpers
# ---------------------------------------------------------------------------

def mask_volume_cc(mask, spacing_mm):
    voxel_cc = float(np.prod(spacing_mm)) / 1000.0
    return float(mask.sum()) * voxel_cc


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_expansion(gtv_mask, ctv_mask, ptv_mask, spacing_mm):
    """
    Check volume expansion ratios against expected ranges.
    Returns a dict with volumes and pass/warn/fail per ratio.
    """
    gtv_cc  = mask_volume_cc(gtv_mask,  spacing_mm)
    ctv_cc  = mask_volume_cc(ctv_mask,  spacing_mm)
    ptv_cc  = mask_volume_cc(ptv_mask,  spacing_mm)

    def _status(ratio, key, gtv_cc):
        # Expected ratio scales with margin/radius — small tumors
        # have higher surface/volume ratio so proportional expansion is larger.
        # For GTVs < 20cc, upper bound is relaxed to 2.5× (CTV/GTV) / 3.0× (PTV/CTV).
        lo, hi = EXPANSION_THRESHOLDS[key]["normal"]
        warn   = EXPANSION_THRESHOLDS[key]["warning_above"]
        if gtv_cc < 20.0:
            hi   = hi   * 1.8   # relax upper normal bound for small tumors
            warn = warn * 1.8
        if ratio > warn:
            return "warn_large"
        elif lo <= ratio <= hi:
            return "pass"
        else:
            return "outside_normal"

    ctv_gtv_ratio = ctv_cc / gtv_cc if gtv_cc > 0 else None
    ptv_ctv_ratio = ptv_cc / ctv_cc if ctv_cc > 0 else None

    return {
        "gtv_cc":        round(gtv_cc,  2),
        "ctv_cc":        round(ctv_cc,  2),
        "ptv_cc":        round(ptv_cc,  2),
        "ctv_gtv_ratio": round(ctv_gtv_ratio, 3) if ctv_gtv_ratio else None,
        "ptv_ctv_ratio": round(ptv_ctv_ratio, 3) if ptv_ctv_ratio else None,
        "ctv_gtv_status": _status(ctv_gtv_ratio, "ctv_gtv", gtv_cc) if ctv_gtv_ratio else "error",
        "ptv_ctv_status": _status(ptv_ctv_ratio, "ptv_ctv", gtv_cc) if ptv_ctv_ratio else "error",
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def verify_ctv_ptv(t1c_path, gtv_path, ctv_path, ptv_path, n_slices=6):
    """
    Overlay GTV / CTV / PTV on T1c axial slices.
    GTV = red, CTV = green (ring), PTV = blue (ring).
    Good result: concentric expansions centered on tumor.
    """
    t1c = nib.load(str(t1c_path)).get_fdata()
    gtv = nib.load(str(gtv_path)).get_fdata().astype(bool)
    ctv = nib.load(str(ctv_path)).get_fdata().astype(bool)
    ptv = nib.load(str(ptv_path)).get_fdata().astype(bool)

    spacing = nib.load(str(t1c_path)).header.get_zooms()[:3]
    voxel_cc = float(np.prod(spacing)) / 1000.0

    print(f"  GTV: {gtv.sum() * voxel_cc:.1f} cc")
    print(f"  CTV: {ctv.sum() * voxel_cc:.1f} cc  (ratio {ctv.sum()/gtv.sum():.2f}x)")
    print(f"  PTV: {ptv.sum() * voxel_cc:.1f} cc  (ratio {ptv.sum()/gtv.sum():.2f}x vs GTV)")

    # Rings for CTV and PTV (subtract inner from outer for cleaner visualization)
    ctv_ring = ctv & ~gtv
    ptv_ring = ptv & ~ctv

    # Find slices where GTV exists
    z_idx = np.where(gtv)[2]
    if len(z_idx) == 0:
        print("GTV is empty — nothing to show")
        return

    z_samples = np.linspace(z_idx.min(), z_idx.max(), n_slices, dtype=int)
    vmin = float(np.percentile(t1c, 1))
    vmax = float(np.percentile(t1c, 99))

    fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3))
    for i, z in enumerate(z_samples):
        ax = axes[i]
        ax.imshow(t1c[:, :, z].T, cmap="gray", origin="lower",
                  vmin=vmin, vmax=vmax)
        ax.imshow(np.ma.masked_where(~gtv[:, :, z].T,     # GTV — red
                  np.ones_like(t1c[:, :, z].T)),
                  alpha=0.55, cmap="Reds",   origin="lower", vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(~ctv_ring[:, :, z].T,  # CTV ring — green
                  np.ones_like(t1c[:, :, z].T)),
                  alpha=0.35, cmap="Greens", origin="lower", vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(~ptv_ring[:, :, z].T,  # PTV ring — blue
                  np.ones_like(t1c[:, :, z].T)),
                  alpha=0.25, cmap="Blues",  origin="lower", vmin=0, vmax=1)
        ax.set_title(f"z={z}", fontsize=8)
        ax.axis("off")

    fig.suptitle("GTV (red)  CTV (green)  PTV (blue)", fontsize=10)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_results(patient_id, result, log_path):
    entry = {"patient_id": patient_id,
             "timestamp": datetime.utcnow().isoformat(),
             **result}
    log_path = Path(log_path)
    existing = []
    if log_path.exists():
        with open(log_path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    existing.append(entry)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)


# ---------------------------------------------------------------------------
# Main per-patient entry point
# ---------------------------------------------------------------------------

def run_patient_ctv_ptv(patient_id, t1c_path, gtv_path, out_dir, log_path,
                         ctv_margin_mm=DEFAULT_CTV_MARGIN_MM,
                         ptv_margin_mm=DEFAULT_PTV_MARGIN_MM):
    """
    Generate CTV and PTV for one patient.

    Args:
        patient_id     : string ID, used for output filenames
        t1c_path       : path to T1c NIfTI (used for affine/spacing)
        gtv_path       : path to GTV mask NIfTI
        out_dir        : output directory
        log_path       : JSON log file (appended)
        ctv_margin_mm  : GTV → CTV dilation in mm (default 3mm)
        ptv_margin_mm  : CTV → PTV dilation in mm (default 5mm)

    Returns dict with paths and validation metrics.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    t1c_img    = nib.load(str(t1c_path))
    gtv_img    = nib.load(str(gtv_path))
    spacing_mm = tuple(float(x) for x in t1c_img.header.get_zooms()[:3])

    gtv_mask = gtv_img.get_fdata().astype(bool)

    if gtv_mask.sum() == 0:
        print(f"[CTV/PTV] WARNING: GTV is empty for {patient_id}")

    # Generate CTV and PTV
    ctv_mask = dilate_mask(gtv_mask, spacing_mm, ctv_margin_mm)
    ptv_mask = dilate_mask(ctv_mask, spacing_mm, ptv_margin_mm)

    # Save — reuse T1c affine/header so everything stays aligned
    ctv_path = Path(out_dir) / f"{patient_id}_ctv.nii.gz"
    ptv_path = Path(out_dir) / f"{patient_id}_ptv.nii.gz"

    nib.save(nib.Nifti1Image(ctv_mask.astype(np.uint8), t1c_img.affine), str(ctv_path))
    nib.save(nib.Nifti1Image(ptv_mask.astype(np.uint8), t1c_img.affine), str(ptv_path))

    # Validate
    validation = validate_expansion(gtv_mask, ctv_mask, ptv_mask, spacing_mm)
    validation["ctv_margin_mm"] = ctv_margin_mm
    validation["ptv_margin_mm"] = ptv_margin_mm

    # Print summary
    print(f"[CTV/PTV] {patient_id}")
    print(f"  GTV  {validation['gtv_cc']:.1f} cc")
    print(f"  CTV  {validation['ctv_cc']:.1f} cc  (×{validation['ctv_gtv_ratio']}  [{validation['ctv_gtv_status']}])")
    print(f"  PTV  {validation['ptv_cc']:.1f} cc  (×{validation['ptv_ctv_ratio']}  [{validation['ptv_ctv_status']}])")

    result = {
        "ctv_path":   str(ctv_path),
        "ptv_path":   str(ptv_path),
        "validation": validation,
    }
    _log_results(patient_id, result, log_path)

    return result
