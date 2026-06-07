"""
oar_segmentation.py
OAR segmentation for BraTS-MEN-RT using 3 methods:
  A) SynthSeg  (FreeSurfer DL, any MRI contrast)
  B) CerebrA   (ANTs atlas propagation, MNI152)
  C) TotalSegmentator (DL, pip-installable)

Usage in Colab:
    from oar_segmentation import run_patient, compute_population_volume_stats
    Define paths/config in cells, call run_patient() per patient.
"""

import os
import json
import shutil
import subprocess
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ---------------------------------------------------------------------------
# Label maps
# Each method has its own integer IDs for the same anatomical structure.
# Structures here are the ones relevant for brain RT planning.
# ---------------------------------------------------------------------------

# SynthSeg uses FreeSurfer label IDs.
SYNTHSEG_LABELS = {
    "brainstem":        16,
    "cerebellum_left":   8,
    "cerebellum_right": 47,
    "thalamus_left":    10,
    "thalamus_right":   49,
    "hippocampus_left": 17,
    "hippocampus_right": 53,
}

# Harvard-Oxford subcortical atlas labels (via nilearn).
# These IDs are fixed — verified against FSL atlas documentation.
# Replaces CerebrA: same RT-relevant structures, cleaner download.
HO_LABELS = {
    "brainstem":         8,   # whole posterior fossa — oversized (~60cc), use TotalSeg for precise contour
    "thalamus_left":     4,
    "thalamus_right":   15,
    "hippocampus_left":  9,
    "hippocampus_right": 19,  # was 20 (Right Amygdala) — fixed to 19 (Right Hippocampus)
    # cerebellum: not in Harvard-Oxford subcortical atlas — requires TotalSeg brain_structures license
}

# TotalSegmentator brain task structure names (v2).
# File names produced by TotalSegmentator, mapped to merged label IDs.
TOTALSEG_BRAIN_FILES = {
    "brainstem":        1,
    "cerebellum":       2,   # no L/R split in brain_structures task
    "thalamus":         3,   # no L/R split — use atlas for L/R lateralization
    "frontal_lobe":     4,
    "temporal_lobe":    5,
    "parietal_lobe":    6,
    "occipital_lobe":   7,
    "caudate_nucleus":  8,
    "lentiform_nucleus": 9,
    "ventricle":        10,
    "internal_capsule": 11,
    "insular_cortex":   12,
}

# Common structure names used for cross-method comparison.
# Maps common name → per-method label ID.
COMMON_STRUCTURES = {
    "brainstem": {
        "synthseg": 16,
        "atlas":     8,
        "totalseg":  1,
    },
    "thalamus_left": {
        "synthseg": 10,
        "atlas":     4,
        "totalseg": None,
    },
    "thalamus_right": {
        "synthseg": 49,
        "atlas":    15,
        "totalseg": None,
    },
    "hippocampus_left": {
        "synthseg": 17,
        "atlas":     9,
        "totalseg": None,
    },
    "hippocampus_right": {
        "synthseg": 53,
        "atlas":    19,
        "totalseg": None,
    },
    # cerebellum: atlas method cannot provide this — TotalSeg brain_structures only
    "cerebellum_left": {
        "synthseg":  8,
        "atlas":    None,
        "totalseg":  2,
    },
    "cerebellum_right": {
        "synthseg": 47,
        "atlas":    None,
        "totalseg":  2,
    },
}

# Literature-based expected volume ranges (cc).
# Sources: RTOG 0933 contouring atlas, Brouwer et al. 2015 (Radiother Oncol),
#          Feng et al. 2019 (Int J Radiat Oncol), standard anatomy references.
# These are used as a PRIOR. Derive empirical ranges from your cohort and
# compare — that double-validation is what you show at defense.
OAR_VOLUME_RANGES_CC = {
    "brainstem":         (15.0,  45.0),
    "cerebellum":       (100.0, 220.0),  # merged L+R from TotalSeg
    "thalamus_left":     ( 5.0,  15.0),
    "thalamus_right":    ( 5.0,  15.0),
    "hippocampus_left":  ( 1.5,   5.0),
    "hippocampus_right": ( 1.5,   5.0),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_patient(t1c_path, gtv_path):
    """Load T1c and GTV, reorient both to RAS canonical."""
    t1c_img = nib.as_closest_canonical(nib.load(str(t1c_path)))
    gtv_img = nib.as_closest_canonical(nib.load(str(gtv_path)))
    return t1c_img, gtv_img


def check_alignment(t1c_img, gtv_img):
    """Return alignment QC dict. Spacing deviation > 0.01 mm triggers alert."""
    t1c_spacing = np.array(t1c_img.header.get_zooms()[:3])
    gtv_spacing = np.array(gtv_img.header.get_zooms()[:3])
    diff = np.abs(t1c_spacing - gtv_spacing)

    return {
        "t1c_shape":            list(t1c_img.shape),
        "gtv_shape":            list(gtv_img.shape),
        "t1c_spacing_mm":       t1c_spacing.tolist(),
        "gtv_spacing_mm":       gtv_spacing.tolist(),
        "spacing_deviation_mm": diff.tolist(),
        "spacing_ok":           bool(np.all(diff <= 0.01)),
        "shape_match":          t1c_img.shape == gtv_img.shape,
    }


# ---------------------------------------------------------------------------
# Method A: SynthSeg
# ---------------------------------------------------------------------------

def _numpy_compat_patch_code(synthseg_dir):
    """Return Python source that patches numpy and adds synthseg_dir to sys.path."""
    return (
        "import sys, numpy as np\n"
        f"sys.path.insert(0, r'{synthseg_dir}')\n"
        "for _k, _v in [('int', np.int64), ('float', np.float64),\n"
        "               ('bool', np.bool_), ('complex', np.complex128)]:\n"
        "    if not hasattr(np, _k): setattr(np, _k, _v)\n"
    )


def _patch_numpy_in_process():
    """Restore removed np.int/np.float aliases that SynthSeg 1.0 requires."""
    for k, v in [('int', np.int64), ('float', np.float64),
                 ('bool', np.bool_), ('complex', np.complex128)]:
        if not hasattr(np, k):
            setattr(np, k, v)


def segment_oar_synthseg(t1c_path, out_dir, patient_id,
                         synthseg_dir="/content/SynthSeg"):
    """
    Run SynthSeg on T1c image. Designed for any MRI contrast including T1c.
    SynthSeg 1.0 uses removed np.int aliases — patched automatically.
    Tries three paths in order:
      1. FreeSurfer CLI (mri_synthseg)
      2. Installed Python package (in-process, numpy patched)
      3. Locally cloned repo via temp wrapper script
    Returns (seg_path_or_None, status_string).
    """
    import tempfile
    out_path = Path(out_dir) / f"{patient_id}_synthseg.nii.gz"

    # Try 1: FreeSurfer CLI
    try:
        r = subprocess.run(
            ["mri_synthseg", "--i", str(t1c_path), "--o", str(out_path),
             "--threads", "4"],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode == 0 and out_path.exists():
            return str(out_path), "synthseg_cli"
        print(f"[SynthSeg CLI] {r.stderr[-200:]}")
    except FileNotFoundError:
        print("[SynthSeg CLI] mri_synthseg not in PATH")
    except subprocess.TimeoutExpired:
        print("[SynthSeg CLI] Timeout")

    # Try 2: Python API — patch numpy and add path first, then import
    try:
        _patch_numpy_in_process()
        import sys as _sys
        _sys.path.insert(0, str(Path(synthseg_dir).resolve()))
        from SynthSeg.predict import predict
        predict(path_images=str(t1c_path), path_segm=str(out_path))
        if out_path.exists():
            return str(out_path), "synthseg_python"
    except ImportError as e:
        print(f"[SynthSeg Python] Import failed: {e}")
    except Exception as e:
        print(f"[SynthSeg Python] {patient_id}: {e}")

    # Try 3: Subprocess via temp wrapper (patches numpy before exec)
    predict_script = Path(synthseg_dir) / "scripts/commands/SynthSeg_predict.py"
    if predict_script.exists():
        wrapper = (
            _numpy_compat_patch_code(synthseg_dir)
            + f"sys.argv = ['p','--i',r'{t1c_path}','--o',r'{out_path}','--v1']\n"
            # runpy.run_path sets __file__ correctly so SynthSeg's path
            # calculations (os.path.dirname(__file__)) resolve to the repo root
            # instead of '/' which exec() would produce.
            + f"import runpy; runpy.run_path(r'{predict_script}', run_name='__main__')\n"
        )
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        tmp.write(wrapper); tmp.flush(); tmp.close()
        try:
            r = subprocess.run(
                ["python", tmp.name],
                capture_output=True, text=True, timeout=900,
            )
            combined = (r.stdout + r.stderr).strip()
            if r.returncode == 0 and out_path.exists():
                return str(out_path), "synthseg_local"
            print(f"[SynthSeg local] exit={r.returncode}\n{combined[-600:]}")
        except subprocess.TimeoutExpired:
            print(f"[SynthSeg local] Timeout for {patient_id}")
        except Exception as e:
            print(f"[SynthSeg local] {e}")
        finally:
            os.unlink(tmp.name)
    else:
        print(f"[SynthSeg] Repo not found at {synthseg_dir}. "
              f"Run: git clone https://github.com/BBillot/SynthSeg.git {synthseg_dir}")

    return None, "synthseg_failed"


# ---------------------------------------------------------------------------
# Method B: CerebrA atlas via ANTs
# ---------------------------------------------------------------------------

def download_ho_atlas(atlas_dir):
    """
    Download Harvard-Oxford subcortical atlas + MNI152 template via nilearn.
    Returns (template_path, labels_path).
    Both files are cached locally after first download.
    Requires: pip install nilearn
    """
    from nilearn import datasets, image
    atlas_dir = Path(atlas_dir)
    atlas_dir.mkdir(parents=True, exist_ok=True)

    template_cache = atlas_dir / "mni152_t1_1mm.nii.gz"
    labels_cache   = atlas_dir / "ho_sub_maxprob_thr25_1mm.nii.gz"

    if not template_cache.exists():
        print("Downloading MNI152 template ...")
        mni = datasets.load_mni152_template(resolution=1)
        nib.save(mni, str(template_cache))

    if not labels_cache.exists():
        print("Downloading Harvard-Oxford subcortical atlas ...")
        ho = datasets.fetch_atlas_harvard_oxford(
            "sub-maxprob-thr25-1mm", data_dir=str(atlas_dir)
        )
        # ho.maps is a path string in older nilearn, Nifti1Image in newer — handle both
        if isinstance(ho.maps, str):
            import shutil as _sh
            _sh.copy(ho.maps, str(labels_cache))
        else:
            nib.save(ho.maps, str(labels_cache))

    return str(template_cache), str(labels_cache)


def segment_oar_atlas(t1c_path, out_dir, patient_id, atlas_dir):
    """
    ANTs SyNRA registration: patient T1c → MNI152, then propagate
    Harvard-Oxford subcortical labels back to patient space.
    Returns (seg_path_or_None, status_string).
    Requires: pip install antspyx nilearn
    """
    try:
        import ants
    except ImportError:
        print("[Atlas] antspyx not installed. Run: pip install antspyx")
        return None, "atlas_no_ants"

    out_seg = Path(out_dir) / f"{patient_id}_atlas.nii.gz"
    tx_dir  = Path(out_dir) / f"{patient_id}_atlas_tx"
    tx_dir.mkdir(parents=True, exist_ok=True)

    try:
        tmpl_path, lbl_path = download_ho_atlas(atlas_dir)

        t1c  = ants.image_read(str(t1c_path))
        tmpl = ants.image_read(tmpl_path)
        lbl  = ants.image_read(lbl_path)

        print(f"[Atlas] Registering {patient_id} → MNI152 ...")
        reg = ants.registration(
            fixed=t1c,
            moving=tmpl,
            type_of_transform="SyNRA",
            verbose=False,
        )

        warped = ants.apply_transforms(
            fixed=t1c,
            moving=lbl,
            transformlist=reg["fwdtransforms"],
            interpolator="nearestNeighbor",
        )
        ants.image_write(warped, str(out_seg))

        # Save transforms — inverse can warp dose back to MNI for population analysis
        for i, tx in enumerate(reg["fwdtransforms"]):
            shutil.copy(tx, tx_dir / f"fwd_{i}{Path(tx).suffix}")
        for i, tx in enumerate(reg["invtransforms"]):
            shutil.copy(tx, tx_dir / f"inv_{i}{Path(tx).suffix}")

        print(f"[Atlas] Done {patient_id}")
        return str(out_seg), "atlas_ants_ho"

    except Exception as e:
        print(f"[Atlas] Failed {patient_id}: {e}")
        return None, "atlas_failed"


# ---------------------------------------------------------------------------
# Method C: TotalSegmentator
# ---------------------------------------------------------------------------

def segment_oar_totalseg(t1c_path, out_dir, patient_id):
    """
    TotalSegmentator brain task.
    Install: pip install TotalSegmentator
    Returns (seg_path_or_None, status_string).

    NOTE: runs WITHOUT --ml so it writes one file per structure.
    _merge_totalseg_labels() then combines them into a single label volume.
    """
    seg_dir = Path(out_dir) / f"{patient_id}_totalseg_raw"
    out_seg = Path(out_dir) / f"{patient_id}_totalseg.nii.gz"
    seg_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[TotalSeg] Running {patient_id} ...")
        r = subprocess.run(
            ["TotalSegmentator",
             "-i", str(t1c_path),
             "-o", str(seg_dir),
             "--task", "brain_structures"],
            capture_output=True, text=True, timeout=900,
        )
        # TotalSegmentator writes progress to stdout, errors to either stream
        combined = (r.stdout + r.stderr).strip()
        if r.returncode != 0:
            raise RuntimeError(combined[-800:] if combined else "empty output")

        # Show what files were actually created (helps debug label name mismatches)
        created = [f.name for f in seg_dir.iterdir() if f.suffix == ".gz"]
        print(f"[TotalSeg] Created {len(created)} files: {created[:5]} ...")

        merged = _merge_totalseg_labels(seg_dir, TOTALSEG_BRAIN_FILES)
        nib.save(merged, str(out_seg))
        print(f"[TotalSeg] Done {patient_id}")
        return str(out_seg), "totalseg"

    except FileNotFoundError:
        print("[TotalSeg] Not installed. Run: pip install TotalSegmentator")
        return None, "totalseg_not_installed"
    except subprocess.TimeoutExpired:
        print(f"[TotalSeg] Timeout for {patient_id}")
        return None, "totalseg_timeout"
    except Exception as e:
        print(f"[TotalSeg] Failed {patient_id}: {e}")
        return None, "totalseg_failed"


def _merge_totalseg_labels(seg_dir, label_map):
    """Merge per-structure NIfTI files into one integer label volume."""
    ref = None
    merged = None
    for fname, label_id in label_map.items():
        fpath = Path(seg_dir) / f"{fname}.nii.gz"
        if not fpath.exists():
            continue
        img = nib.load(str(fpath))
        data = img.get_fdata()
        if ref is None:
            ref = img
            merged = np.zeros(img.shape, dtype=np.int16)
        merged[data > 0.5] = label_id

    if ref is None:
        raise RuntimeError(f"No TotalSegmentator output files found in {seg_dir}")
    return nib.Nifti1Image(merged, ref.affine, ref.header)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def compute_structure_volumes(seg_path, label_map, spacing_mm):
    """
    Compute structure volumes in cc from a label map.
    label_map: dict of {structure_name: integer_label_id}
    spacing_mm: (dx, dy, dz) from img.header.get_zooms()
    """
    seg = nib.load(str(seg_path)).get_fdata()
    voxel_cc = float(np.prod(spacing_mm)) / 1000.0  # mm³ → cc

    return {
        name: round(float(np.sum(seg == label_id)) * voxel_cc, 3)
        for name, label_id in label_map.items()
    }


def validate_oar_volumes(volumes, expected_ranges=None):
    """
    Check computed volumes against expected_ranges.
    Status: pass / warn / fail_too_small / fail_too_large / no_range
    """
    if expected_ranges is None:
        expected_ranges = OAR_VOLUME_RANGES_CC

    report = {}
    for name, vol in volumes.items():
        if name not in expected_ranges:
            report[name] = {"volume_cc": vol, "status": "no_range"}
            continue
        lo, hi = expected_ranges[name]
        if vol < lo * 0.3:
            status = "fail_too_small"
        elif vol > hi * 2.0:
            status = "fail_too_large"
        elif vol < lo or vol > hi:
            status = "warn"
        else:
            status = "pass"
        report[name] = {
            "volume_cc": vol,
            "expected_lo_cc": lo,
            "expected_hi_cc": hi,
            "status": status,
        }
    return report


def compute_dice(seg_a_path, seg_b_path, label_a, label_b=None):
    """
    Dice coefficient between two binary masks extracted from segmentation files.
    label_b defaults to label_a (use when both methods use same ID — they don't here,
    so always pass both explicitly).
    """
    if label_b is None:
        label_b = label_a
    a = nib.load(str(seg_a_path)).get_fdata() == label_a
    b = nib.load(str(seg_b_path)).get_fdata() == label_b
    intersection = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return None  # structure absent in both — skip, don't count as agreement
    return round(float(2.0 * intersection / denom), 4)


def compare_methods(seg_paths):
    """
    Pairwise Dice across all method pairs for every common structure.
    seg_paths: dict of {method_name: path_or_None}
    Uses COMMON_STRUCTURES for label ID translation.
    """
    active = {m: p for m, p in seg_paths.items() if p is not None}
    names  = list(active.keys())
    results = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            m_a, m_b = names[i], names[j]
            pair = f"{m_a}_vs_{m_b}"
            results[pair] = {}
            for struct, method_labels in COMMON_STRUCTURES.items():
                label_a = method_labels.get(m_a)
                label_b = method_labels.get(m_b)
                if label_a is None or label_b is None:
                    results[pair][struct] = None
                    continue
                results[pair][struct] = compute_dice(
                    active[m_a], active[m_b], label_a, label_b
                )
    return results


# ---------------------------------------------------------------------------
# Empirical threshold derivation (run after processing a batch)
# ---------------------------------------------------------------------------

def compute_population_volume_stats(all_volumes):
    """
    Derive data-driven thresholds from a population of per-patient volume dicts.

    all_volumes: list of dicts returned by compute_structure_volumes()

    Returns per-structure stats including mean ± 3σ and p5/p95.
    Use these alongside OAR_VOLUME_RANGES_CC to cross-validate at defense:
      - If empirical mean agrees with literature → method is trustworthy
      - If they diverge → quantify the gap and explain why (domain shift, etc.)
    """
    per_struct = defaultdict(list)
    for vol_dict in all_volumes:
        for name, vol in vol_dict.items():
            if vol > 0:
                per_struct[name].append(vol)

    stats = {}
    for name, vals in per_struct.items():
        arr = np.array(vals)
        m, s = arr.mean(), arr.std()
        stats[name] = {
            "n":           int(len(arr)),
            "mean_cc":     round(float(m), 3),
            "std_cc":      round(float(s), 3),
            "lo_3sigma":   round(float(m - 3 * s), 3),
            "hi_3sigma":   round(float(m + 3 * s), 3),
            "p5_cc":       round(float(np.percentile(arr, 5)), 3),
            "p95_cc":      round(float(np.percentile(arr, 95)), 3),
            "literature":  OAR_VOLUME_RANGES_CC.get(name),
        }
    return stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def verify_segmentation(t1c_path, seg_path, label_map, n_slices=5):
    """
    Plot each labeled structure overlaid on T1c axial slices.
    Good result: structure appears in anatomically correct location.
    Bad result: overlay on skull/background, or volume wildly off.
    """
    t1c = nib.load(str(t1c_path)).get_fdata()
    seg = nib.load(str(seg_path)).get_fdata()
    spacing = nib.load(str(t1c_path)).header.get_zooms()[:3]
    voxel_cc = float(np.prod(spacing)) / 1000.0

    for name, label_id in label_map.items():
        mask = (seg == label_id)
        vol_cc = float(mask.sum()) * voxel_cc
        z_idx = np.where(mask)[2]

        if len(z_idx) == 0:
            print(f"  {name}: NOT FOUND (label {label_id})")
            continue

        print(f"  {name}: {vol_cc:.1f} cc  (z range {z_idx.min()}–{z_idx.max()})")

        z_samples = np.linspace(z_idx.min(), z_idx.max(), n_slices, dtype=int)
        fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3))
        vmin = float(np.percentile(t1c, 1))
        vmax = float(np.percentile(t1c, 99))
        for i, z in enumerate(z_samples):
            axes[i].imshow(t1c[:, :, z].T, cmap="gray", origin="lower",
                           vmin=vmin, vmax=vmax)
            axes[i].imshow(mask[:, :, z].T, alpha=0.45, cmap="Reds", origin="lower")
            axes[i].set_title(f"z={z}", fontsize=8)
            axes[i].axis("off")
        fig.suptitle(f"{name}  ({vol_cc:.1f} cc)", fontsize=10)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# JSON logging
# ---------------------------------------------------------------------------

def log_patient_results(patient_id, result_dict, log_path):
    """Append one patient's full result to a JSON array log file."""
    entry = {"patient_id": patient_id,
             "timestamp": datetime.utcnow().isoformat(),
             **result_dict}

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

def run_patient(patient_id, t1c_path, gtv_path, out_dir, atlas_dir, log_path,
                methods=("synthseg", "atlas", "totalseg")):
    """
    Full OAR segmentation pipeline for one patient.
    Call this from a Colab cell.

    Args:
        patient_id  : string ID, used for output filenames
        t1c_path    : path to T1c NIfTI
        gtv_path    : path to GTV mask NIfTI
        out_dir     : directory for segmentation outputs
        atlas_dir   : directory where CerebrA atlas will be downloaded/cached
        log_path    : path to JSON log file (appended, not overwritten)
        methods     : tuple of methods to run; remove any you want to skip

    Returns dict with keys: alignment, seg_paths, volume_reports, comparison
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    t1c_img, gtv_img = load_patient(t1c_path, gtv_path)
    spacing_mm = tuple(float(x) for x in t1c_img.header.get_zooms()[:3])

    alignment = check_alignment(t1c_img, gtv_img)
    if not alignment["spacing_ok"]:
        print(f"[WARN] {patient_id}: voxel spacing mismatch {alignment['spacing_deviation_mm']}")
    if not alignment["shape_match"]:
        print(f"[WARN] {patient_id}: T1c/GTV shape mismatch — check registration")

    seg_paths = {}
    seg_status = {}

    if "synthseg" in methods:
        seg_paths["synthseg"], seg_status["synthseg"] = \
            segment_oar_synthseg(t1c_path, out_dir, patient_id)

    if "atlas" in methods:
        seg_paths["atlas"], seg_status["atlas"] = \
            segment_oar_atlas(t1c_path, out_dir, patient_id, atlas_dir)

    if "totalseg" in methods:
        seg_paths["totalseg"], seg_status["totalseg"] = \
            segment_oar_totalseg(t1c_path, out_dir, patient_id)

    # Per-method volume validation
    volume_reports = {}
    method_label_maps = {
        "synthseg": SYNTHSEG_LABELS,
        "atlas":    HO_LABELS,
        "totalseg": TOTALSEG_BRAIN_FILES,
    }
    for method, path in seg_paths.items():
        if path is None:
            continue
        lmap = method_label_maps.get(method, {})
        vols = compute_structure_volumes(path, lmap, spacing_mm)
        volume_reports[method] = validate_oar_volumes(vols)

    # Cross-method Dice comparison
    comparison = compare_methods(seg_paths)

    result = {
        "alignment":       alignment,
        "seg_status":      seg_status,
        "seg_paths":       seg_paths,
        "volume_reports":  volume_reports,
        "comparison_dice": comparison,
    }
    log_patient_results(patient_id, result, log_path)

    return result
