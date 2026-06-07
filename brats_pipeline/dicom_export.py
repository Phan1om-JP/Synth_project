"""
dicom_export.py
Convert sCT NIfTI and mask NIfTIs to DICOM CT + RTSTRUCT for matRad.

Steps:
  1. sct_to_dicom_series() : sCT NIfTI -> DICOM CT folder (one .dcm per slice)
  2. create_rtstruct()     : NIfTI masks -> DICOM RTSTRUCT referencing CT folder

Call run_patient_dicom_export() which runs both.

Note on orientation: SimpleITK writes DICOM in LPS convention.
rt-utils reads the DICOM geometry headers to place contours correctly.
After export, verify alignment in matRad by checking GTV contour overlaps CT.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime


# Structure display colors (R, G, B)
STRUCTURE_COLORS = {
    "GTV":           [255,   0,   0],
    "CTV":           [  0, 220,   0],
    "PTV":           [  0,   0, 255],
    "Brainstem":     [255, 165,   0],
    "Thalamus_L":    [160,  32, 240],
    "Thalamus_R":    [160,  32, 240],
    "Hippocampus_L": [  0, 200, 200],
    "Hippocampus_R": [  0, 200, 200],
}


# ---------------------------------------------------------------------------
# Step 1: sCT NIfTI -> DICOM CT series
# ---------------------------------------------------------------------------

def sct_to_dicom_series(sct_path, out_dir, patient_id):
    """
    Write one DICOM .dcm file per axial slice.
    Uses SimpleITK which handles the NIfTI RAS -> DICOM LPS conversion.
    Returns (output_directory, series_instance_uid).
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("Run: pip install SimpleITK")

    from pydicom.uid import generate_uid

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img      = sitk.Cast(sitk.ReadImage(str(sct_path)), sitk.sitkInt16)
    n_slices = img.GetDepth()
    spacing  = img.GetSpacing()       # (sx, sy, sz) in mm
    origin   = img.GetOrigin()        # (x0, y0, z0) of voxel (0,0,0) in LPS mm
    dirn     = img.GetDirection()     # 9-element row-major direction matrix

    # dirn layout: [d00,d01,d02, d10,d11,d12, d20,d21,d22]
    # Physical column of z-axis = (dirn[2], dirn[5], dirn[8])
    # Image Position Patient for slice i = origin + i * sz * z_col
    z_col = (dirn[2], dirn[5], dirn[8])

    # Image Orientation Patient: row cosines = x-axis col, col cosines = y-axis col
    iop = (f"{dirn[0]}\\{dirn[3]}\\{dirn[6]}\\"
           f"{dirn[1]}\\{dirn[4]}\\{dirn[7]}")

    study_uid  = generate_uid()
    series_uid = generate_uid()
    frame_uid  = generate_uid()
    now        = datetime.now()

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    for i in range(n_slices):
        slc = img[:, :, i]

        # Compute the physical position of this slice's first pixel
        ipp_x = origin[0] + i * spacing[2] * z_col[0]
        ipp_y = origin[1] + i * spacing[2] * z_col[1]
        ipp_z = origin[2] + i * spacing[2] * z_col[2]

        slc.SetMetaData("0008|0060", "CT")
        slc.SetMetaData("0008|0008", "DERIVED\\SECONDARY")
        slc.SetMetaData("0008|103e", "sCT SynthRAD")
        slc.SetMetaData("0008|0020", now.strftime("%Y%m%d"))
        slc.SetMetaData("0008|0030", now.strftime("%H%M%S"))
        slc.SetMetaData("0010|0010", patient_id)
        slc.SetMetaData("0010|0020", patient_id)
        slc.SetMetaData("0020|000d", study_uid)
        slc.SetMetaData("0020|000e", series_uid)
        slc.SetMetaData("0020|0052", frame_uid)
        slc.SetMetaData("0020|0013", str(i + 1))
        slc.SetMetaData("0020|0032",              # Image Position Patient
                        f"{ipp_x}\\{ipp_y}\\{ipp_z}")
        slc.SetMetaData("0020|0037", iop)         # Image Orientation Patient
        slc.SetMetaData("0020|1041", str(ipp_z))  # Slice Location
        slc.SetMetaData("0018|0050", f"{spacing[2]:.4f}")
        slc.SetMetaData("0028|0030",
                        f"{spacing[0]:.4f}\\{spacing[1]:.4f}")
        slc.SetMetaData("0028|1052", "0")
        slc.SetMetaData("0028|1053", "1")
        slc.SetMetaData("0028|1054", "HU")

        writer.SetFileName(str(out_dir / f"CT_{i:04d}.dcm"))
        writer.Execute(slc)

    print(f"DICOM CT: {n_slices} slices written to {out_dir}")
    return str(out_dir), series_uid


# ---------------------------------------------------------------------------
# Step 2: NIfTI masks -> RTSTRUCT
# ---------------------------------------------------------------------------

def create_rtstruct(dicom_ct_dir, structure_masks, out_path):
    """
    Create DICOM RTSTRUCT from a dict of {name: bool_3d_numpy_array}.
    The masks must be in the same voxel space as the sCT NIfTI.
    Requires: pip install rt-utils
    """
    try:
        from rt_utils import RTStructBuilder
    except ImportError:
        raise ImportError("Run: pip install rt-utils")

    rtstruct = RTStructBuilder.create_new(
        dicom_series_path=str(dicom_ct_dir)
    )

    for name, mask in structure_masks.items():
        if mask is None or mask.sum() == 0:
            print(f"  Skipping {name} — empty mask")
            continue
        color = STRUCTURE_COLORS.get(name, [200, 200, 200])
        rtstruct.add_roi(
            mask=mask.astype(bool),
            color=color,
            name=name,
        )
        vol_cc = mask.sum() * 1.0 / 1000.0   # rough cc, assumes 1mm voxels
        print(f"  Added {name}: {vol_cc:.1f} cc")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rtstruct.save(str(out_path))
    print(f"RTSTRUCT saved: {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Main per-patient entry point
# ---------------------------------------------------------------------------

def run_patient_dicom_export(
    patient_id,
    sct_path,
    gtv_path,
    ctv_path,
    ptv_path,
    atlas_seg_path,
    out_dir,
):
    """
    Full DICOM export pipeline for one patient.

    Args:
        patient_id     : string ID used for DICOM metadata and filenames
        sct_path       : sCT NIfTI from sct_inference.run_patient_sct()
        gtv_path       : GTV mask NIfTI from BraTS
        ctv_path       : CTV mask NIfTI from ctv_ptv.run_patient_ctv_ptv()
        ptv_path       : PTV mask NIfTI from ctv_ptv.run_patient_ctv_ptv()
        atlas_seg_path : OAR atlas label map from oar_segmentation.run_patient()
        out_dir        : root output folder; creates DICOM_CT/ subfolder

    Returns dict with paths to DICOM CT dir and RTSTRUCT file.
    """
    from oar_segmentation import HO_LABELS

    out_dir = Path(out_dir)
    ct_dir  = out_dir / "DICOM_CT"

    # Step 1
    ct_dir_str, _ = sct_to_dicom_series(sct_path, ct_dir, patient_id)

    # Step 2 — load masks via SimpleITK so axis ordering matches the DICOM CT
    # (SimpleITK handles NIfTI RAS -> DICOM LPS internally, nibabel does not)
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("Run: pip install SimpleITK")

    ref_img = sitk.ReadImage(str(sct_path))

    def _load_sitk(path):
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        # SimpleITK gives (z, y, x); rt-utils expects (row, col, slice) = (y, x, z)
        return arr.transpose(1, 2, 0).astype(bool)

    def _load_label_sitk(path, label_id):
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        return (arr.transpose(1, 2, 0) == label_id)

    # rt-utils expects mask as (z, y, x) — matching DICOM slice/row/col order
    masks = {
        "GTV": _load_sitk(gtv_path),
        "CTV": _load_sitk(ctv_path),
        "PTV": _load_sitk(ptv_path),
    }

    if atlas_seg_path and Path(atlas_seg_path).exists():
        masks["Brainstem"]     = _load_label_sitk(atlas_seg_path, HO_LABELS["brainstem"])
        masks["Thalamus_L"]    = _load_label_sitk(atlas_seg_path, HO_LABELS["thalamus_left"])
        masks["Thalamus_R"]    = _load_label_sitk(atlas_seg_path, HO_LABELS["thalamus_right"])
        masks["Hippocampus_L"] = _load_label_sitk(atlas_seg_path, HO_LABELS["hippocampus_left"])
        masks["Hippocampus_R"] = _load_label_sitk(atlas_seg_path, HO_LABELS["hippocampus_right"])
    else:
        print("Atlas seg not found — exporting GTV/CTV/PTV only")

    # Body: largest connected component above -500 HU → clean patient outline.
    # Simple threshold includes padding artifacts and sCT boundary noise;
    # keeping only the largest CC eliminates those scattered voxels.
    from scipy import ndimage as _ndi
    sct_arr  = sitk.GetArrayFromImage(sitk.ReadImage(str(sct_path)))
    body_raw = sct_arr.transpose(1, 2, 0) > -500
    labeled, n_cc = _ndi.label(body_raw)
    if n_cc > 0:
        counts      = np.bincount(labeled.ravel())
        counts[0]   = 0
        body_mask   = labeled == counts.argmax()
        body_mask   = _ndi.binary_fill_holes(body_mask)
    else:
        body_mask   = body_raw
    masks["Body"] = body_mask

    rtstruct_path = str(out_dir / f"{patient_id}_rtstruct.dcm")
    create_rtstruct(ct_dir_str, masks, rtstruct_path)

    result = {
        "dicom_ct_dir":  ct_dir_str,
        "rtstruct_path": rtstruct_path,
    }
    print(f"\nDICOM export done for {patient_id}")
    print(f"  CT dir  : {ct_dir_str}")
    print(f"  RTSTRUCT: {rtstruct_path}")
    return result
