import os
import numpy as np
import nibabel as nib

from dose.photon import compute_photon_dose
from dose.proton import compute_proton_dose
from dose.matrad_bridge import run_matrad_dose


def compute_and_save_dose(patient_out_dir, cfg, modality="photon"):
    ct_path  = os.path.join(patient_out_dir, "ct.nii.gz")
    sct_path = os.path.join(patient_out_dir, "sct.nii.gz")
    ptv_path = os.path.join(patient_out_dir, "structures", "PTV_thalamus.nii.gz")

    if not os.path.exists(ptv_path):
        raise FileNotFoundError(f"PTV not found: {ptv_path}")

    ct_nib   = nib.load(ct_path)
    sct_nib  = nib.load(sct_path)
    ptv_nib  = nib.load(ptv_path)

    ct_arr   = ct_nib.get_fdata(dtype=np.float32)
    sct_arr  = sct_nib.get_fdata(dtype=np.float32)
    ptv_mask = ptv_nib.get_fdata() > 0
    spacing  = np.array(ct_nib.header.get_zooms())

    use_matrad = cfg["dose"]["use_matrad"]
    angles     = cfg["dose"][f"{modality}_beam_angles"]

    for tag, arr, nib_ref in [("ct", ct_arr, ct_nib), ("sct", sct_arr, sct_nib)]:
        out_path = os.path.join(patient_out_dir, f"dose_{tag}_{modality}.nii.gz")

        if use_matrad:
            src_path = ct_path if tag == "ct" else sct_path
            run_matrad_dose(src_path, ptv_path, out_path,
                            cfg["dose"]["matrad_path"], modality)
        else:
            if modality == "photon":
                dose = compute_photon_dose(arr, ptv_mask, spacing, angles)
            else:
                dose = compute_proton_dose(arr, ptv_mask, spacing, angles)
            nib.save(nib.Nifti1Image(dose, nib_ref.affine, nib_ref.header), out_path)

        print(f"  Dose saved: {out_path}")
