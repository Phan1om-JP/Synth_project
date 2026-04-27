import os
import subprocess
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy.special import erf


def hu_to_electron_density(hu_arr):
    rho = np.ones_like(hu_arr, dtype=np.float32)
    rho[hu_arr < -950] = 0.0
    m1 = (hu_arr >= -950) & (hu_arr < -700)
    rho[m1] = (hu_arr[m1] + 950) / 250 * 0.05
    m2 = (hu_arr >= -700) & (hu_arr < -100)
    rho[m2] = 0.05 + (hu_arr[m2] + 700) / 600 * 0.95
    m3 = (hu_arr >= -100) & (hu_arr < 100)
    rho[m3] = 1.0 + hu_arr[m3] / 1000.0 * 0.5
    m4 = (hu_arr >= 100) & (hu_arr < 1000)
    rho[m4] = 1.05 + (hu_arr[m4] - 100) / 900 * 0.6
    m5 = hu_arr >= 1000
    rho[m5] = 1.65 + (hu_arr[m5] - 1000) / 1000 * 0.3
    return np.clip(rho, 0, None)


def hu_to_stopping_power(hu_arr):
    rsp = np.ones_like(hu_arr, dtype=np.float32)
    rsp[hu_arr < -950] = 0.0
    m1 = (hu_arr >= -950) & (hu_arr < -120)
    rsp[m1] = 1.0 + hu_arr[m1] / 1000.0
    m2 = (hu_arr >= -120) & (hu_arr < 200)
    rsp[m2] = 1.0 + 0.5 * hu_arr[m2] / 1000.0
    m3 = hu_arr >= 200
    rsp[m3] = 1.13 + 0.568 * (hu_arr[m3] - 200) / 1000.0
    return np.clip(rsp, 0, None)


def compute_photon_dose(ct_arr, ptv_mask, spacing, beam_angles_deg, mu_eff=0.006):
    dose_accum = np.zeros_like(ct_arr, dtype=np.float32)
    rho        = hu_to_electron_density(ct_arr)
    ptv_center = np.array(np.argwhere(ptv_mask).mean(axis=0))
    sigma_vox  = 5.0 / spacing.mean()

    for angle_deg in beam_angles_deg:
        angle_rad = np.deg2rad(angle_deg)
        beam_dir  = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
        dom_axis  = int(np.argmax(np.abs(beam_dir[:2])))

        rad_depth = np.cumsum(rho * spacing[dom_axis], axis=dom_axis)
        cz        = int(ptv_center[dom_axis])
        idx       = [int(ptv_center[i]) for i in range(3)]
        ptv_depth = rad_depth[idx[0], idx[1], idx[2]]

        dose_beam = np.exp(-mu_eff * np.abs(rad_depth - ptv_depth)).astype(np.float32)
        sigma_arr = [sigma_vox if i != dom_axis else 0 for i in range(3)]
        dose_beam = gaussian_filter(dose_beam, sigma=sigma_arr)
        dose_accum += dose_beam

    ptv_max = dose_accum[ptv_mask].max()
    if ptv_max > 0:
        dose_accum /= ptv_max
    return dose_accum


def compute_proton_dose(ct_arr, ptv_mask, spacing, beam_angles_deg,
                        sigma_mm=6.0, distal_mm=4.0):
    dose_accum = np.zeros_like(ct_arr, dtype=np.float32)
    rsp        = hu_to_stopping_power(ct_arr)
    ptv_center = np.array(np.argwhere(ptv_mask).mean(axis=0)).astype(int)

    for angle_deg in beam_angles_deg:
        angle_rad = np.deg2rad(angle_deg)
        beam_dir  = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
        dom_axis  = int(np.argmax(np.abs(beam_dir[:2])))

        wed  = np.cumsum(rsp * spacing[dom_axis], axis=dom_axis)
        R0   = float(wed[ptv_center[0], ptv_center[1], ptv_center[2]])

        phi  = np.exp(-((wed - R0)**2) / (2 * sigma_mm**2))
        tail = 0.5 * (1 - erf((wed - R0) / (distal_mm * np.sqrt(2))))
        dose_beam = (phi + 0.05 * tail).astype(np.float32)
        dose_beam[rsp < 1e-6] = 0.0
        dose_accum += dose_beam

    ptv_max = dose_accum[ptv_mask].max()
    if ptv_max > 0:
        dose_accum /= ptv_max
    return dose_accum


def run_matrad_dose(ct_path, ptv_path, out_dose_path, matrad_path, modality="photon"):
    script = f"""
    addpath(genpath('{matrad_path}'));
    ct_nii = load_nii('{ct_path}');
    ptv_nii = load_nii('{ptv_path}');
    ct.cube = double(ct_nii.img);
    ct.resolution.x = ct_nii.hdr.dime.pixdim(2);
    ct.resolution.y = ct_nii.hdr.dime.pixdim(3);
    ct.resolution.z = ct_nii.hdr.dime.pixdim(4);
    cst = matRad_createCSTfromMask(ptv_nii.img, 'PTV');
    pln.radiationMode = '{modality}';
    pln.machine = 'Generic';
    pln.numOfFractions = 1;
    pln.propStf.gantryAngles = [0 90 180 270];
    pln.propStf.couchAngles = zeros(1,4);
    pln.propStf.bixelWidth = 5;
    pln.propStf.numOfBeams = 4;
    pln.propStf.isoCenter = matRad_getIsoCenter(cst, ct, 0);
    stf = matRad_generateStf(ct, cst, pln);
    dij = matRad_calcPhotonDose(ct, stf, pln, cst);
    resultGUI = matRad_fluenceOptimization(dij, cst, pln);
    dose = resultGUI.physicalDose;
    save_nii(make_nii(single(dose)), '{out_dose_path}');
    exit;
    """
    script_path = out_dose_path.replace(".nii.gz", "_matrad.m")
    with open(script_path, "w") as f:
        f.write(script)

    result = subprocess.run(
        ["octave", "--no-gui", "--quiet", script_path],
        capture_output=True, text=True, timeout=3600
    )
    if result.returncode != 0:
        raise RuntimeError(f"matRad failed:\n{result.stderr[-500:]}")
    return result.returncode


def compute_and_save_dose(patient_out_dir, cfg, modality="photon"):
    ct_path   = os.path.join(patient_out_dir, "ct.nii.gz")
    sct_path  = os.path.join(patient_out_dir, "sct.nii.gz")
    ptv_path  = os.path.join(patient_out_dir, "structures", "PTV_thalamus.nii.gz")

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
    angles_key = f"{modality}_beam_angles"
    angles     = cfg["dose"][angles_key]

    for tag, arr, nib_ref in [("ct", ct_arr, ct_nib), ("sct", sct_arr, sct_nib)]:
        out_path = os.path.join(patient_out_dir, f"dose_{tag}_{modality}.nii.gz")

        if use_matrad:
            src_path = ct_path if tag == "ct" else sct_path
            run_matrad_dose(
                src_path, ptv_path, out_path,
                cfg["dose"]["matrad_path"], modality
            )
        else:
            if modality == "photon":
                dose = compute_photon_dose(arr, ptv_mask, spacing, angles)
            else:
                dose = compute_proton_dose(arr, ptv_mask, spacing, angles)

            nib.save(nib.Nifti1Image(dose, nib_ref.affine, nib_ref.header), out_path)

        print(f"  Dose saved: {out_path}")