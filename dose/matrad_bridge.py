import subprocess


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
