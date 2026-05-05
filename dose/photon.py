import numpy as np
from scipy.ndimage import gaussian_filter


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
