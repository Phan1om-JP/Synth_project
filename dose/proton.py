import numpy as np
from scipy.special import erf


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

        phi       = np.exp(-((wed - R0)**2) / (2 * sigma_mm**2))
        tail      = 0.5 * (1 - erf((wed - R0) / (distal_mm * np.sqrt(2))))
        dose_beam = (phi + 0.05 * tail).astype(np.float32)
        dose_beam[rsp < 1e-6] = 0.0
        dose_accum += dose_beam

    ptv_max = dose_accum[ptv_mask].max()
    if ptv_max > 0:
        dose_accum /= ptv_max
    return dose_accum
