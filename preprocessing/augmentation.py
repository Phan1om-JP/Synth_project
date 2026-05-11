import numpy as np
from scipy.ndimage import rotate, map_coordinates, gaussian_filter


def sample_params(cfg):
    """Draw augmentation parameters once so they can be reused across slices."""
    aug = cfg.get("augmentation", {})
    params = {}

    if aug.get("flip", False):
        params["flip_h"] = np.random.rand() < 0.5
        params["flip_v"] = np.random.rand() < 0.5

    rot_max = aug.get("rotation_max", 0)
    params["angle"] = float(np.random.uniform(-rot_max, rot_max)) if rot_max > 0 else 0.0

    jitter = aug.get("intensity_jitter", 0.0)
    if jitter > 0:
        params["brightness"] = float(np.random.uniform(-jitter, jitter))
        params["contrast"]   = float(np.random.uniform(1 - jitter, 1 + jitter))
    else:
        params["brightness"] = 0.0
        params["contrast"]   = 1.0

    if aug.get("elastic", False):
        alpha = aug.get("elastic_alpha", 30)
        sigma = aug.get("elastic_sigma", 4)
        params["elastic_alpha"] = alpha
        params["elastic_sigma"] = sigma
        params["elastic_dx"]    = None   # generated lazily on first call (shape-dependent)
        params["elastic_dy"]    = None
        params["elastic_dz"]    = None
    else:
        params["elastic_alpha"] = 0

    return params


def _get_elastic_fields_2d(params, shape):
    if params["elastic_dx"] is None or params["elastic_dx"].shape != shape:
        sigma = params["elastic_sigma"]
        alpha = params["elastic_alpha"]
        params["elastic_dx"] = gaussian_filter(np.random.randn(*shape), sigma) * alpha
        params["elastic_dy"] = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    return params["elastic_dx"], params["elastic_dy"]


def _get_elastic_fields_3d(params, shape):
    if params["elastic_dz"] is None or params["elastic_dz"].shape != shape:
        sigma = params["elastic_sigma"]
        alpha = params["elastic_alpha"]
        params["elastic_dz"] = gaussian_filter(np.random.randn(*shape), sigma) * alpha
        params["elastic_dx"] = gaussian_filter(np.random.randn(*shape), sigma) * alpha
        params["elastic_dy"] = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    return params["elastic_dz"], params["elastic_dx"], params["elastic_dy"]


def _apply_spatial_2d(arr, params, order=1):
    """Apply flip + rotation + elastic to a 2D array."""
    if params.get("flip_h", False):
        arr = np.flip(arr, axis=0).copy()
    if params.get("flip_v", False):
        arr = np.flip(arr, axis=1).copy()
    if params["angle"] != 0.0:
        arr = rotate(arr, params["angle"], reshape=False, order=order, mode="nearest")
    if params["elastic_alpha"] > 0:
        dx, dy = _get_elastic_fields_2d(params, arr.shape)
        x, y   = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
        coords  = [np.clip(y + dy, 0, arr.shape[0] - 1).ravel(),
                   np.clip(x + dx, 0, arr.shape[1] - 1).ravel()]
        arr = map_coordinates(arr, coords, order=order, mode="reflect").reshape(arr.shape)
    return arr


def _apply_spatial_3d(arr, params, order=1):
    """Apply flip + rotation (axial plane) + elastic to a 3D array."""
    if params.get("flip_h", False):
        arr = np.flip(arr, axis=0).copy()
    if params.get("flip_v", False):
        arr = np.flip(arr, axis=1).copy()
    if params["angle"] != 0.0:
        arr = rotate(arr, params["angle"], axes=(0, 1), reshape=False,
                     order=order, mode="nearest")
    if params["elastic_alpha"] > 0:
        dz, dy, dx = _get_elastic_fields_3d(params, arr.shape)
        z, y, x    = np.mgrid[0:arr.shape[0], 0:arr.shape[1], 0:arr.shape[2]]
        coords = [np.clip(z + dz, 0, arr.shape[0] - 1).ravel(),
                  np.clip(y + dy, 0, arr.shape[1] - 1).ravel(),
                  np.clip(x + dx, 0, arr.shape[2] - 1).ravel()]
        arr = map_coordinates(arr, coords, order=order, mode="reflect").reshape(arr.shape)
    return arr


def apply_params(inp, ct, mask, params, is_3d=False):
    """
    Apply pre-sampled augmentation parameters to (inp, ct, mask) numpy arrays.
    inp:  (H,W) or (D,H,W) — input modality
    ct:   (H,W) or (D,H,W) — target CT
    mask: (H,W) or (D,H,W) — binary mask
    """
    spatial = _apply_spatial_3d if is_3d else _apply_spatial_2d

    inp  = spatial(inp,  params, order=1)
    ct   = spatial(ct,   params, order=1)
    mask = spatial(mask, params, order=0)

    # Intensity jitter on input only
    b = params["brightness"]
    c = params["contrast"]
    if b != 0.0 or c != 1.0:
        inp = np.clip(inp * c + b, inp.min(), inp.max())

    mask = (mask > 0.5).astype(np.float32)
    return inp.astype(np.float32), ct.astype(np.float32), mask


def apply_augmentation(inp, ct, mask, cfg, is_3d=False):
    """Convenience wrapper: sample params then apply. For 2D/3D single-input datasets."""
    if not cfg.get("augmentation", {}).get("enabled", False):
        return inp, ct, mask
    params = sample_params(cfg)
    return apply_params(inp, ct, mask, params, is_3d=is_3d)


def apply_augmentation_25d(mr_slices, ct, mask, cfg):
    """
    Augment 2.5D: same spatial transform across all MR context slices.
    mr_slices: list of (H,W) arrays (length = n_adjacent)
    ct:        (H,W)
    mask:      (H,W)
    Returns: (augmented_mr_slices, ct, mask)
    """
    aug = cfg.get("augmentation", {})
    if not aug.get("enabled", False):
        return mr_slices, ct, mask

    params = sample_params(cfg)
    # Apply same spatial transform to center CT + mask
    _, ct, mask = apply_params(mr_slices[len(mr_slices) // 2], ct, mask, params)
    # Apply spatial + per-slice independent intensity jitter to each MR slice
    out_slices = []
    for sl in mr_slices:
        sl = _apply_spatial_2d(sl, params, order=1)
        # Independent intensity jitter per slice (simulates scanner variation)
        jitter = aug.get("intensity_jitter", 0.0)
        if jitter > 0:
            b = float(np.random.uniform(-jitter, jitter))
            c = float(np.random.uniform(1 - jitter, 1 + jitter))
            sl = np.clip(sl * c + b, sl.min(), sl.max())
        out_slices.append(sl.astype(np.float32))
    return out_slices, ct, mask
