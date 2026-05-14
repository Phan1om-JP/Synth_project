import os
import sys
import shutil
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import load_config
from preprocessing.preprocess import (
    load_or_compute_stats, scan_patients,
    preprocess_mr, postprocess_ct, pad_to_target
)
from models.unet import UNet2D, UNet3D


def load_generator(ckpt_path, cfg):
    """Returns (model, diffusion_or_None). diffusion is only set for architecture=diffusion."""
    mode    = cfg["data"]["spatial_mode"]
    in_ch   = cfg["model"]["in_ch"]
    out_ch  = cfg["model"]["out_ch"]
    base_ch = cfg["model"]["base_ch"]
    arch    = cfg["model"].get("architecture", "unet")

    diffusion = None
    if arch == "diffusion":
        from models.diffusion import DiffusionUNet, GaussianDiffusion
        model = DiffusionUNet(in_ch=2, base_ch=base_ch)
        diffusion = GaussianDiffusion(
            timesteps=cfg["diffusion"]["timesteps"],
            beta_schedule=cfg["diffusion"]["beta_schedule"],
        )
    elif mode in ("2D", "2.5D"):
        n_in  = cfg["data"]["n_adjacent"] if mode == "2.5D" else in_ch
        model = UNet2D(n_in, out_ch, base_ch)
    elif mode == "3D":
        model = UNet3D(in_ch, out_ch, base_ch)
    else:
        raise ValueError(f"Unknown spatial_mode: {mode}")

    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, diffusion


def infer_patient_2d(model, mr_vol, mask_vol, stats, cfg, device, diffusion=None):
    target_size = cfg["preprocessing"]["target_size"]
    n_ddim      = cfg.get("diffusion", {}).get("inference_steps", 50)
    mr_proc     = preprocess_mr(mr_vol, mask=mask_vol)
    sct_norm    = np.zeros_like(mr_proc)

    for z in range(mr_proc.shape[2]):
        sl   = mr_proc[:, :, z]
        h, w = sl.shape
        pt   = (target_size - h) // 2
        pl   = (target_size - w) // 2
        inp  = torch.from_numpy(
            pad_to_target(sl, target_size, target_size)
        ).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            if diffusion is not None:
                out = diffusion.ddim_sample(model, inp, n_steps=n_ddim).squeeze().cpu().numpy()
            else:
                out = model(inp).squeeze().cpu().numpy()

        sct_norm[:, :, z] = out[pt:pt+h, pl:pl+w]

    return postprocess_ct(sct_norm, stats)


def infer_patient_25d(model, mr_vol, mask_vol, stats, cfg, device):
    target_size = cfg["preprocessing"]["target_size"]
    n_adjacent  = cfg["data"]["n_adjacent"]
    half        = n_adjacent // 2
    mr_proc     = preprocess_mr(mr_vol, mask=mask_vol)
    sct_norm    = np.zeros_like(mr_proc)
    n_z         = mr_proc.shape[2]

    for z in range(n_z):
        slices = []
        for offset in range(-half, half + 1):
            zi = np.clip(z + offset, 0, n_z - 1)
            sl = mr_proc[:, :, zi]
            h, w = sl.shape
            slices.append(pad_to_target(sl, target_size, target_size))

        pt  = (target_size - h) // 2
        pl  = (target_size - w) // 2
        inp = torch.from_numpy(np.stack(slices, axis=0)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            out = model(inp).squeeze().cpu().numpy()

        sct_norm[:, :, z] = out[pt:pt+h, pl:pl+w]

    return postprocess_ct(sct_norm, stats)


def infer_patient_3d(model, mr_vol, mask_vol, stats, cfg, device):
    mr_proc  = preprocess_mr(mr_vol, mask=mask_vol)
    orig_shape = mr_proc.shape

    def pad_to_mult(arr, mult=8):
        pads = [(0, (-s % mult)) for s in arr.shape]
        return np.pad(arr, pads, mode="reflect"), pads

    mr_pad, pads = pad_to_mult(mr_proc, mult=8)
    inp = torch.from_numpy(mr_pad).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(inp).squeeze().cpu().numpy()

    out_crop = out[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
    return postprocess_ct(out_crop, stats)


def run_inference(cfg_path="config/config.yaml", ckpt_path=None, split="train"):
    cfg   = load_config(cfg_path)
    stats = load_or_compute_stats(cfg)
    device = cfg["device"]

    if ckpt_path is None:
        ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pt")

    model, diffusion = load_generator(ckpt_path, cfg)
    model = model.to(device)
    arch  = cfg["model"].get("architecture", "unet")
    print(f"Model loaded from: {ckpt_path}  (arch={arch})")

    task       = cfg.get("task", "task1")
    input_file = "cbct.nii.gz" if task == "task2" else "mr.nii.gz"
    root       = cfg["paths"].get("task2_train" if task == "task2" else "task1_train",
                                  cfg["paths"]["task1_train"]) if split == "train" \
                 else cfg["paths"].get("task2_val" if task == "task2" else "task1_val",
                                       cfg["paths"]["task1_val"])
    require_ct = split == "train"
    patients   = scan_patients(root, require_ct=require_ct)
    mode       = cfg["data"]["spatial_mode"]

    if arch == "diffusion" and mode != "2D":
        raise ValueError("Diffusion inference is only supported for spatial_mode=2D")

    for p in tqdm(patients, desc="Inference"):
        pid  = p["patient_id"]
        pdir = p["path"]

        out_dir = os.path.join(cfg["paths"]["output_dir"], pid)
        os.makedirs(out_dir, exist_ok=True)

        mr_nib   = nib.load(os.path.join(pdir, input_file))
        mask_nib = nib.load(os.path.join(pdir, "mask.nii.gz"))
        mr_vol   = mr_nib.get_fdata(dtype=np.float32)
        mask_vol = mask_nib.get_fdata(dtype=np.float32)

        if mode == "2D":
            sct = infer_patient_2d(model, mr_vol, mask_vol, stats, cfg, device,
                                   diffusion=diffusion)
        elif mode == "2.5D":
            sct = infer_patient_25d(model, mr_vol, mask_vol, stats, cfg, device)
        elif mode == "3D":
            sct = infer_patient_3d(model, mr_vol, mask_vol, stats, cfg, device)
        else:
            raise ValueError(f"Unknown spatial_mode: {mode}")

        sct_nib = nib.Nifti1Image(sct, mr_nib.affine, mr_nib.header)
        nib.save(sct_nib, os.path.join(out_dir, "sct.nii.gz"))

        for fname in ["mr.nii.gz", "ct.nii.gz", "mask.nii.gz"]:
            src = os.path.join(pdir, fname)
            dst = os.path.join(out_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    print(f"Inference complete. Outputs in: {cfg['paths']['output_dir']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--ckpt",   default=None)
    parser.add_argument("--split",  default="train", choices=["train", "val"])
    args = parser.parse_args()
    run_inference(args.config, args.ckpt, args.split)