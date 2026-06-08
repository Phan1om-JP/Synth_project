import os
import sys
import csv
import time
import random
from collections import deque

import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import load_config
from preprocessing.preprocess import (
    load_or_compute_stats, build_cache,
    load_or_compute_stats_task2, build_cache_task2,
)
from dataloader.dataset import build_dataloaders
from models.gan import build_model
from models.losses import masked_l1, gan_generator_loss, gan_discriminator_loss, residual_skip

# Official SynthRAD2023 evaluation range (matches official-metrics repo)
_HU_MIN = -1024.0
_HU_MAX =  3000.0
_DR     = _HU_MAX - _HU_MIN   # 4024 HU — fixed population dynamic range


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fmt_seconds(s):
    s = int(s)
    h, m = divmod(s, 3600)
    m, s = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_validation(model, loader, stats, device, diffusion=None, n_ddim_steps=50,
                   use_residual=False):
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            if batch is None:
                continue
            inp, ct, mask = batch
            inp, ct, mask = inp.to(device), ct.to(device), mask.to(device)
            if diffusion is not None:
                pred = diffusion.ddim_sample(model, inp, n_steps=n_ddim_steps)
            else:
                pred = model(inp)
            if use_residual:
                pred = residual_skip(pred, inp, stats)

            ct_std  = stats["ct_global_std"]
            ct_mean = stats["ct_global_mean"]

            # Clip to official evaluation range for comparable metrics
            pred_np = np.clip(pred.cpu().numpy()[:, 0] * ct_std + ct_mean, _HU_MIN, _HU_MAX)
            ct_np   = np.clip(ct.cpu().numpy()[:, 0]   * ct_std + ct_mean, _HU_MIN, _HU_MAX)
            mask_np = mask.cpu().numpy()[:, 0]

            for b in range(pred_np.shape[0]):
                m = mask_np[b] > 0
                if m.sum() == 0:
                    continue
                mae  = np.abs(pred_np[b][m] - ct_np[b][m]).mean()
                # Set outside-mask to HU_MIN (matches official masking approach)
                p_sl = pred_np[b].copy(); p_sl[~m] = _HU_MIN
                c_sl = ct_np[b].copy();   c_sl[~m] = _HU_MIN
                mae_list.append(mae)
                ssim_list.append(ssim_fn(c_sl, p_sl, data_range=_DR))
                psnr_list.append(psnr_fn(c_sl, p_sl, data_range=_DR))

    if not mae_list:
        print("  [WARN] Validation produced no valid batches — skipping metric update.")
        return None
    return {
        "mae" : float(np.mean(mae_list)),
        "ssim": float(np.mean(ssim_list)),
        "psnr": float(np.mean(psnr_list)),
    }


def train(cfg_path="config/config.yaml", resume_path=None):
    cfg = load_config(cfg_path)
    set_seed(cfg["project"]["seed"])

    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["cache_dir"],      exist_ok=True)

    task  = cfg.get("task", "task1")
    stats = load_or_compute_stats_task2(cfg) if task == "task2" else load_or_compute_stats(cfg)
    if cfg["data"]["spatial_mode"] in ("2D", "2.5D") and not resume_path:
        if task == "task2":
            build_cache_task2(cfg, stats)
        else:
            build_cache(cfg, stats)

    tr_loader, hold_loader, _, _ = build_dataloaders(cfg, stats)
    generator, discriminator     = build_model(cfg)

    device       = cfg["device"]
    arch         = cfg["model"].get("architecture", "unet")
    loss_type    = cfg["training"]["loss_type"]
    use_residual = cfg["model"].get("residual", False)
    lr         = cfg["training"]["lr"]
    epochs     = cfg["training"]["epochs"]
    val_every  = cfg["training"]["val_every"]
    save_every = cfg["training"]["save_every"]
    patience   = cfg["training"]["early_stopping_patience"]
    gan_lambda = cfg["training"]["gan_lambda"]
    ckpt_dir   = cfg["paths"]["checkpoint_dir"]

    # discriminator holds the GaussianDiffusion object when arch == "diffusion"
    diffusion = discriminator if arch == "diffusion" else None

    # --- Build optimizer (hybrid loss has learnable weight parameters) ---
    hybrid_loss_fn = None
    if loss_type == "hybrid":
        from models.losses import HybridLoss
        hybrid_loss_fn = HybridLoss().to(device)
        opt_G = torch.optim.Adam(
            list(generator.parameters()) + list(hybrid_loss_fn.parameters()),
            lr=lr, betas=(0.5, 0.999),
        )
    else:
        opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    opt_D = None
    if loss_type == "l1_gan" and discriminator is not None and arch != "diffusion":
        opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    history          = {"train_loss": [], "val_mae": [], "val_ssim": [], "val_psnr": []}
    best_val_mae     = float("inf")
    patience_counter = 0
    start_epoch      = 1
    best_ckpt_path   = os.path.join(ckpt_dir, "best_model.pt")

    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model_target = generator.module if hasattr(generator, "module") else generator
        model_target.load_state_dict(ckpt["model_state"])
        try:
            opt_G.load_state_dict(ckpt["opt_state"])
        except ValueError:
            print("  [WARN] Optimizer state skipped (loss type changed) — using fresh optimizer")
        start_epoch      = ckpt["epoch"] + 1
        history          = ckpt.get("history", history)
        best_val_mae     = ckpt.get("best_val_mae", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        if hybrid_loss_fn is not None and "loss_fn_state" in ckpt:
            hybrid_loss_fn.load_state_dict(ckpt["loss_fn_state"])
        print(f"  Resumed at epoch {start_epoch} | best MAE so far: {best_val_mae:.4f}")
    elif resume_path:
        print(f"  [WARN] Resume checkpoint not found: {resume_path} — starting fresh")

    # --- CSV metrics logger ---
    log_path = os.path.join(ckpt_dir, "metrics_log.csv")
    if not os.path.isfile(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss",
                                    "val_mae", "val_ssim", "val_psnr"])

    epoch_times = deque(maxlen=5)
    train_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        generator.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in tqdm(tr_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            if batch is None:
                continue
            inp, ct, mask = batch
            inp, ct, mask = inp.to(device), ct.to(device), mask.to(device)
            n_batches += 1

            if arch == "diffusion":
                t    = torch.randint(0, diffusion.T, (inp.size(0),), device=device)
                loss = diffusion.p_losses(generator, ct, inp, t)
                opt_G.zero_grad()
                loss.backward()
                opt_G.step()

            elif loss_type == "l1":
                pred = generator(inp)
                if use_residual:
                    pred = residual_skip(pred, inp, stats)
                loss = masked_l1(pred, ct, mask)
                opt_G.zero_grad()
                loss.backward()
                opt_G.step()

            elif loss_type == "hybrid":
                pred = generator(inp)
                if use_residual:
                    pred = residual_skip(pred, inp, stats)
                loss, _ = hybrid_loss_fn(pred, ct, mask)
                opt_G.zero_grad()
                loss.backward()
                opt_G.step()

            elif loss_type == "l1_gan":
                pred = generator(inp)
                if use_residual:
                    pred = residual_skip(pred, inp, stats)
                loss_l1    = masked_l1(pred, ct, mask)
                disc_fake  = discriminator(inp, pred)
                loss_g_gan = gan_generator_loss(disc_fake)
                loss       = loss_l1 + gan_lambda * loss_g_gan
                opt_G.zero_grad()
                loss.backward()
                opt_G.step()

                disc_real = discriminator(inp, ct)
                disc_fake = discriminator(inp, pred.detach())
                loss_d    = gan_discriminator_loss(disc_real, disc_fake)
                opt_D.zero_grad()
                loss_d.backward()
                opt_D.step()

            epoch_loss += loss.item()

        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - epoch
        eta = avg_epoch_time * remaining_epochs

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Track whether this epoch produced an improvement (for patience)
        improved = False

        ddim_steps = cfg.get("diffusion", {}).get("inference_steps", 50)
        if epoch % val_every == 0 or epoch == start_epoch:
            val_metrics = run_validation(generator, hold_loader, stats, device,
                                         diffusion=diffusion, n_ddim_steps=ddim_steps,
                                         use_residual=use_residual)

            if val_metrics is None:
                # All val batches were invalid — log train-only row and skip improvement check
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, f"{avg_loss:.6f}", "", "", ""])
                print(
                    f"Epoch {epoch:04d}/{epochs} | "
                    f"loss {avg_loss:.4f} | "
                    f"val SKIPPED (no valid batches) | "
                    f"epoch {epoch_elapsed:.0f}s | "
                    f"ETA {_fmt_seconds(eta)}"
                )
            else:
                history["val_mae"].append(val_metrics["mae"])
                history["val_ssim"].append(val_metrics["ssim"])
                history["val_psnr"].append(val_metrics["psnr"])

                print(
                    f"Epoch {epoch:04d}/{epochs} | "
                    f"loss {avg_loss:.4f} | "
                    f"MAE {val_metrics['mae']:.4f} | "
                    f"SSIM {val_metrics['ssim']:.4f} | "
                    f"PSNR {val_metrics['psnr']:.2f} | "
                    f"epoch {epoch_elapsed:.0f}s | "
                    f"ETA {_fmt_seconds(eta)}"
                )
                if loss_type == "hybrid" and hybrid_loss_fn is not None:
                    lv = hybrid_loss_fn.log_vars.detach().cpu()
                    w  = torch.exp(-lv.clamp(-4, 4))
                    print(f"  Adaptive weights — "
                          f"L1:{w[0]:.3f}  SSIM:{w[1]:.3f}  "
                          f"grad:{w[2]:.3f}  freq:{w[3]:.3f}")

                # Log to CSV (val epoch row)
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch, f"{avg_loss:.6f}",
                        f"{val_metrics['mae']:.4f}",
                        f"{val_metrics['ssim']:.4f}",
                        f"{val_metrics['psnr']:.4f}",
                    ])

                if val_metrics["mae"] < best_val_mae:
                    best_val_mae     = val_metrics["mae"]
                    patience_counter = 0
                    improved         = True
                    model_state = (generator.module.state_dict()
                                   if hasattr(generator, "module")
                                   else generator.state_dict())
                    ckpt_data = {
                        "epoch"           : epoch,
                        "model_state"     : model_state,
                        "opt_state"       : opt_G.state_dict(),
                        "history"         : history,
                        "best_val_mae"    : best_val_mae,
                        "patience_counter": 0,
                        "cfg"             : cfg,
                    }
                    if hybrid_loss_fn is not None:
                        ckpt_data["loss_fn_state"] = hybrid_loss_fn.state_dict()
                    torch.save(ckpt_data, best_ckpt_path)
                    print(f"  Best saved (MAE={best_val_mae:.4f})")
                else:
                    print(f"  No improvement. Patience: {patience_counter + 1}/{patience}")

        else:
            # Log to CSV (train-only row, no val metrics)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{avg_loss:.6f}", "", "", ""])

            print(
                f"Epoch {epoch:04d}/{epochs} | "
                f"loss {avg_loss:.4f} | "
                f"patience {patience_counter}/{patience} | "
                f"epoch {epoch_elapsed:.0f}s | "
                f"ETA {_fmt_seconds(eta)}"
            )

        # Patience is tracked per epoch (not per val event)
        if not improved:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        if epoch % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch:04d}.pt")
            model_state = (generator.module.state_dict()
                           if hasattr(generator, "module")
                           else generator.state_dict())
            ckpt_data = {
                "epoch"      : epoch,
                "model_state": model_state,
                "opt_state"  : opt_G.state_dict(),
                "history"    : history,
            }
            if hybrid_loss_fn is not None:
                ckpt_data["loss_fn_state"] = hybrid_loss_fn.state_dict()
            torch.save(ckpt_data, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    total_time = time.time() - train_start
    print(f"Training done. Best MAE: {best_val_mae:.4f} | Total time: {_fmt_seconds(total_time)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg",    nargs="?", default="config/config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args.cfg, resume_path=args.resume)
