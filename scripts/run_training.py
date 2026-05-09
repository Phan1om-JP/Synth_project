import os
import sys
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
from preprocessing.preprocess import load_or_compute_stats, build_cache
from dataloader.dataset import build_dataloaders
from models.gan import build_model
from models.losses import masked_l1, gan_generator_loss, gan_discriminator_loss


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


def run_validation(model, loader, stats, device):
    model.eval()
    mae_list, ssim_list, psnr_list = [], [], []

    with torch.no_grad():
        for mr, ct, mask in tqdm(loader, desc="Val", leave=False):
            mr, ct, mask = mr.to(device), ct.to(device), mask.to(device)
            pred = model(mr)

            ct_std  = stats["ct_global_std"]
            ct_mean = stats["ct_global_mean"]
            ct_min  = stats["ct_clip_min"]
            ct_max  = stats["ct_clip_max"]

            pred_np = np.clip(pred.cpu().numpy()[:, 0] * ct_std + ct_mean, ct_min, ct_max)
            ct_np   = np.clip(ct.cpu().numpy()[:, 0]   * ct_std + ct_mean, ct_min, ct_max)
            mask_np = mask.cpu().numpy()[:, 0]

            for b in range(pred_np.shape[0]):
                m = mask_np[b] > 0
                if m.sum() == 0:
                    continue
                mae        = np.abs(pred_np[b][m] - ct_np[b][m]).mean()
                data_range = ct_np[b][m].max() - ct_np[b][m].min()
                p_sl       = pred_np[b].copy(); p_sl[~m] = 0
                c_sl       = ct_np[b].copy();   c_sl[~m] = 0
                mae_list.append(mae)
                ssim_list.append(ssim_fn(c_sl, p_sl, data_range=float(data_range)))
                psnr_list.append(psnr_fn(c_sl, p_sl, data_range=float(data_range)))

    return {
        "mae" : float(np.mean(mae_list))  if mae_list  else 0.0,
        "ssim": float(np.mean(ssim_list)) if ssim_list else 0.0,
        "psnr": float(np.mean(psnr_list)) if psnr_list else 0.0,
    }


def train(cfg_path="config/config.yaml"):
    cfg = load_config(cfg_path)
    set_seed(cfg["project"]["seed"])

    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["cache_dir"],      exist_ok=True)

    stats = load_or_compute_stats(cfg)
    if cfg["data"]["spatial_mode"] in ("2D", "2.5D"):
        build_cache(cfg, stats)

    tr_loader, hold_loader, _, _ = build_dataloaders(cfg, stats)
    generator, discriminator     = build_model(cfg)

    device     = cfg["device"]
    loss_type  = cfg["training"]["loss_type"]
    lr         = cfg["training"]["lr"]
    epochs     = cfg["training"]["epochs"]
    val_every  = cfg["training"]["val_every"]
    save_every = cfg["training"]["save_every"]
    patience   = cfg["training"]["early_stopping_patience"]
    gan_lambda = cfg["training"]["gan_lambda"]
    ckpt_dir   = cfg["paths"]["checkpoint_dir"]

    opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = None
    if loss_type == "l1_gan" and discriminator is not None:
        opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    history          = {"train_loss": [], "val_mae": [], "val_ssim": [], "val_psnr": []}
    best_val_mae     = float("inf")
    patience_counter = 0
    best_ckpt_path   = os.path.join(ckpt_dir, "best_model.pt")

    epoch_times = deque(maxlen=5)
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        generator.train()
        epoch_loss = 0.0

        for mr, ct, mask in tqdm(tr_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            mr, ct, mask = mr.to(device), ct.to(device), mask.to(device)
            pred = generator(mr)

            if loss_type == "l1":
                loss = masked_l1(pred, ct, mask)
                opt_G.zero_grad()
                loss.backward()
                opt_G.step()

            elif loss_type == "l1_gan":
                loss_l1    = masked_l1(pred, ct, mask)
                disc_fake  = discriminator(mr, pred)
                loss_g_gan = gan_generator_loss(disc_fake)
                loss       = loss_l1 + gan_lambda * loss_g_gan
                opt_G.zero_grad()
                loss.backward()
                opt_G.step()

                disc_real = discriminator(mr, ct)
                disc_fake = discriminator(mr, pred.detach())
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

        avg_loss = epoch_loss / len(tr_loader)
        history["train_loss"].append(avg_loss)

        if epoch % val_every == 0 or epoch == 1:
            val_metrics = run_validation(generator, hold_loader, stats, device)
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

            if val_metrics["mae"] < best_val_mae:
                best_val_mae     = val_metrics["mae"]
                patience_counter = 0
                model_state = (generator.module.state_dict()
                               if hasattr(generator, "module")
                               else generator.state_dict())
                torch.save({
                    "epoch"       : epoch,
                    "model_state" : model_state,
                    "opt_state"   : opt_G.state_dict(),
                    "history"     : history,
                    "best_val_mae": best_val_mae,
                    "cfg"         : cfg,
                }, best_ckpt_path)
                print(f"  Best saved (MAE={best_val_mae:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break
        else:
            print(
                f"Epoch {epoch:04d}/{epochs} | "
                f"loss {avg_loss:.4f} | "
                f"epoch {epoch_elapsed:.0f}s | "
                f"ETA {_fmt_seconds(eta)}"
            )

        if epoch % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch:04d}.pt")
            model_state = (generator.module.state_dict()
                           if hasattr(generator, "module")
                           else generator.state_dict())
            torch.save({
                "epoch"      : epoch,
                "model_state": model_state,
                "opt_state"  : opt_G.state_dict(),
                "history"    : history,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    total_time = time.time() - train_start
    print(f"Training done. Best MAE: {best_val_mae:.4f} | Total time: {_fmt_seconds(total_time)}")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    train(cfg_path)
