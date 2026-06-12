"""
Diffusion model training runner for Google Colab (T4 GPU).

Setup — paste each block into a separate Colab cell:

    # Cell 1: mount Drive (checkpoints + data live here)
    from google.colab import drive
    drive.mount('/content/drive')

    # Cell 2: clone repo and install deps
    !git clone https://github.com/Phan1om-JP/Synth_project.git
    %cd Synth_project
    !pip install -r requirement.txt

    # Cell 3: edit CONFIG below, then run
    !python scripts/run_colab_diffusion.py

    # To resume after a session timeout:
    !python scripts/run_colab_diffusion.py --resume /content/drive/MyDrive/synthrad/ckpt/diffusion_task1/best_model.pt

Data layout expected on Google Drive (Task 1 default):
    MyDrive/synthrad/Task1/Task1/brain/{patient_id}/mr.nii.gz
    MyDrive/synthrad/Task1/Task1/brain/{patient_id}/ct.nii.gz
    MyDrive/synthrad/Task1/Task1/brain/{patient_id}/mask.nii.gz
    MyDrive/synthrad/Task1_val/Task1/brain/{patient_id}/...

For Task 2 swap the commented blocks in CONFIG below.
"""

import os
import sys
import argparse
import tempfile
import yaml

# =============================================================================
# CONFIG — only section you need to edit
# =============================================================================
CONFIG = {
    # ── "task1" (MRI→CT) | "task2" (CBCT→CT) ────────────────────────────────
    "task": "task1",

    "project": {
        "name": "synthrad_diffusion_colab",
        "seed": 15,
        "output_dir": "outputs_colab",
    },

    "paths": {
        # ── Task 1 paths — data downloaded by scripts/download_synthrad.py ───
        "task1_train": "/content/storage/Task1/Task1/brain",
        "task1_val":   "/content/storage/Task1_val/Task1/brain",
        # stats computed once and cached; delete to force recompute
        "stats_path":  "/content/storage/stats_task1.json",

        # ── Task 2 paths — uncomment and set task: task2 above ───────────────
        # "task2_train": "/content/storage/Task2/Task2/brain",
        # "task2_val":   "/content/storage/Task2_val/Task2/brain",
        # "stats_path":  "/content/storage/stats_task2.json",

        # all under /content/storage/ — fast local SSD, lost on session end
        # save checkpoints to Drive if you want them to survive timeouts:
        #   "checkpoint_dir": "/content/drive/MyDrive/synthrad/ckpt/diffusion_task1",
        "cache_dir":      "/content/storage/cache",
        "checkpoint_dir": "/content/storage/checkpoints/diffusion_task1",
        "output_dir":     "/content/storage/outputs/diffusion_task1",
    },

    "preprocessing": {
        # ── Task 1 (MRI→CT) ──────────────────────────────────────────────────
        "ct_clip_min": -1000.0,
        "ct_clip_max":  1000.0,

        # ── Task 2 only — uncomment if task: task2 ───────────────────────────
        # "cbct_clip_min": -1024.0,
        # "cbct_clip_max":  3000.0,

        "min_mask_coverage": 0.10,
        "target_size": 320,
    },

    "data": {
        "plane":           "axial",
        "spatial_mode":    "2D",   # diffusion only supports 2D
        "n_adjacent":      3,
        "patch_size":      96,
        "train_val_split": 0.9,
        "batch_size":      8,      # safe for T4 16 GB; try 16 if memory allows
        "num_workers":     2,      # Colab has 2 CPUs; not NFS so workers are fine
        "pin_memory":      True,
    },

    "model": {
        "architecture": "diffusion",
        "in_ch":  2,    # noisy CT (1ch) + condition MR/CBCT (1ch) — do not change
        "out_ch": 1,
        "base_ch": 64,
        # residual skip does not apply to diffusion — omit
    },

    "diffusion": {
        "timesteps":        1000,
        "beta_schedule":    "cosine",
        "inference_steps":  50,    # DDIM steps at val/inference
    },

    "augmentation": {
        "enabled":         True,
        "flip":            True,
        "rotation_max":    10,
        "intensity_jitter": 0.1,
        "elastic":         True,
        "elastic_alpha":   30,
        "elastic_sigma":   4,
    },

    "training": {
        "loss_type":               "l1",    # ignored for diffusion; kept for compat
        "lr":                      1e-4,
        "epochs":                  500,     # colab session ~12h; resume if needed
        "val_every":               5,
        "save_every":              5,       # save often so Drive has recent ckpt
        "early_stopping_patience": 30,
        "gan_lambda":              0.1,
        "multi_gpu":               False,   # Colab = 1 GPU
    },
}
# =============================================================================


def _write_config(cfg: dict) -> str:
    path = os.path.join(tempfile.gettempdir(), "config_colab_diffusion.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return path


def _check_paths(cfg: dict):
    task = cfg.get("task", "task1")
    train_key = "task2_train" if task == "task2" else "task1_train"
    train_root = cfg["paths"].get(train_key, "")
    if not os.path.isdir(train_root):
        print(f"\n[ERROR] Training data not found: {train_root}")
        print("  Make sure Google Drive is mounted and the path is correct.")
        sys.exit(1)
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Data root : {train_root}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Cache     : {cfg['paths']['cache_dir']}  (local SSD)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    _check_paths(CONFIG)
    cfg_path = _write_config(CONFIG)
    print(f"Config    : {cfg_path}\n")

    from scripts.run_training import train
    train(cfg_path, resume_path=args.resume)


if __name__ == "__main__":
    main()
