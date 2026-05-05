import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from config.config_loader import load_config
from registration.register import generate_structures
from dose.dose import compute_and_save_dose
from evaluation.metrics import compute_image_metrics, compute_dose_metrics


def run_dose_pipeline(cfg_path="config/config.yaml"):
    cfg        = load_config(cfg_path)
    output_dir = cfg["paths"]["output_dir"]

    patient_dirs = sorted([
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ])

    all_results = {}

    for pid in tqdm(patient_dirs, desc="Dose pipeline"):
        pdir     = os.path.join(output_dir, pid)
        sct_path = os.path.join(pdir, "sct.nii.gz")
        ct_path  = os.path.join(pdir, "ct.nii.gz")

        if not os.path.exists(sct_path) or not os.path.exists(ct_path):
            print(f"  Skipping {pid}: missing ct or sct.")
            continue

        print(f"\nProcessing {pid}")

        struct_voxels = generate_structures(pdir)
        print(f"  Structures: {struct_voxels}")

        result = {"structures": struct_voxels}

        mask_path   = os.path.join(pdir, "brain_mask.nii.gz")
        img_metrics = compute_image_metrics(ct_path, sct_path, mask_path)
        result["image_metrics"] = img_metrics
        print(f"  MAE={img_metrics['mae']:.2f} SSIM={img_metrics['ssim']:.4f} "
              f"PSNR={img_metrics['psnr']:.2f}")

        for modality in ["photon", "proton"]:
            compute_and_save_dose(pdir, cfg, modality)
            dose_metrics = compute_dose_metrics(pdir, modality, cfg)
            result[f"dose_{modality}"] = {
                k: v for k, v in dose_metrics.items() if k != "dvh"
            }
            print(f"  {modality}: gamma={dose_metrics['gamma_pass_rate']:.2f}% "
                  f"mean_diff={dose_metrics['global_mean_diff_pct']:.3f}%")

        all_results[pid] = result

    results_path = os.path.join(output_dir, "dose_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x))
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run_dose_pipeline(args.config)
