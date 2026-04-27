import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
import ants
import antspynet
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import load_config
from dose.dose import compute_and_save_dose
from evaluation.metrics import compute_image_metrics, compute_dose_metrics


MNI_STRUCTURES = {
    "PTV_thalamus"    : {"centers": [(-12,-16,4),(12,-16,4)],   "radius_mm": 12},
    "OAR_brainstem"   : {"centers": [(0,-32,-34)],              "radius_mm": 14},
    "OAR_cerebellum"  : {"centers": [(-24,-54,-30),(24,-54,-30)],"radius_mm": 18},
    "OAR_hippocampus" : {"centers": [(-26,-18,-16),(26,-18,-16)],"radius_mm": 8},
}


def ants_to_affine(ants_img):
    sp  = np.array(ants_img.spacing)
    ori = np.array(ants_img.origin)
    d   = np.array(ants_img.direction)
    affine         = np.eye(4)
    affine[:3, :3] = d * sp[np.newaxis, :]
    affine[:3,  3] = ori
    return affine


def make_sphere_mask(shape, affine, centers_mni, radius_mm):
    mask = np.zeros(shape, dtype=np.uint8)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    coords_vox  = np.stack([xx.ravel(), yy.ravel(), zz.ravel(),
                            np.ones(xx.size)], axis=0)
    coords_mni  = (affine @ coords_vox)[:3].T
    for center in centers_mni:
        c    = np.array(center)
        dist = np.sqrt(((coords_mni - c)**2).sum(axis=1))
        mask.ravel()[dist <= radius_mm] = 1
    return mask


def generate_structures(patient_out_dir):
    mr_path   = os.path.join(patient_out_dir, "mr.nii.gz")
    mask_path = os.path.join(patient_out_dir, "mask.nii.gz")
    struct_dir = os.path.join(patient_out_dir, "structures")
    os.makedirs(struct_dir, exist_ok=True)

    mr_ants = ants.image_read(mr_path)

    mr_n4 = ants.n4_bias_field_correction(mr_ants, verbose=False)
    prob  = antspynet.brain_extraction(mr_n4, modality="t1", verbose=False)
    tight_mask_ants = ants.threshold_image(prob, 0.5, 1.0)
    mr_brain = mr_n4 * tight_mask_ants

    tight_mask_nib = nib.Nifti1Image(
        tight_mask_ants.numpy().astype(np.uint8),
        nib.load(mr_path).affine
    )
    nib.save(tight_mask_nib, os.path.join(patient_out_dir, "brain_mask.nii.gz"))

    mni_template = ants.image_read(ants.get_ants_data("mni"))
    reg = ants.registration(
        fixed=mni_template,
        moving=mr_brain,
        type_of_transform="SyN",
        verbose=False
    )

    mr_raw_ants  = ants.image_read(mr_path)
    tight_mask_np = tight_mask_ants.numpy().astype(np.uint8)
    mni_affine   = ants_to_affine(mni_template)
    mr_nib       = nib.load(mr_path)

    for name, cfg_s in MNI_STRUCTURES.items():
        mni_mask = make_sphere_mask(
            tuple(mni_template.shape), mni_affine,
            cfg_s["centers"], cfg_s["radius_mm"]
        )
        mni_mask_ants = ants.from_numpy(
            mni_mask.astype(np.float32),
            origin=mni_template.origin,
            spacing=mni_template.spacing,
            direction=mni_template.direction
        )
        mask_in_mr = ants.apply_transforms(
            fixed=mr_raw_ants,
            moving=mni_mask_ants,
            transformlist=reg["invtransforms"],
            interpolator="nearestNeighbor"
        )
        mask_data = (mask_in_mr.numpy() > 0.5).astype(np.uint8) * tight_mask_np
        nib.save(
            nib.Nifti1Image(mask_data, mr_nib.affine, mr_nib.header),
            os.path.join(struct_dir, f"{name}.nii.gz")
        )

    return {n: int(nib.load(os.path.join(struct_dir, f"{n}.nii.gz"))
                   .get_fdata().sum()) for n in MNI_STRUCTURES}


def run_dose_pipeline(cfg_path="config/config.yaml"):
    cfg        = load_config(cfg_path)
    output_dir = cfg["paths"]["output_dir"]

    patient_dirs = sorted([
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ])

    all_results = {}

    for pid in tqdm(patient_dirs, desc="Dose pipeline"):
        pdir = os.path.join(output_dir, pid)
        sct_path = os.path.join(pdir, "sct.nii.gz")
        ct_path  = os.path.join(pdir, "ct.nii.gz")

        if not os.path.exists(sct_path) or not os.path.exists(ct_path):
            print(f"  Skipping {pid}: missing ct or sct.")
            continue

        print(f"\nProcessing {pid}")

        struct_voxels = generate_structures(pdir)
        print(f"  Structures: {struct_voxels}")

        result = {"structures": struct_voxels}

        mask_path = os.path.join(pdir, "brain_mask.nii.gz")
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