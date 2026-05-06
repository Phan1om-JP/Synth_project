import os
import numpy as np
import nibabel as nib
import ants
import antspynet


MNI_STRUCTURES = {
    "PTV_thalamus"    : {"centers": [(-12,-16,4),(12,-16,4)],    "radius_mm": 12},
    "OAR_brainstem"   : {"centers": [(0,-32,-34)],               "radius_mm": 14},
    "OAR_cerebellum"  : {"centers": [(-24,-54,-30),(24,-54,-30)], "radius_mm": 18},
    "OAR_hippocampus" : {"centers": [(-26,-18,-16),(26,-18,-16)], "radius_mm": 8},
}


def _ants_to_affine(ants_img):
    sp  = np.array(ants_img.spacing)
    ori = np.array(ants_img.origin)
    d   = np.array(ants_img.direction)
    affine         = np.eye(4)
    affine[:3, :3] = d * sp[np.newaxis, :]
    affine[:3,  3] = ori
    return affine


def _make_sphere_mask(shape, affine, centers_mni, radius_mm):
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
    mr_path    = os.path.join(patient_out_dir, "mr.nii.gz")
    struct_dir = os.path.join(patient_out_dir, "structures")
    os.makedirs(struct_dir, exist_ok=True)

    mr_ants = ants.image_read(mr_path)
    mr_n4   = ants.n4_bias_field_correction(mr_ants, verbose=False)
    prob    = antspynet.brain_extraction(mr_n4, modality="t1", verbose=False)
    tight_mask_ants = ants.threshold_image(prob, 0.5, 1.0)
    mr_brain = mr_n4 * tight_mask_ants

    nib.save(
        nib.Nifti1Image(tight_mask_ants.numpy().astype(np.uint8),
                        nib.load(mr_path).affine),
        os.path.join(patient_out_dir, "brain_mask.nii.gz")
    )

    mni_template = ants.image_read(ants.get_ants_data("mni"))
    reg = ants.registration(
        fixed=mni_template, moving=mr_brain,
        type_of_transform="SyN", verbose=False
    )

    mr_raw_ants   = ants.image_read(mr_path)
    tight_mask_np = tight_mask_ants.numpy().astype(np.uint8)
    mni_affine    = _ants_to_affine(mni_template)
    mr_nib        = nib.load(mr_path)

    for name, cfg_s in MNI_STRUCTURES.items():
        mni_mask = _make_sphere_mask(
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
            fixed=mr_raw_ants, moving=mni_mask_ants,
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
