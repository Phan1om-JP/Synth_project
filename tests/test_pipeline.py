import os
import sys
import json
import tempfile
import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_fake_volume(shape=(64, 64, 64), hu_range=(-1000, 1000)):
    data = np.random.uniform(hu_range[0], hu_range[1], shape).astype(np.float32)
    affine = np.eye(4)
    return nib.Nifti1Image(data, affine)


def make_fake_stats():
    return {
        "ct_clip_min"    : -1000.0,
        "ct_clip_max"    : 1000.0,
        "ct_global_mean" : -626.0,
        "ct_global_std"  : 551.0,
        "mr_p01_mean"    : 0.0,
        "mr_p99_mean"    : 1.0,
        "mr_p01_std"     : 0.0,
        "mr_p99_std"     : 0.0,
        "n_patients"     : 10,
    }


def test_preprocess_mr():
    from preprocessing.preprocess import preprocess_mr
    vol  = np.random.rand(50, 50, 50).astype(np.float32) * 1000
    mask = np.ones((50, 50, 50), dtype=np.float32)
    out  = preprocess_mr(vol, mask)
    assert out.min() >= 0.0 - 1e-5
    assert out.max() <= 1.0 + 1e-5
    assert out.dtype == np.float32
    print("PASS test_preprocess_mr")


def test_preprocess_ct():
    from preprocessing.preprocess import preprocess_ct, postprocess_ct
    stats = make_fake_stats()
    vol   = np.random.uniform(-1000, 1000, (50, 50, 50)).astype(np.float32)
    proc  = preprocess_ct(vol, stats)
    back  = postprocess_ct(proc, stats)
    assert back.min() >= stats["ct_clip_min"] - 1e-3
    assert back.max() <= stats["ct_clip_max"] + 1e-3
    print("PASS test_preprocess_ct")


def test_pad_to_target():
    from preprocessing.preprocess import pad_to_target
    arr = np.ones((200, 180), dtype=np.float32)
    out = pad_to_target(arr, 320, 320)
    assert out.shape == (320, 320)
    print("PASS test_pad_to_target")


def test_unet2d_forward():
    from models.unet import UNet2D
    model = UNet2D(in_ch=1, out_ch=1, base_ch=16)
    x     = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 256, 256), f"Expected (2,1,256,256) got {y.shape}"
    print("PASS test_unet2d_forward")


def test_unet2d_25d_forward():
    from models.unet import UNet2D
    model = UNet2D(in_ch=5, out_ch=1, base_ch=16)
    x     = torch.randn(2, 5, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 256, 256)
    print("PASS test_unet2d_25d_forward (n_adjacent=5)")


def test_unet3d_forward():
    from models.unet import UNet3D
    model = UNet3D(in_ch=1, out_ch=1, base_ch=4)
    x     = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 64, 64, 64), f"Expected (1,1,64,64,64) got {y.shape}"
    print("PASS test_unet3d_forward")


def test_losses():
    from models.losses import masked_l1, gan_generator_loss, gan_discriminator_loss
    pred   = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    mask   = torch.ones(2, 1, 64, 64)
    loss   = masked_l1(pred, target, mask)
    assert loss.item() >= 0
    disc_fake = torch.randn(2, 1, 30, 30)
    disc_real = torch.randn(2, 1, 30, 30)
    g_loss = gan_generator_loss(disc_fake)
    d_loss = gan_discriminator_loss(disc_real, disc_fake)
    assert g_loss.item() >= 0
    assert d_loss.item() >= 0
    print("PASS test_losses")


def test_photon_dose():
    from dose.dose import compute_photon_dose
    ct_arr   = np.random.uniform(-1000, 1000, (60, 60, 60)).astype(np.float32)
    ptv_mask = np.zeros((60, 60, 60), dtype=bool)
    ptv_mask[25:35, 25:35, 25:35] = True
    spacing  = np.array([1.0, 1.0, 1.0])
    dose     = compute_photon_dose(ct_arr, ptv_mask, spacing, [0, 90])
    assert dose.shape == ct_arr.shape
    assert dose[ptv_mask].max() <= 1.0 + 1e-5
    print("PASS test_photon_dose")


def test_proton_dose():
    from dose.dose import compute_proton_dose
    ct_arr   = np.random.uniform(-100, 100, (60, 60, 60)).astype(np.float32)
    ptv_mask = np.zeros((60, 60, 60), dtype=bool)
    ptv_mask[25:35, 25:35, 25:35] = True
    spacing  = np.array([1.0, 1.0, 1.0])
    dose     = compute_proton_dose(ct_arr, ptv_mask, spacing, [0])
    assert dose.shape == ct_arr.shape
    print("PASS test_proton_dose")


def test_image_metrics():
    from evaluation.metrics import compute_image_metrics
    with tempfile.TemporaryDirectory() as tmp:
        ct_vol  = np.random.uniform(-500, 500, (50, 50, 50)).astype(np.float32)
        sct_vol = ct_vol + np.random.normal(0, 50, ct_vol.shape).astype(np.float32)
        mask    = np.ones((50, 50, 50), dtype=np.float32)
        affine  = np.eye(4)
        nib.save(nib.Nifti1Image(ct_vol,  affine), os.path.join(tmp, "ct.nii.gz"))
        nib.save(nib.Nifti1Image(sct_vol, affine), os.path.join(tmp, "sct.nii.gz"))
        nib.save(nib.Nifti1Image(mask,    affine), os.path.join(tmp, "mask.nii.gz"))
        metrics = compute_image_metrics(
            os.path.join(tmp, "ct.nii.gz"),
            os.path.join(tmp, "sct.nii.gz"),
            os.path.join(tmp, "mask.nii.gz"),
        )
        assert "mae" in metrics and metrics["mae"] >= 0
        assert "ssim" in metrics
        assert "psnr" in metrics
    print("PASS test_image_metrics")


def test_gamma_index():
    from evaluation.metrics import gamma_index_3d
    ref  = np.zeros((30, 30, 30), dtype=np.float32)
    ref[10:20, 10:20, 10:20] = 1.0
    evl  = ref.copy()
    _, pass_rate = gamma_index_3d(ref, evl, np.array([1.0, 1.0, 1.0]))
    assert abs(pass_rate - 100.0) < 1e-3, f"Identical arrays should give 100% pass rate, got {pass_rate}"
    print("PASS test_gamma_index")


def test_config_loader():
    from config.config_loader import load_config, get_n_input_channels
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("""
project:
  name: test
  seed: 42
  output_dir: outputs
paths:
  data_root: /tmp
  task1_train: /tmp
  task1_val: /tmp
  stats_path: /tmp/stats.json
  cache_dir: /tmp/cache
  checkpoint_dir: /tmp/ckpt
  output_dir: /tmp/out
preprocessing:
  ct_clip_min: -1000.0
  ct_clip_max: 1000.0
  min_mask_coverage: 0.1
  n4_bias_correction: false
  brain_extraction: false
  target_size: 320
data:
  plane: axial
  spatial_mode: 2D
  n_adjacent: 3
  patch_size: 96
  train_val_split: 0.9
  batch_size: 4
  num_workers: 0
  pin_memory: false
model:
  type: unet
  in_ch: 1
  out_ch: 1
  base_ch: 16
training:
  loss_type: l1
  lr: 0.0002
  epochs: 10
  val_every: 5
  save_every: 5
  early_stopping_patience: 5
  gan_lambda: 0.1
  multi_gpu: false
registration:
  transform_type: SyN
  mni_radius:
    PTV_thalamus: 12
  mni_centers:
    PTV_thalamus: [[-12, -16, 4], [12, -16, 4]]
dose:
  photon_beam_angles: [0, 90]
  proton_beam_angles: [0]
  gamma_dd: 0.02
  gamma_dta_mm: 2.0
  gamma_threshold: 0.1
  use_matrad: false
  matrad_path: /tmp/matrad
""")
        tmp_cfg = f.name

    cfg = load_config(tmp_cfg)
    assert cfg["project"]["seed"] == 42
    assert "device" in cfg
    n = get_n_input_channels(cfg)
    assert n == 1
    os.unlink(tmp_cfg)
    print("PASS test_config_loader")


TESTS = [
    test_config_loader,
    test_preprocess_mr,
    test_preprocess_ct,
    test_pad_to_target,
    test_unet2d_forward,
    test_unet2d_25d_forward,
    test_unet3d_forward,
    test_losses,
    test_photon_dose,
    test_proton_dose,
    test_image_metrics,
    test_gamma_index,
]

if __name__ == "__main__":
    passed = 0
    failed = 0
    for test_fn in TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed.")
    sys.exit(0 if failed == 0 else 1)