# SynthRAD2023 — MRI to sCT with Dosimetric Evaluation

## Structure
```
synthrad/
├── config/          config.yaml + loader
├── data/            dataset classes (2D, 2.5D, 3D)
├── models/          UNet2D, UNet3D, PatchGAN, losses
├── preprocessing/   stats, normalization, slice caching
├── registration/    ANTs MNI registration, structure masks
├── dose/            Python photon/proton engines + matRad bridge
├── evaluation/      MAE/SSIM/PSNR, gamma index, DVH
├── scripts/         run_training.py, run_inference.py, run_dose.py
├── tests/           test_pipeline.py
└── outputs/
    └── {patient_id}/
        ├── mr.nii.gz
        ├── ct.nii.gz
        ├── sct.nii.gz
        ├── brain_mask.nii.gz
        ├── structures/
        ├── dose_ct_photon.nii.gz
        ├── dose_sct_photon.nii.gz
        ├── dose_ct_proton.nii.gz
        └── dose_sct_proton.nii.gz
```

## Setup
```bash
conda create -n synthrad python=3.9
conda activate synthrad
pip install -r requirements.txt
```

## Edit config before running
All paths and hyperparameters are in `config/config.yaml`.
Change `paths.task1_train`, `paths.output_dir`, etc. to match your server.

## Run tests first
```bash
cd synthrad
python tests/test_pipeline.py
```

## Training
```bash
python scripts/run_training.py --config config/config.yaml
# or with tmux:
tmux new -s train
python scripts/run_training.py
# Ctrl+B then D to detach
```

## Inference (generates sct.nii.gz per patient)
```bash
python scripts/run_inference.py --config config/config.yaml --split train
```

## Dose pipeline (structures + photon + proton dose per patient)
```bash
python scripts/run_dose.py --config config/config.yaml
```

## Switching experiments via config only
- 2D vs 2.5D vs 3D: change `data.spatial_mode`
- n_adjacent for 2.5D: change `data.n_adjacent` (must be odd)
- patch size for 3D: change `data.patch_size`
- L1 vs GAN: change `training.loss_type` to `l1` or `l1_gan`
- matRad vs Python dose: change `dose.use_matrad` to true/false

## Multi-GPU
DataParallel is enabled automatically when `training.multi_gpu: true`
and more than one GPU is visible. No code changes needed.
To restrict GPUs: `CUDA_VISIBLE_DEVICES=0,1 python scripts/run_training.py`