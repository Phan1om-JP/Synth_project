import yaml
import torch
from pathlib import Path


def load_config(path="config/config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def get_n_input_channels(cfg):
    mode = cfg["data"]["spatial_mode"]
    if mode == "2D":
        return cfg["model"]["in_ch"]
    if mode == "2.5D":
        return cfg["data"]["n_adjacent"]
    if mode == "3D":
        return cfg["model"]["in_ch"]
    raise ValueError(f"Unknown spatial_mode: {mode}")