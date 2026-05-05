import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import load_config
from preprocessing.preprocess import compute_and_save_stats, build_cache


def run_preprocessing(cfg_path="config/config.yaml", force_stats=False):
    cfg = load_config(cfg_path)

    os.makedirs(cfg["paths"]["cache_dir"], exist_ok=True)

    stats_path = cfg["paths"]["stats_path"]
    if force_stats and os.path.exists(stats_path):
        os.remove(stats_path)
        print(f"Removed existing stats: {stats_path}")

    if os.path.exists(stats_path):
        import json
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"Stats loaded from: {stats_path}")
    else:
        stats = compute_and_save_stats(
            cfg["paths"]["task1_train"],
            stats_path,
            cfg["preprocessing"]["ct_clip_min"],
            cfg["preprocessing"]["ct_clip_max"],
        )

    build_cache(cfg, stats)
    print("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--force-stats", action="store_true",
                        help="Recompute stats even if JSON exists")
    args = parser.parse_args()
    run_preprocessing(args.config, args.force_stats)
