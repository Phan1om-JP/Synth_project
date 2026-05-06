import os
import sys
import time
import json
import platform
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


def check_python():
    section("Python / Environment")
    v = sys.version_info
    print(f"  Python {v.major}.{v.minor}.{v.micro}  ({sys.executable})")
    if v.major == 3 and v.minor >= 8:
        print(f"  {PASS} Python version OK")
    else:
        print(f"  {WARN} Expected Python 3.8+")
    print(f"  Platform: {platform.platform()}")


def check_torch():
    section("PyTorch / CUDA")
    try:
        import torch
        print(f"  torch {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_ok}")
        if cuda_ok:
            n = torch.cuda.device_count()
            print(f"  {PASS} {n} GPU(s) found")
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                vram  = props.total_memory / 1024**3
                print(f"    GPU {i}: {props.name} | {vram:.1f} GB VRAM")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print(f"  {WARN} No CUDA — will train on CPU (very slow)")
    except ImportError:
        print(f"  {FAIL} torch not installed")


def check_imports():
    section("Required Libraries")
    libs = [
        ("nibabel",      "nibabel"),
        ("numpy",        "numpy"),
        ("scipy",        "scipy"),
        ("skimage",      "scikit-image"),
        ("tqdm",         "tqdm"),
        ("yaml",         "pyyaml"),
        ("ants",         "antspyx"),
        ("antspynet",    "antspynet"),
        ("matplotlib",   "matplotlib"),
    ]
    all_ok = True
    for mod, pkg in libs:
        try:
            __import__(mod)
            print(f"  {PASS} {pkg}")
        except ImportError:
            print(f"  {FAIL} {pkg} — run: pip install {pkg}")
            all_ok = False
    return all_ok


def check_project_imports():
    section("Project Module Imports")
    modules = [
        "config.config_loader",
        "preprocessing.preprocess",
        "data.dataset",
        "models.unet",
        "models.gan",
        "models.losses",
        "dose.photon",
        "dose.proton",
        "dose.matrad_bridge",
        "dose.dose",
        "registration.register",
        "evaluation.metrics",
    ]
    all_ok = True
    for mod in modules:
        try:
            __import__(mod)
            print(f"  {PASS} {mod}")
        except Exception as e:
            print(f"  {FAIL} {mod}: {e}")
            all_ok = False
    return all_ok


def check_disk(paths):
    section("Disk Space")
    for label, path in paths:
        if not os.path.exists(path):
            print(f"  {WARN} Path not found: {path}  ({label})")
            continue
        st   = os.statvfs(path)
        free = st.f_bavail * st.f_frsize / 1024**3
        total = st.f_blocks * st.f_frsize / 1024**3
        used  = total - free
        tag   = PASS if free > 20 else WARN
        print(f"  {tag} {label}: {free:.1f} GB free / {total:.1f} GB total  ({path})")


def check_memory():
    section("System RAM")
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                k, v = line.split(":")
                info[k.strip()] = v.strip()
        total_kb = int(info["MemTotal"].split()[0])
        free_kb  = int(info["MemAvailable"].split()[0])
        total_gb = total_kb / 1024**2
        free_gb  = free_kb  / 1024**2
        tag = PASS if free_gb > 8 else WARN
        print(f"  {tag} RAM: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    except Exception as e:
        print(f"  {WARN} Could not read /proc/meminfo: {e}")


def check_dataset(cfg):
    section("Dataset Paths")
    paths_to_check = [
        ("task1_train", cfg["paths"]["task1_train"]),
        ("task1_val",   cfg["paths"]["task1_val"]),
        ("stats_path",  cfg["paths"]["stats_path"]),
        ("cache_dir",   cfg["paths"]["cache_dir"]),
        ("checkpoint_dir", cfg["paths"]["checkpoint_dir"]),
        ("output_dir",  cfg["paths"]["output_dir"]),
    ]
    for label, path in paths_to_check:
        exists = os.path.exists(path)
        tag    = PASS if exists else WARN
        print(f"  {tag} {label}: {path}")
        if exists and os.path.isdir(path) and label == "task1_train":
            patients = [d for d in os.listdir(path)
                        if os.path.isdir(os.path.join(path, d)) and d != "overview"]
            print(f"        -> {len(patients)} patient folders found")
        if exists and label == "stats_path":
            try:
                with open(path) as f:
                    st = json.load(f)
                print(f"        -> n_patients={st.get('n_patients','?')}  "
                      f"ct_mean={st.get('ct_global_mean', 0):.2f}")
            except Exception:
                pass
        if exists and os.path.isdir(path) and label == "cache_dir":
            patient_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            total_slices = sum(
                len(os.listdir(os.path.join(path, d))) for d in patient_dirs
            )
            print(f"        -> {len(patient_dirs)} cached patients, {total_slices} slices")


def check_gpu_forward_pass():
    section("GPU Forward Pass (smoke test)")
    try:
        import torch
        from models.unet import UNet2D
        if not torch.cuda.is_available():
            print(f"  {WARN} Skipping — no CUDA")
            return
        model = UNet2D(in_ch=1, out_ch=1, base_ch=16).cuda()
        x = torch.randn(2, 1, 256, 256).cuda()
        t0 = time.time()
        with torch.no_grad():
            y = model(x)
        elapsed = (time.time() - t0) * 1000
        assert y.shape == (2, 1, 256, 256)
        print(f"  {PASS} UNet2D forward pass OK ({elapsed:.0f} ms for batch=2, 256x256)")

        from models.unet import UNet3D
        model3d = UNet3D(in_ch=1, out_ch=1, base_ch=8).cuda()
        x3 = torch.randn(1, 1, 64, 64, 64).cuda()
        t0 = time.time()
        with torch.no_grad():
            y3 = model3d(x3)
        elapsed3 = (time.time() - t0) * 1000
        assert y3.shape == (1, 1, 64, 64, 64)
        print(f"  {PASS} UNet3D forward pass OK ({elapsed3:.0f} ms for batch=1, 64³)")
    except Exception as e:
        print(f"  {FAIL} Forward pass failed: {e}")


def estimate_training_time():
    section("Training Time Estimate")
    try:
        import torch
        if not torch.cuda.is_available():
            print(f"  {WARN} Cannot estimate without CUDA")
            return

        from models.unet import UNet2D
        model = UNet2D(in_ch=1, out_ch=1, base_ch=64).cuda()
        opt   = torch.optim.Adam(model.parameters(), lr=2e-4)

        x = torch.randn(8, 1, 320, 320).cuda()
        y = torch.randn(8, 1, 320, 320).cuda()

        for _ in range(3):
            opt.zero_grad()
            torch.abs(model(x) - y).mean().backward()
            opt.step()

        times = []
        for _ in range(5):
            t0 = time.time()
            opt.zero_grad()
            torch.abs(model(x) - y).mean().backward()
            opt.step()
            torch.cuda.synchronize()
            times.append(time.time() - t0)

        sec_per_batch = sum(times) / len(times)

        slices_2d   = 180 * 150
        batches_2d  = slices_2d / 8
        sec_epoch_2d = sec_per_batch * batches_2d
        print(f"  2D  (~{slices_2d} slices, batch=8): "
              f"~{sec_epoch_2d/60:.1f} min/epoch  "
              f"| 1000 epochs ≈ {sec_epoch_2d*1000/3600:.0f} h")

        slices_25d   = 180 * 140
        batches_25d  = slices_25d / 8
        sec_epoch_25d = sec_per_batch * batches_25d * 1.15
        print(f"  2.5D (~{slices_25d} slices, batch=8): "
              f"~{sec_epoch_25d/60:.1f} min/epoch  "
              f"| 1000 epochs ≈ {sec_epoch_25d*1000/3600:.0f} h")

        patches_3d   = 180 * 8
        sec_epoch_3d = sec_per_batch * (patches_3d / 1) * 4.5
        print(f"  3D  (~{patches_3d} patches, batch=1): "
              f"~{sec_epoch_3d/60:.1f} min/epoch  "
              f"| 1000 epochs ≈ {sec_epoch_3d*1000/3600:.0f} h")

        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            print(f"  (Estimates above are for 1 GPU. {n_gpu} GPUs ~{n_gpu}x faster with DataParallel)")

    except Exception as e:
        print(f"  {FAIL} Estimation failed: {e}")


def main():
    print("\nSynthRAD2023 Server Readiness Check")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    check_python()
    check_torch()
    check_memory()

    libs_ok = check_imports()

    if libs_ok:
        proj_ok = check_project_imports()
    else:
        print(f"\n  {WARN} Skipping project imports until missing libs are installed.")
        proj_ok = False

    try:
        from config.config_loader import load_config
        cfg = load_config("config/config.yaml")
        check_dataset(cfg)
    except Exception as e:
        print(f"\n  {WARN} Could not load config: {e}")
        cfg = None

    check_disk([
        ("storage", "/storage/student4"),
        ("home",    "/home/student4"),
    ])

    if proj_ok:
        check_gpu_forward_pass()
        estimate_training_time()

    print(f"\n{'='*50}")
    print("  Check complete.")
    print('='*50)


if __name__ == "__main__":
    main()
