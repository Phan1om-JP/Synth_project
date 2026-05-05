import os
import subprocess
import sys

STORAGE_DIR = "/storage/student4/synth"

SYNTHRAD_FILES = [
    {
        "name": "Task1.zip",
        "url": "https://zenodo.org/records/7260705/files/Task1.zip?download=1",
    },
    {
        "name": "Task2.zip",
        "url": "https://zenodo.org/records/7260705/files/Task2.zip?download=1",
    },
]

VAL_FILES = [
    {
        "name": "Task1_val.zip",
        "url": "https://zenodo.org/records/7868169/files/Task1_val.zip?download=1",
    },
    {
        "name": "Task2_val.zip",
        "url": "https://zenodo.org/records/7868169/files/Task2_val.zip?download=1",
    },
]


def download_and_extract(file_info, save_dir):
    import zipfile

    name = file_info["name"]
    url = file_info["url"]
    zip_path = os.path.join(save_dir, name)
    extract_dir = os.path.join(save_dir, name.replace(".zip", "").replace(".rar", ""))

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1024:
        print(f"Downloading {name} ...")
        subprocess.run(
            ["curl", "-L", "-C", "-", "--progress-bar", "-o", zip_path, url], check=True
        )
        print(f"Downloaded: {name}")
    else:
        print(f"Already exists: {name} ({os.path.getsize(zip_path)/1e9:.2f} GB)")

    ext = os.path.splitext(name)[1].lower()
    print(f"Extracting {name} -> {extract_dir} ...")

    if ext == ".zip":
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            for i, member in enumerate(members):
                zf.extract(member, extract_dir)
                if i % 50 == 0:
                    print(f"  {i}/{len(members)} files extracted...", end="\r")
        print(f"\nExtracted: {name}")
    else:
        print(f"Unknown extension {ext}, skipping extraction.")
        return


def check_disk_space(path, required_gb=200):
    stat = os.statvfs(path)
    avail = stat.f_bavail * stat.f_frsize / 1e9
    print(f"Available space at {path}: {avail:.1f} GB")
    if avail < required_gb:
        print(f"WARNING: less than {required_gb}GB free. Proceed carefully.")
    return avail


if __name__ == "__main__":
    print("=== SynthRAD2023 Dataset Download ===")
    check_disk_space("/storage", required_gb=200)

    all_files = SYNTHRAD_FILES + VAL_FILES
    for file_info in all_files:
        try:
            download_and_extract(file_info, STORAGE_DIR)
        except subprocess.CalledProcessError as e:
            print(f"ERROR on {file_info['name']}: {e}")
            sys.exit(1)

    print("=== All downloads and extractions complete ===")
    print(f"Data location: {STORAGE_DIR}")
