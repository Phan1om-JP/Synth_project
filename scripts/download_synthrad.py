"""
Download SynthRAD2023 Task 1 dataset to /content/storage/.

Run in Colab:
    !python scripts/download_synthrad.py

After download, data will be at:
    /content/storage/Task1/brain/
    /content/storage/Task1_val/brain/

Both zips are removed after extraction to save disk space.
Total extracted size: ~6-10 GB.
"""

import os
import subprocess
import sys

DEST = "/content/storage"

DOWNLOADS = [
    (
        "Task1.zip",
        "https://zenodo.org/records/7260705/files/Task1.zip?download=1",
        "Task1",
    ),
    (
        "Task1_val.zip",
        "https://zenodo.org/records/7868169/files/Task1_val.zip?download=1",
        "Task1_val",
    ),
]


def download(url: str, dest_path: str):
    """wget with resume support (-c) and visible progress bar."""
    subprocess.run(
        [
            "wget", "-c",
            "--progress=bar:force:noscroll",
            "-O", dest_path,
            url,
        ],
        check=True,
    )


def extract(zip_path: str, dest_dir: str):
    subprocess.run(["unzip", "-q", "-o", zip_path, "-d", dest_dir], check=True)


def main():
    os.makedirs(DEST, exist_ok=True)

    for fname, url, folder in DOWNLOADS:
        zip_path = os.path.join(DEST, fname)
        out_dir  = os.path.join(DEST, folder)

        if os.path.isdir(out_dir):
            print(f"[skip] {folder}/ already present")
        else:
            print(f"\nDownloading {fname}  (this may take 10-20 min on Colab) ...")
            download(url, zip_path)

            print(f"Extracting {fname} ...")
            extract(zip_path, DEST)

            print(f"Removing zip to free space ...")
            os.remove(zip_path)

        # Sanity check — zip extracts flat: Task1/brain/ (not Task1/Task1/brain/)
        brain_dir = os.path.join(out_dir, "brain")
        if os.path.isdir(brain_dir):
            n = len(os.listdir(brain_dir))
            print(f"  OK  {brain_dir}  ({n} patients)")
        else:
            print(f"  [WARN] Expected brain dir not found: {brain_dir}")
            print(f"         Actual contents: {os.listdir(out_dir)}")

    print("\nAll done. Paths to use in config:")
    print(f"  task1_train : {DEST}/Task1/brain")
    print(f"  task1_val   : {DEST}/Task1_val/brain")


if __name__ == "__main__":
    main()
