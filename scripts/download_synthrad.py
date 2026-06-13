"""
Download SynthRAD2023 dataset to /content/storage/.

Run in Colab:
    !python scripts/download_synthrad.py           # Task 1 only (default)
    !python scripts/download_synthrad.py --task 2  # Task 2 only
    !python scripts/download_synthrad.py --task 1  # Task 1 only

After download, data will be at:
    Task 1: /content/storage/Task1/brain/  and  /content/storage/Task1_val/brain/
    Task 2: /content/storage/Task2/brain/  and  /content/storage/Task2_val/brain/

Zips are removed after extraction to save disk space.
"""

import argparse
import os
import subprocess

DEST = "/content/storage"

TASK_DOWNLOADS = {
    "1": [
        ("Task1.zip",     "https://zenodo.org/records/7260705/files/Task1.zip?download=1",     "Task1"),
        ("Task1_val.zip", "https://zenodo.org/records/7868169/files/Task1_val.zip?download=1", "Task1_val"),
    ],
    "2": [
        ("Task2.zip",     "https://zenodo.org/records/7260705/files/Task2.zip?download=1",     "Task2"),
        ("Task2_val.zip", "https://zenodo.org/records/7868169/files/Task2_val.zip?download=1", "Task2_val"),
    ],
}


def download(url: str, dest_path: str):
    subprocess.run(
        ["wget", "-c", "--progress=bar:force:noscroll", "-O", dest_path, url],
        check=True,
    )


def extract(zip_path: str, dest_dir: str):
    subprocess.run(["unzip", "-q", "-o", zip_path, "-d", dest_dir], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="1", choices=["1", "2"],
                        help="Which task to download (default: 1)")
    args = parser.parse_args()

    os.makedirs(DEST, exist_ok=True)
    downloads = TASK_DOWNLOADS[args.task]

    for fname, url, folder in downloads:
        zip_path = os.path.join(DEST, fname)
        out_dir  = os.path.join(DEST, folder)

        if os.path.isdir(out_dir):
            print(f"[skip] {folder}/ already present")
        else:
            print(f"\nDownloading {fname}  (may take 10-20 min) ...")
            download(url, zip_path)

            print(f"Extracting {fname} ...")
            extract(zip_path, DEST)

            print(f"Removing zip to free space ...")
            os.remove(zip_path)

        brain_dir = os.path.join(out_dir, "brain")
        if os.path.isdir(brain_dir):
            n = len(os.listdir(brain_dir))
            print(f"  OK  {brain_dir}  ({n} patients)")
        else:
            print(f"  [WARN] brain dir not found: {brain_dir}")
            print(f"         Actual contents: {os.listdir(out_dir)}")

    task = args.task
    print(f"\nDone. Paths to use in config:")
    print(f"  task{task}_train : {DEST}/Task{task}/brain")
    print(f"  task{task}_val   : {DEST}/Task{task}_val/brain")


if __name__ == "__main__":
    main()
