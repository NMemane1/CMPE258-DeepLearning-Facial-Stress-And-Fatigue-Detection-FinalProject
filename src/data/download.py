"""Download and prepare datasets.

Usage (from repo root):
    python -m src.data.download --output-root data/processed

Requires the Kaggle API token in ~/.kaggle/kaggle.json. On Kaggle notebooks
the datasets are already mounted at /kaggle/input — this script no-ops there.
"""
from __future__ import annotations

import os
import sys
import argparse
import zipfile
from pathlib import Path
import subprocess
import shutil


KAGGLE_DROWSINESS = "dheerajperumandla/drowsiness-dataset"
KAGGLE_FER2013    = "msambare/fer2013"


def _on_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download_drowsiness(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _on_kaggle():
        src = Path("/kaggle/input/drowsiness-dataset")
        if src.exists():
            for child in src.iterdir():
                tgt = out_dir / child.name
                if not tgt.exists():
                    shutil.copytree(child, tgt) if child.is_dir() else shutil.copy(child, tgt)
            return
    _run(["kaggle", "datasets", "download", "-d", KAGGLE_DROWSINESS, "-p", str(out_dir), "--unzip"])


def download_fer2013(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _on_kaggle():
        src = Path("/kaggle/input/fer2013")
        if src.exists():
            for child in src.iterdir():
                tgt = out_dir / child.name
                if not tgt.exists():
                    shutil.copytree(child, tgt) if child.is_dir() else shutil.copy(child, tgt)
            return
    _run(["kaggle", "datasets", "download", "-d", KAGGLE_FER2013, "-p", str(out_dir), "--unzip"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--skip-drowsiness", action="store_true")
    parser.add_argument("--skip-fer", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    if not args.skip_drowsiness:
        download_drowsiness(root / "drowsiness")
    if not args.skip_fer:
        download_fer2013(root / "fer2013")

    print("Done.")


if __name__ == "__main__":
    main()
