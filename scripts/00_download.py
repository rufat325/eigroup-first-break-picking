"""Download and decompress an asset's HDF5 file.

Usage:
    python scripts/00_download.py --asset lalor
    python scripts/00_download.py --asset halfmile
"""
import argparse
import lzma
import shutil
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ASSETS, DATA_DIR


def download(url, dest):
    if dest.exists():
        print(f"{dest.name} already exists, skipping download.")
        return
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))


def decompress(xz_path, out_path):
    if out_path.exists():
        print(f"{out_path.name} already exists, skipping decompression.")
        return
    print(f"Decompressing {xz_path.name}...")
    with lzma.open(xz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1 << 24)
    print(f"  -> {out_path.name} ({out_path.stat().st_size/1e9:.2f} GB)")


def main(asset):
    cfg = ASSETS[asset]
    xz  = DATA_DIR / Path(cfg["url"]).name
    download(cfg["url"], xz)
    decompress(xz, cfg["hdf5"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, choices=list(ASSETS.keys()))
    args = parser.parse_args()
    main(args.asset)
