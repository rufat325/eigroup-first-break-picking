"""Build and cache shot indices for an asset.

Usage:
    python scripts/02_build_gathers.py --asset lalor
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ASSETS
from src.shot_gathers import build_shot_index, save_shot_index


def main(asset):
    cfg = ASSETS[asset]
    if not cfg["hdf5"].exists():
        print(f"ERROR: {cfg['hdf5']} not found.")
        sys.exit(1)

    print(f"Building shot index for {asset}...")
    cache = build_shot_index(cfg["hdf5"])
    n_shots = len(cache["shot_index"])
    print(f"  Usable shots:     {n_shots}")

    total_labeled = sum(v["n_labeled"] for v in cache["shot_index"].values())
    total_traces  = sum(v["n_traces"]  for v in cache["shot_index"].values())
    print(f"  Total traces:     {total_traces:,}")
    print(f"  Labeled traces:   {total_labeled:,}")

    save_shot_index(cache, cfg["index_pkl"])
    print(f"  Saved to {cfg['index_pkl']} ({cfg['index_pkl'].stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, choices=list(ASSETS.keys()))
    args = parser.parse_args()
    main(args.asset)
