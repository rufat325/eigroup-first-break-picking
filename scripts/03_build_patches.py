"""Extract fixed-size training patches from all shots of an asset.

Usage:
    python scripts/03_build_patches.py --asset lalor
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ASSETS
from src.shot_gathers import load_shot_index
from src.patches import (
    shot_split, extract_patches_from_shot, save_patch, save_manifest,
)


def main(asset):
    cfg = ASSETS[asset]
    if not cfg["index_pkl"].exists():
        print(f"ERROR: run scripts/02_build_gathers.py --asset {asset} first.")
        sys.exit(1)

    cache = load_shot_index(cfg["index_pkl"])
    shot_index = cache["shot_index"]
    samp_rate_ms = cache["samp_rate_ms"]

    train_ids, val_ids, test_ids = shot_split(shot_index.keys())
    print(f"Split — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)} shots")

    def split_of(sid):
        if sid in train_ids: return "train"
        if sid in val_ids:   return "val"
        return "test"

    manifest = []
    t0 = time.time()
    for sid, info in tqdm(shot_index.items(), desc=f"{asset} patches"):
        split = split_of(sid)
        for pidx, img, mask, fb, valid in extract_patches_from_shot(
            cfg["hdf5"], info, samp_rate_ms,
        ):
            entry = save_patch(cfg["patch_dir"], sid, pidx, img, mask, fb, valid, split)
            manifest.append(entry)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min; {len(manifest)} patches total")
    for s, n in Counter(e["split"] for e in manifest).items():
        print(f"  {s:5s}: {n}")

    save_manifest(manifest, cfg["manifest_pkl"])
    print(f"Manifest: {cfg['manifest_pkl']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, choices=list(ASSETS.keys()))
    args = parser.parse_args()
    main(args.asset)
