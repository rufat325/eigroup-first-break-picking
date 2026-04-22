"""Load and report summary statistics for an asset.

Usage:
    python scripts/01_explore.py --asset lalor
    python scripts/01_explore.py --asset halfmile
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ASSETS
from src.data_loader import load_metadata, extract_constants, labeled_mask


def main(asset):
    cfg = ASSETS[asset]
    if not cfg["hdf5"].exists():
        print(f"ERROR: {cfg['hdf5']} not found. Download first.")
        print(f"       URL: {cfg['url']}")
        sys.exit(1)

    print(f"Loading {asset} metadata from {cfg['hdf5'].name}...")
    meta, data_shape = load_metadata(cfg["hdf5"])
    consts = extract_constants(meta)

    n_traces, n_samples = data_shape
    print(f"\n=== {asset.upper()} summary ===")
    print(f"  Traces:           {n_traces:,}")
    print(f"  Samples per trace:{n_samples}")
    print(f"  Sampling rate:    {consts['samp_rate_ms']} ms ({consts['samp_rate_us']} us)")
    print(f"  Trace duration:   {consts['samp_num'] * consts['samp_rate_ms']:.0f} ms")
    print(f"  COORD_SCALE:      {consts['coord_scale']}")
    print(f"  HT_SCALE:         {consts['ht_scale']}")

    lbl = labeled_mask(meta)
    fb_valid = meta["SPARE1"][lbl].astype(np.float64)
    print(f"\n  Labeled traces:   {int(lbl.sum()):,} / {n_traces:,}  ({100*lbl.mean():.1f}%)")
    print(f"  First-break ms:   min={fb_valid.min():.1f}, "
          f"median={np.median(fb_valid):.1f}, max={fb_valid.max():.1f}")

    shot_ids = meta["SHOTID"]
    uniq, counts = np.unique(shot_ids, return_counts=True)
    print(f"\n  Shots:            {len(uniq)}")
    print(f"  Traces per shot:  min={counts.min()}, median={int(np.median(counts))}, "
          f"max={counts.max()}, mean={counts.mean():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, choices=list(ASSETS.keys()))
    args = parser.parse_args()
    main(args.asset)
