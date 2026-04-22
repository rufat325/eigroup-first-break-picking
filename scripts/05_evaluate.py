"""Evaluate a method (STA/LTA or a trained U-Net) on an asset's test split.

Usage:
    # STA/LTA baseline on Lalor:
    python scripts/05_evaluate.py --method stalta --asset lalor

    # U-Net inference:
    python scripts/05_evaluate.py --method unet --asset lalor --model cache/unet_lalor.keras
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ASSETS, CACHE_DIR, PATCH_HEIGHT
from src.shot_gathers import load_shot_index, load_shot_gather, fb_samples_from_shot_info
from src.patches import load_manifest
from src.model import CUSTOM_OBJECTS
from src.inference import predict_shot
from src.baseline_stalta import sta_lta, pick_first_break
from src.metrics import summarize, print_summary


def _test_shot_ids(manifest_pkl):
    manifest = load_manifest(manifest_pkl)
    return sorted(set(e["shot_id"] for e in manifest if e["split"] == "test"))


def _evaluate_stalta(asset):
    cfg = ASSETS[asset]
    cache = load_shot_index(cfg["index_pkl"])
    shot_index = cache["shot_index"]
    samp_rate_ms = cache["samp_rate_ms"]

    test_ids = _test_shot_ids(cfg["manifest_pkl"])
    print(f"Evaluating STA/LTA on {len(test_ids)} {asset} test shots")

    all_errors_ms = []
    for sid in tqdm(test_ids):
        info  = shot_index[sid]
        g     = load_shot_gather(cfg["hdf5"], info)
        fb_t  = fb_samples_from_shot_info(info, samp_rate_ms)
        ratio = sta_lta(g, sta_len=20, lta_len=200)
        fb_p  = pick_first_break(ratio, threshold=3.0)
        m     = ~np.isnan(fb_t) & ~np.isnan(fb_p)
        all_errors_ms.append((fb_p[m] - fb_t[m]) * samp_rate_ms)

    return np.concatenate(all_errors_ms)


def _evaluate_unet(asset, model_path):
    cfg = ASSETS[asset]
    cache = load_shot_index(cfg["index_pkl"])
    shot_index = cache["shot_index"]
    samp_rate_ms = cache["samp_rate_ms"]

    model = tf.keras.models.load_model(
        str(model_path), custom_objects=CUSTOM_OBJECTS, compile=False,
    )
    print(f"Loaded model: {model_path}")

    test_ids = _test_shot_ids(cfg["manifest_pkl"])
    print(f"Evaluating U-Net on {len(test_ids)} {asset} test shots")

    all_errors_ms = []
    for sid in tqdm(test_ids):
        info = shot_index[sid]
        # Crop only Lalor (1 ms) gathers; other assets get upsampled inside predict_shot.
        time_crop = PATCH_HEIGHT if samp_rate_ms == 1.0 else None
        g    = load_shot_gather(cfg["hdf5"], info, time_crop=time_crop)
        fb_t = fb_samples_from_shot_info(info, samp_rate_ms)

        fb_pred_1ms, _ = predict_shot(g, model, samp_rate_ms)

        # Bring predictions into native-sample coordinates, then ms.
        fb_pred_native = fb_pred_1ms * (1.0 / samp_rate_ms)
        m = ~np.isnan(fb_t) & ~np.isnan(fb_pred_native)
        all_errors_ms.append((fb_pred_native[m] - fb_t[m]) * samp_rate_ms)

    return np.concatenate(all_errors_ms)


def main(args):
    if args.method == "stalta":
        errs = _evaluate_stalta(args.asset)
        label = f"STA/LTA on {args.asset}"
    else:
        if args.model is None:
            print("ERROR: --model is required for method=unet")
            sys.exit(1)
        errs = _evaluate_unet(args.asset, args.model)
        label = f"{Path(args.model).stem} on {args.asset}"

    summary = summarize(errs, label)
    print()
    print_summary(summary)

    if args.save:
        out = CACHE_DIR / f"eval_{args.method}_{args.asset}.pkl"
        with open(out, "wb") as f:
            pickle.dump({"summary": summary, "errors_ms": errs}, f)
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["stalta", "unet"])
    parser.add_argument("--asset",  required=True, choices=list(ASSETS.keys()))
    parser.add_argument("--model",  type=Path, default=None,
                        help="Path to .keras model (required for method=unet)")
    parser.add_argument("--save",   action="store_true",
                        help="Save raw errors and summary to a pkl file.")
    args = parser.parse_args()
    main(args)
