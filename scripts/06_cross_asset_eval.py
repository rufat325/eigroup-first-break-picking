"""Build the 2x2 cross-asset evaluation matrix and comparison plots.

For each of two models x two assets, run inference on the asset's test
split and record MAE, RMSE, and hit-rate statistics.

Usage:
    python scripts/06_cross_asset_eval.py \\
        --lalor-model cache/unet_lalor.keras \\
        --halfmile-model cache/unet_halfmile_ft.keras
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ASSETS, CACHE_DIR, FIGURES_DIR, PATCH_HEIGHT
from src.shot_gathers import load_shot_index, load_shot_gather, fb_samples_from_shot_info
from src.patches import load_manifest
from src.model import CUSTOM_OBJECTS
from src.inference import predict_shot
from src.metrics import summarize


def test_shot_ids(asset):
    return sorted(set(e["shot_id"]
                      for e in load_manifest(ASSETS[asset]["manifest_pkl"])
                      if e["split"] == "test"))


def evaluate_pair(model, asset):
    """Run `model` on `asset`'s test shots; return error array in ms."""
    cfg = ASSETS[asset]
    cache = load_shot_index(cfg["index_pkl"])
    shot_index = cache["shot_index"]
    samp_rate_ms = cache["samp_rate_ms"]

    all_errors_ms = []
    for sid in tqdm(test_shot_ids(asset), desc=asset):
        info = shot_index[sid]
        time_crop = PATCH_HEIGHT if samp_rate_ms == 1.0 else None
        g    = load_shot_gather(cfg["hdf5"], info, time_crop=time_crop)
        fb_t = fb_samples_from_shot_info(info, samp_rate_ms)

        fb_pred_1ms, _ = predict_shot(g, model, samp_rate_ms)
        fb_pred_native = fb_pred_1ms * (1.0 / samp_rate_ms)

        m = ~np.isnan(fb_t) & ~np.isnan(fb_pred_native)
        all_errors_ms.append((fb_pred_native[m] - fb_t[m]) * samp_rate_ms)

    return np.concatenate(all_errors_ms)


def main(lalor_model_path, halfmile_model_path):
    print(f"Lalor model:    {lalor_model_path}")
    print(f"Halfmile model: {halfmile_model_path}")
    lalor_model    = tf.keras.models.load_model(str(lalor_model_path),    custom_objects=CUSTOM_OBJECTS, compile=False)
    halfmile_model = tf.keras.models.load_model(str(halfmile_model_path), custom_objects=CUSTOM_OBJECTS, compile=False)

    combos = [
        ("Lalor model",    lalor_model,    "lalor"),
        ("Lalor model",    lalor_model,    "halfmile"),
        ("Halfmile-FT",    halfmile_model, "lalor"),
        ("Halfmile-FT",    halfmile_model, "halfmile"),
    ]

    cells = {}
    err_dists = {}
    for model_name, model, asset_name in combos:
        print(f"\n--- {model_name} on {asset_name} ---")
        errs = evaluate_pair(model, asset_name)
        summary = summarize(errs, f"{model_name} -> {asset_name}")
        cells[(model_name, asset_name)] = summary
        err_dists[(model_name, asset_name)] = errs
        print(f"    MAE={summary['mae_ms']:.3f} ms, "
              f"median |e|={summary['median_abs_ms']:.2f} ms, "
              f"hit<=5ms={100*summary['hit_5ms']:.1f}%")

    _print_matrix(cells)
    _plot_histograms(err_dists, cells)
    _plot_bar(cells)

    out = CACHE_DIR / "cross_asset_results.pkl"
    with open(out, "wb") as f:
        pickle.dump({"cells": cells,
                     "err_dists": {str(k): v for k, v in err_dists.items()}}, f)
    print(f"\nResults -> {out}")


def _print_matrix(cells):
    print("\n" + "=" * 70)
    print("CROSS-ASSET MAE MATRIX (ms)")
    print("=" * 70)
    print(f"{'':<20s}  {'Lalor test':>15s}  {'Halfmile test':>15s}")
    for mn in ("Lalor model", "Halfmile-FT"):
        row = f"{mn:<20s}  "
        for an in ("lalor", "halfmile"):
            row += f"{cells[(mn, an)]['mae_ms']:>13.3f} ms  "
        print(row)


def _plot_histograms(err_dists, cells):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    entries = [
        (("Lalor model", "lalor"),    axes[0, 0], "Lalor model -> Lalor (in-asset)"),
        (("Lalor model", "halfmile"), axes[0, 1], "Lalor model -> Halfmile (zero-shot)"),
        (("Halfmile-FT", "lalor"),    axes[1, 0], "Halfmile-FT -> Lalor (forgetting check)"),
        (("Halfmile-FT", "halfmile"), axes[1, 1], "Halfmile-FT -> Halfmile (fine-tuned)"),
    ]
    for key, ax, title in entries:
        errs = err_dists[key]; s = cells[key]
        ax.hist(errs, bins=150, range=(-100, 100),
                color="steelblue", alpha=0.8, edgecolor="k", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Error (ms)"); ax.set_ylabel("Count")
        ax.set_title(f"{title}\nMAE={s['mae_ms']:.2f} ms, hit<=5ms={100*s['hit_5ms']:.1f}%")
    plt.tight_layout()
    out = FIGURES_DIR / "cross_asset_histograms.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")


def _plot_bar(cells):
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{mn}\n-> {an}" for (mn, an) in cells]
    maes   = [cells[k]["mae_ms"] for k in cells]
    colors = ["steelblue", "crimson", "coral", "forestgreen"]
    bars = ax.bar(labels, maes, color=colors)
    ax.set_ylabel("MAE (ms)")
    ax.set_title("Cross-asset evaluation summary")
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{mae:.2f} ms", ha="center", fontweight="bold")
    ax.set_ylim(0, max(maes) * 1.2)
    plt.tight_layout()
    out = FIGURES_DIR / "cross_asset_bar.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lalor-model",    required=True, type=Path)
    parser.add_argument("--halfmile-model", required=True, type=Path)
    args = parser.parse_args()
    main(args.lalor_model, args.halfmile_model)
