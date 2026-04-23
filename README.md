# First-Break Picking from Hardrock Seismic Reflection Data

A deep learning pipeline for automated first-break detection on 3D seismic
surveys, with honest cross-asset generalization analysis. Submitted as a
technical task for the Junior ML/DL Algorithm Developer role at eiGroup LLC.

## Summary of results

**In-asset performance (Lalor):**

| Method         | MAE     | RMSE     | Hit ≤ 5 ms |
|----------------|---------|----------|------------|
| STA/LTA        | 32.7 ms | 110.0 ms | 62.4%      |
| **U-Net**      | **1.13 ms** | **1.71 ms** | **98.5%** |

The U-Net reduces MAE by **29×** and RMSE by **64×** over the classical
baseline, with bias that is essentially zero.

**Cross-asset evaluation (MAE in ms):**

|                    | Lalor test  | Halfmile test |
|--------------------|-------------|---------------|
| Lalor-trained U-Net     | **1.13 ms** | 13.72 ms |
| Halfmile-fine-tuned     | 4.29 ms     | **2.20 ms** |

The Halfmile-fine-tuned model retains strong performance on Lalor
(4.29 ms MAE, 84.3% hit ≤ 5 ms), demonstrating that fine-tuning did
not cause catastrophic forgetting — effectively a two-asset model.

## Project layout

```
eigroup_first_break_picking/
├── README.md
├── requirements.txt
├── src/                       # Reusable library modules
│   ├── config.py              # Asset definitions, paths, hyperparameters
│   ├── data_loader.py         # HDF5 metadata, coordinate scaling
│   ├── shot_gathers.py        # Shot-index construction and loading
│   ├── patches.py             # Patch extraction for training
│   ├── model.py               # U-Net architecture, losses
│   ├── dataset.py             # tf.data input pipeline
│   ├── inference.py           # Sliding-window prediction on full gathers
│   ├── baseline_stalta.py     # Classical STA/LTA picker
│   └── metrics.py             # MAE, hit-rate, summary helpers
├── scripts/                   # Runnable entry points (argparse-based)
│   ├── 00_download.py         # Download + decompress dataset
│   ├── 01_explore.py          # Summary statistics for an asset
│   ├── 02_build_gathers.py    # Build shot index
│   ├── 03_build_patches.py    # Extract training patches
│   ├── 04_train.py            # Train from scratch or fine-tune
│   ├── 05_evaluate.py         # STA/LTA or U-Net evaluation
│   └── 06_cross_asset_eval.py # 2×2 cross-asset matrix + plots
├── notebooks/
│   └── results_overview.ipynb # Regenerate presentation figures
├── cache/                     # Created at runtime (pickles, models, patches)
├── figures/                   # Saved plots
├── seismic_data/              # HDF5 files (downloaded, not in repo)
└── presentation/
    └── first_break_picking.pdf
```

## Installation

Requires Python 3.10 (TensorFlow 2.10 does not support Python 3.11+).
On Windows, GPU support works natively for TF 2.10 with CUDA 11.2 and
cuDNN 8.1; on Linux, any modern CUDA works.

```bash
# Create an environment (Windows example, using the py launcher)
py -3.10 -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Verify the GPU is visible:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If TensorFlow sees the GPU and lists a `PhysicalDevice`, you are set.

## Reproducing the results

Each script is a thin wrapper around library functions in `src/`. The
numbered prefix indicates execution order.

```bash
# 1. Download and decompress a dataset (one asset at a time)
python scripts/00_download.py --asset lalor
python scripts/00_download.py --asset halfmile

# 2. Per-asset exploration
python scripts/01_explore.py --asset lalor
python scripts/01_explore.py --asset halfmile

# 3. Build shot indices (one-time per asset, ~10s each)
python scripts/02_build_gathers.py --asset lalor
python scripts/02_build_gathers.py --asset halfmile

# 4. Extract training patches (~5–7 min per asset)
python scripts/03_build_patches.py --asset lalor
python scripts/03_build_patches.py --asset halfmile

# 5a. Train Lalor U-Net from scratch (~3 h on an RTX 4070)
python scripts/04_train.py --asset lalor \
    --output cache/unet_lalor.keras

# 5b. Fine-tune on Halfmile (~50 min)
python scripts/04_train.py --asset halfmile \
    --init cache/unet_lalor.keras --finetune \
    --output cache/unet_halfmile_ft.keras

# 6. Classical baseline on Lalor
python scripts/05_evaluate.py --method stalta --asset lalor --save

# 7. U-Net evaluation on each asset
python scripts/05_evaluate.py --method unet --asset lalor \
    --model cache/unet_lalor.keras --save
python scripts/05_evaluate.py --method unet --asset halfmile \
    --model cache/unet_halfmile_ft.keras --save

# 8. Full 2×2 cross-asset evaluation
python scripts/06_cross_asset_eval.py \
    --lalor-model cache/unet_lalor.keras \
    --halfmile-model cache/unet_halfmile_ft.keras
```

If a pretrained `cache/unet_halfmile_ft.keras` is provided, steps 5a and
5b can be skipped and inference run directly.

## Methodology highlights

**Data pipeline.** Raw HDF5 files contain a flat (n_traces × n_samples)
array with per-trace metadata columns. Traces are grouped by `SHOTID` and
sorted by total source-receiver offset to produce clean 2D shot gathers
with monotonic first-break curves.

**Patches.** Each shot gather is split into overlapping 256 × 1024 patches
(stride 128). For non-Lalor assets with 2 ms sampling, traces are
polyphase-upsampled to 1 ms before patching, keeping the U-Net input
format consistent across assets and enabling fine-tuning without any
architecture changes. Per-patch amplitude is percentile-normalized
per-trace so far-offset traces with weaker signal are not drowned by
near-offset strong arrivals.

**Model.** A 4-level U-Net (~7.8 M parameters, base 32) with batch
normalization. Output is a single-channel sigmoid of the same spatial
size as the input, representing per-pixel probability of being on the
first-break line. The target mask uses ±2 samples of thickness around
the true pick.

**Loss.** Combined weighted binary cross-entropy (positive-class weight
50) and soft Dice. The weighted BCE handles the severe class imbalance
(first-break pixels are ~0.5% of the mask) and Dice provides a
complementary gradient signal at low recall.

**Inference.** The trained model is applied in a sliding-window fashion
across a full shot gather; overlapping patch predictions are averaged
and the first-break pick per trace is extracted as the argmax in time,
with a minimum-probability gate filtering out dead traces.

**Fine-tuning.** Transfer from Lalor to Halfmile starts from the trained
Lalor weights with all layers unfrozen, a one-order-of-magnitude-lower
learning rate (1e-4) and at most 10 epochs. Pre-fine-tuning validation
is evaluated up-front so the improvement from fine-tuning is explicit.

## Known limitations

- **Two assets, not four.** Brunswick and Sudbury are not included in
  the evaluation; the pipeline is asset-agnostic and they can be added
  by the same scripts, but results were not validated on them within
  the time budget.
- **Sampling rate mismatch handled via upsampling**, which is a
  pragmatic choice — training a native-2-ms variant might give slightly
  different results.
- **Picking convention difference** between Lalor (onset) and Halfmile
  (trough) is learned implicitly during fine-tuning rather than handled
  as a modeling assumption. A convention-aware preprocessing step is a
  natural next step.
- **No uncertainty estimation.** Production first-break picking
  typically wants per-pick confidence so a geophysicist can triage
  edge cases; adding MC-dropout or a calibrated threshold would be
  straightforward.

## Attribution

Datasets provided by (eiGroup LLC technical task) and
originally compiled by researchers at Natural Resources Canada, Glencore,
and Trevali Mining Corporation; see the Mila-IQIA HardPicks benchmark
(GEOPHYSICS 2024) for the published version of these data.

Author: Rufat Mahmudov
Date: April 2026
