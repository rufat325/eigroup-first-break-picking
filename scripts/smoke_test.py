"""Minimal smoke test: verifies imports, config, and model instantiation.

Does NOT require any data files. Run this first after installation to
catch environment problems before downloading multi-GB HDF5 files.

Usage:
    python scripts/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    print("Checking imports...")
    import numpy as np
    import h5py
    import scipy
    import matplotlib
    import tensorflow as tf
    print(f"  numpy={np.__version__}, h5py={h5py.__version__}, "
          f"scipy={scipy.__version__}, tf={tf.__version__}")

    print("\nChecking GPU...")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  {len(gpus)} GPU(s) detected: {[g.name for g in gpus]}")
    else:
        print("  WARNING: no GPU detected; training will be very slow")

    print("\nChecking project config...")
    from src.config import ASSETS, PATCH_WIDTH, PATCH_HEIGHT
    print(f"  {len(ASSETS)} assets configured")
    print(f"  Patch size: {PATCH_WIDTH} traces x {PATCH_HEIGHT} samples")

    print("\nBuilding U-Net (checks architecture + losses)...")
    from src.model import build_unet, combined_loss, dice_coef
    model = build_unet()
    print(f"  Parameters: {model.count_params():,}")

    print("\nRunning a forward pass on random input...")
    x = np.random.randn(1, PATCH_WIDTH, PATCH_HEIGHT, 1).astype(np.float32)
    y = model(x, training=False).numpy()
    print(f"  Output shape: {y.shape}, range [{y.min():.3f}, {y.max():.3f}]")

    print("\nChecking STA/LTA baseline...")
    from src.baseline_stalta import sta_lta, pick_first_break
    traces = np.random.randn(5, 1024).astype(np.float32)
    ratio = sta_lta(traces)
    picks = pick_first_break(ratio, threshold=3.0)
    print(f"  STA/LTA output shape: {ratio.shape}, picks: {picks}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
