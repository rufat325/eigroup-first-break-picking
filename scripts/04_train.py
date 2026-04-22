"""Train a U-Net from scratch, or fine-tune an existing model, on an asset.

Usage:
    # Train Lalor from scratch:
    python scripts/04_train.py --asset lalor --output cache/unet_lalor.keras

    # Fine-tune Lalor model on Halfmile:
    python scripts/04_train.py --asset halfmile \\
        --init cache/unet_lalor.keras --finetune \\
        --output cache/unet_halfmile_ft.keras
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ASSETS, CACHE_DIR,
    LR_FROM_SCRATCH, LR_FINETUNE,
    EPOCHS_SCRATCH, EPOCHS_FINETUNE,
)
from src.model import build_unet, combined_loss, dice_coef, CUSTOM_OBJECTS
from src.dataset import build_dataset
from src.patches import load_manifest


def main(asset, output_path, init_path=None, finetune=False, epochs=None, lr=None):
    # Enable GPU memory growth so TF doesn't pre-allocate all VRAM.
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    cfg = ASSETS[asset]
    if not cfg["manifest_pkl"].exists():
        print(f"ERROR: run scripts/03_build_patches.py --asset {asset} first.")
        sys.exit(1)

    manifest = load_manifest(cfg["manifest_pkl"])
    train_paths = [e["path"] for e in manifest if e["split"] == "train"]
    val_paths   = [e["path"] for e in manifest if e["split"] == "val"]
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    train_ds = build_dataset(train_paths, shuffle=True,  augment=True)
    val_ds   = build_dataset(val_paths,   shuffle=False, augment=False)

    if init_path is not None:
        print(f"Loading initial weights from {init_path}")
        model = tf.keras.models.load_model(
            str(init_path), custom_objects=CUSTOM_OBJECTS, compile=False,
        )
    else:
        print("Building U-Net from scratch")
        model = build_unet()

    learning_rate = lr or (LR_FINETUNE if finetune else LR_FROM_SCRATCH)
    n_epochs      = epochs or (EPOCHS_FINETUNE if finetune else EPOCHS_SCRATCH)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=combined_loss,
        metrics=[dice_coef],
    )
    print(f"Parameters: {model.count_params():,}")

    if init_path is not None:
        print("\nPre-training val performance on target asset:")
        pre = model.evaluate(val_ds, verbose=1)
        print(f"  val_loss={pre[0]:.4f}, val_dice={pre[1]:.4f}")

    callbacks = [
        ModelCheckpoint(str(output_path), monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=2 if finetune else 3,
                          min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss",
                      patience=4 if finetune else 7,
                      restore_best_weights=True, verbose=1),
    ]

    print(f"\nTraining — {n_epochs} epochs, lr={learning_rate}")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

    history_path = output_path.with_suffix(".history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    print(f"Model  -> {output_path}")
    print(f"History-> {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset",    required=True, choices=list(ASSETS.keys()))
    parser.add_argument("--output",   required=True, type=Path)
    parser.add_argument("--init",     default=None,  type=Path,
                        help="Path to model to load weights from (optional).")
    parser.add_argument("--finetune", action="store_true",
                        help="Use fine-tuning hyperparameters (lower LR, fewer epochs).")
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--lr",       type=float, default=None)
    args = parser.parse_args()
    main(args.asset, args.output, args.init, args.finetune, args.epochs, args.lr)
