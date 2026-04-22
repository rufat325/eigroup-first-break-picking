"""Extract fixed-size training patches from shot gathers.

Patches are 256 traces x 1024 samples at 1 ms (or resampled-to-1ms) sampling.
For assets with native 2 ms sampling (Halfmile, Brunswick, Sudbury), traces
are upsampled before patching so the U-Net sees a consistent input format
and can be fine-tuned directly from a Lalor-trained starting point.
"""
import pickle

import numpy as np
import h5py
from scipy.signal import resample_poly

from .config import (
    PATCH_WIDTH, PATCH_HEIGHT, PATCH_STRIDE, MASK_THICKNESS,
    MIN_LABELED_FRAC, TRAIN_FRAC, VAL_FRAC, SPLIT_SEED,
)


def shot_split(shot_ids, seed=SPLIT_SEED):
    """Deterministic shot-level train/val/test split."""
    rng = np.random.default_rng(seed)
    ids = sorted(shot_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(TRAIN_FRAC * n)
    n_va = int(VAL_FRAC   * n)
    return (
        set(ids[:n_tr]),
        set(ids[n_tr:n_tr + n_va]),
        set(ids[n_tr + n_va:]),
    )


def _upsample_to_1ms(gather, native_ms):
    """Upsample a gather along the time axis from native_ms to 1 ms."""
    up = int(round(native_ms / 1.0))
    if up <= 1:
        return gather
    return resample_poly(gather, up=up, down=1, axis=1).astype(np.float32)


def _normalize_patch(patch):
    """Per-trace amplitude normalization with robust percentile."""
    norms = np.percentile(np.abs(patch), 99, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return np.clip(patch / norms, -5.0, 5.0).astype(np.float32)


def _build_mask(fb_samples_1ms, valid):
    """Binary first-break mask, thickness MASK_THICKNESS above and below."""
    mask = np.zeros((PATCH_WIDTH, PATCH_HEIGHT), dtype=np.uint8)
    for i in range(PATCH_WIDTH):
        if valid[i] and not np.isnan(fb_samples_1ms[i]):
            fb = int(round(fb_samples_1ms[i]))
            if 0 <= fb < PATCH_HEIGHT:
                lo = max(0, fb - MASK_THICKNESS)
                hi = min(PATCH_HEIGHT, fb + MASK_THICKNESS + 1)
                mask[i, lo:hi] = 1
    return mask


def extract_patches_from_shot(hdf5_path, shot_info, native_samp_rate_ms):
    """Generate (image, mask, fb, valid) patches covering one shot gather.

    Yields
    ------
    (patch_idx, image, mask, fb_1ms, valid) tuples.
    image and mask have shape (PATCH_WIDTH, PATCH_HEIGHT).
    """
    idx = shot_info["trace_idx"]
    read_order = np.argsort(idx)
    inv_order  = np.argsort(read_order)

    with h5py.File(hdf5_path, "r") as f:
        raw = f["TRACE_DATA/DEFAULT/data_array"][idx[read_order], :]
    gather_native = raw[inv_order].astype(np.float32)

    gather_1ms = _upsample_to_1ms(gather_native, native_samp_rate_ms)
    if gather_1ms.shape[1] >= PATCH_HEIGHT:
        gather_1ms = gather_1ms[:, :PATCH_HEIGHT]
    else:
        gather_1ms = np.pad(
            gather_1ms,
            ((0, 0), (0, PATCH_HEIGHT - gather_1ms.shape[1])),
        )

    # SPARE1 is in ms, so its value equals the 1-ms-sample index directly.
    fb_1ms = np.where(shot_info["labeled"], shot_info["fb_ms"], np.nan).astype(np.float32)
    valid  = shot_info["labeled"].astype(np.uint8)

    n_traces = gather_1ms.shape[0]
    counter  = 0
    for start in range(0, n_traces - PATCH_WIDTH + 1, PATCH_STRIDE):
        end = start + PATCH_WIDTH
        patch_valid = valid[start:end]
        if patch_valid.mean() < MIN_LABELED_FRAC:
            continue

        patch_img = _normalize_patch(gather_1ms[start:end, :])
        patch_fb  = fb_1ms[start:end]
        patch_mask = _build_mask(patch_fb, patch_valid)

        yield counter, patch_img, patch_mask, patch_fb, patch_valid
        counter += 1


def save_patch(patch_dir, shot_id, patch_idx, image, mask, fb, valid, split):
    """Save one patch as a compressed .npz file and return its manifest entry."""
    split_dir = patch_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    fname = f"shot_{shot_id:08d}_patch_{patch_idx:02d}.npz"
    fpath = split_dir / fname
    np.savez_compressed(fpath, image=image, mask=mask, fb=fb, valid=valid)
    return {
        "path":         str(fpath),
        "shot_id":      int(shot_id),
        "split":        split,
        "patch_idx":    patch_idx,
        "labeled_frac": float(valid.mean()),
    }


def save_manifest(manifest, manifest_pkl):
    with open(manifest_pkl, "wb") as f:
        pickle.dump(manifest, f)


def load_manifest(manifest_pkl):
    with open(manifest_pkl, "rb") as f:
        return pickle.load(f)
