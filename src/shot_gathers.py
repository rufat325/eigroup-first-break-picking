"""Build shot indices and load individual shot gathers.

Each shot gather is the set of traces from a single source, sorted here by
total source-receiver offset for a clean monotonic first-break curve.
"""
import pickle

import numpy as np
import h5py

from .data_loader import (
    load_metadata, extract_constants, compute_offsets, labeled_mask,
)


def build_shot_index(hdf5_path, min_traces_per_shot=32):
    """Group traces by SHOTID and sort each group by total offset.

    Returns
    -------
    cache : dict
        Serializable dict with shot_index plus file-level constants.
    """
    meta, _ = load_metadata(hdf5_path)
    consts  = extract_constants(meta)

    offset = compute_offsets(meta, consts["coord_scale"])
    fb_ms  = meta["SPARE1"].astype(np.float64)
    is_lbl = labeled_mask(meta)
    shot_ids = meta["SHOTID"]

    index = {}
    for sid in np.unique(shot_ids):
        trace_idx = np.where(shot_ids == sid)[0]
        if len(trace_idx) < min_traces_per_shot:
            continue
        order = np.argsort(offset[trace_idx])
        sorted_idx = trace_idx[order]
        index[int(sid)] = {
            "trace_idx": sorted_idx,
            "offset":    offset[sorted_idx],
            "fb_ms":     fb_ms[sorted_idx],
            "labeled":   is_lbl[sorted_idx],
            "n_traces":  len(sorted_idx),
            "n_labeled": int(is_lbl[sorted_idx].sum()),
        }

    return {"shot_index": index, **consts}


def save_shot_index(cache, pickle_path):
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_path, "wb") as f:
        pickle.dump(cache, f)


def load_shot_index(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def load_shot_gather(hdf5_path, shot_info, time_crop=None):
    """Load a single shot gather from disk.

    Parameters
    ----------
    hdf5_path : Path
        HDF5 file containing the trace data.
    shot_info : dict
        An entry from shot_index — must contain "trace_idx" and "fb_ms".
    time_crop : int or None
        If given, truncate each trace to this many samples.

    Returns
    -------
    gather : np.ndarray (n_traces, n_samples) float32
        Traces sorted by total offset.
    fb_samples : np.ndarray (n_traces,) float32
        First-break time in native samples; NaN where unlabeled.
    """
    idx = shot_info["trace_idx"]
    # h5py reads fastest with ascending indices, so read then restore order.
    read_order = np.argsort(idx)
    inv_order  = np.argsort(read_order)

    with h5py.File(hdf5_path, "r") as f:
        arr = f["TRACE_DATA/DEFAULT/data_array"]
        raw = arr[idx[read_order], :time_crop] if time_crop else arr[idx[read_order], :]

    gather = raw[inv_order].astype(np.float32)
    return gather


def fb_samples_from_shot_info(shot_info, samp_rate_ms):
    """Convert SPARE1 ms values to native-sample indices (NaN where unlabeled)."""
    return np.where(
        shot_info["labeled"],
        shot_info["fb_ms"] / samp_rate_ms,
        np.nan,
    ).astype(np.float32)
