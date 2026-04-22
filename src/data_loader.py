"""HDF5 metadata loading and coordinate handling.

The provided datasets follow SEG-Y conventions. Scale factors work as:
    positive N : multiply raw coordinate by N
    negative N : divide raw coordinate by |N|
    0 or 1     : no change
"""
import numpy as np
import h5py

META_KEYS = [
    "SHOTID", "SOURCE_X", "SOURCE_Y", "SOURCE_HT",
    "REC_X", "REC_Y", "REC_HT",
    "SAMP_RATE", "COORD_SCALE", "HT_SCALE", "SAMP_NUM", "SPARE1",
]


def apply_scale(raw_values, scale_factor):
    """Apply SEG-Y scale factor to raw coordinate or height values."""
    sf = int(scale_factor)
    if sf == 0 or sf == 1:
        return raw_values.astype(np.float64)
    if sf < 0:
        return raw_values.astype(np.float64) / abs(sf)
    return raw_values.astype(np.float64) * sf


def load_metadata(hdf5_path):
    """Load all per-trace metadata arrays from an HDF5 file.

    Returns
    -------
    meta : dict[str, np.ndarray]
        1D arrays of length n_traces for each key in META_KEYS that exists.
    data_shape : tuple[int, int]
        Shape of data_array (n_traces, n_samples).
    """
    meta = {}
    with h5py.File(hdf5_path, "r") as f:
        grp = f["TRACE_DATA/DEFAULT"]
        for key in META_KEYS:
            if key in grp:
                meta[key] = np.asarray(grp[key][:]).squeeze()
        data_shape = grp["data_array"].shape
    return meta, data_shape


def extract_constants(meta):
    """Extract and validate the constant-per-file fields."""
    samp_rate_us = int(np.unique(meta["SAMP_RATE"])[0])
    samp_num     = int(np.unique(meta["SAMP_NUM"])[0])
    coord_scale  = int(np.unique(meta["COORD_SCALE"])[0])
    ht_scale     = int(np.unique(meta["HT_SCALE"])[0])
    return {
        "samp_rate_us": samp_rate_us,
        "samp_rate_ms": samp_rate_us / 1000.0,
        "samp_num":     samp_num,
        "coord_scale":  coord_scale,
        "ht_scale":     ht_scale,
    }


def compute_offsets(meta, coord_scale):
    """Compute source-to-receiver horizontal distance for every trace."""
    sx = apply_scale(meta["SOURCE_X"], coord_scale)
    sy = apply_scale(meta["SOURCE_Y"], coord_scale)
    rx = apply_scale(meta["REC_X"],    coord_scale)
    ry = apply_scale(meta["REC_Y"],    coord_scale)
    return np.sqrt((rx - sx) ** 2 + (ry - sy) ** 2)


def labeled_mask(meta):
    """Boolean mask of traces with a valid first-break annotation.

    SPARE1 is the first-break time in milliseconds; values of 0 or -1
    indicate unlabeled traces.
    """
    fb_ms = meta["SPARE1"].astype(np.float64)
    return fb_ms > 0
