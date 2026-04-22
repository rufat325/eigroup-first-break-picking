"""STA/LTA (Short-Term-Average / Long-Term-Average) classical first-break picker.

A standard algorithm in seismology that flags the first sample at which a
short-window energy estimate exceeds a multiple of the long-window energy
baseline — i.e. the point where signal energy rises above noise.
"""
import numpy as np


def sta_lta(traces, sta_len=20, lta_len=200):
    """Compute STA/LTA ratio for each row in a batch of traces.

    Parameters
    ----------
    traces : np.ndarray (n_traces, n_samples)
    sta_len, lta_len : int
        Window lengths in samples.

    Returns
    -------
    ratio : np.ndarray (n_traces, n_samples) float32
        STA/LTA ratio. Samples before index lta_len are zero.
    """
    energy = traces.astype(np.float32) ** 2
    n, T = energy.shape

    cumsum = np.concatenate(
        [np.zeros((n, 1), dtype=np.float64), np.cumsum(energy, axis=1)],
        axis=1,
    )

    def windowed_mean(win):
        return (cumsum[:, win:] - cumsum[:, :-win]) / win

    sta_mean = windowed_mean(sta_len)
    lta_mean = windowed_mean(lta_len)

    sta_full = np.zeros_like(energy); sta_full[:, sta_len - 1:] = sta_mean
    lta_full = np.zeros_like(energy); lta_full[:, lta_len - 1:] = lta_mean

    ratio = sta_full / np.maximum(lta_full, 1e-12)
    ratio[:, :lta_len] = 0.0
    return ratio.astype(np.float32)


def pick_first_break(ratio, threshold=3.0, search_start=0, search_end=None):
    """Index of the first sample whose STA/LTA ratio exceeds threshold.

    Returns NaN for traces where the threshold is never crossed.
    """
    n, T = ratio.shape
    search_end = search_end if search_end is not None else T
    window = ratio[:, search_start:search_end]
    above = window > threshold

    has_any = above.any(axis=1)
    picks = np.full(n, np.nan, dtype=np.float32)
    picks[has_any] = np.argmax(above[has_any], axis=1) + search_start
    return picks
