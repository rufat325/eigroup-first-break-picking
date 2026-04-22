"""Run U-Net inference over a full shot gather using overlapping patches.

Predictions from overlapping patches are averaged. The first-break sample
for each trace is then extracted as the argmax along the time axis, with
a minimum-probability gate to filter out dead or ambiguous traces.
"""
import numpy as np
from scipy.signal import resample_poly

from .config import (
    PATCH_WIDTH, PATCH_HEIGHT, INFERENCE_STRIDE, INFERENCE_BATCH,
    MIN_PROB_THRESHOLD,
)


def _normalize_for_inference(gather):
    """Per-trace percentile normalization, same as training."""
    norms = np.percentile(np.abs(gather), 99, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return np.clip(gather / norms, -5.0, 5.0).astype(np.float32)


def _prepare_input(gather, native_samp_rate_ms):
    """Upsample to 1 ms (if needed) and crop/pad to PATCH_HEIGHT."""
    up = int(round(native_samp_rate_ms / 1.0))
    gather_1ms = (
        resample_poly(gather, up=up, down=1, axis=1).astype(np.float32)
        if up > 1 else gather.astype(np.float32)
    )
    if gather_1ms.shape[1] >= PATCH_HEIGHT:
        gather_1ms = gather_1ms[:, :PATCH_HEIGHT]
    else:
        gather_1ms = np.pad(
            gather_1ms,
            ((0, 0), (0, PATCH_HEIGHT - gather_1ms.shape[1])),
        )
    return _normalize_for_inference(gather_1ms)


def predict_shot(
    gather,
    model,
    native_samp_rate_ms,
    patch_width=PATCH_WIDTH,
    stride=INFERENCE_STRIDE,
    batch_size=INFERENCE_BATCH,
    min_prob=MIN_PROB_THRESHOLD,
):
    """Predict first-break sample for every trace in a shot gather.

    Returns
    -------
    fb_pred_1ms : np.ndarray (n_traces,) float32
        Predicted first-break index in 1-ms-sample units.
        Divide by (native_samp_rate_ms / 1.0) to get native-sample indices.
        NaN where model confidence is below `min_prob`.
    prob_map : np.ndarray (n_traces, PATCH_HEIGHT) float32
        Per-trace averaged probability map (useful for visualization).
    """
    img = _prepare_input(gather, native_samp_rate_ms)
    n_traces = img.shape[0]

    prob_sum = np.zeros((n_traces, PATCH_HEIGHT), dtype=np.float32)
    prob_cnt = np.zeros((n_traces,),              dtype=np.int32)

    starts = list(range(0, n_traces - patch_width + 1, stride))
    if starts and starts[-1] + patch_width < n_traces:
        starts.append(n_traces - patch_width)

    for i in range(0, len(starts), batch_size):
        bs = starts[i:i + batch_size]
        batch = np.stack([img[s:s + patch_width, :] for s in bs], axis=0)[..., None]
        preds = model.predict(batch, verbose=0)[..., 0]
        for s, p in zip(bs, preds):
            prob_sum[s:s + patch_width, :] += p
            prob_cnt[s:s + patch_width]    += 1

    prob_map  = prob_sum / np.maximum(prob_cnt[:, None], 1)
    max_probs = prob_map.max(axis=1)
    argmax_t  = prob_map.argmax(axis=1).astype(np.float32)
    fb_pred_1ms = np.where(max_probs > min_prob, argmax_t, np.nan)
    return fb_pred_1ms, prob_map
