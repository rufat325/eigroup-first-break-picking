"""Metrics for first-break picking accuracy."""
import numpy as np


def summarize(errors_ms, label=""):
    """Compute MAE, RMSE, median, and hit-rate-at-tolerance metrics.

    Hit rates are reported at millisecond tolerances so they are comparable
    across assets with different sampling rates.
    """
    errs = np.asarray(errors_ms, dtype=np.float64)
    if errs.size == 0:
        return {"method": label, "n": 0}
    return {
        "method":        label,
        "n":             int(errs.size),
        "bias_ms":       float(errs.mean()),
        "mae_ms":        float(np.abs(errs).mean()),
        "rmse_ms":       float(np.sqrt((errs ** 2).mean())),
        "median_abs_ms": float(np.median(np.abs(errs))),
        "hit_2ms":       float((np.abs(errs) <= 2).mean()),
        "hit_5ms":       float((np.abs(errs) <= 5).mean()),
        "hit_10ms":      float((np.abs(errs) <= 10).mean()),
        "hit_20ms":      float((np.abs(errs) <= 20).mean()),
    }


def print_summary(summary):
    """Pretty-print a summary dict to stdout."""
    if summary.get("n", 0) == 0:
        print(f"{summary.get('method', '')}: no samples")
        return
    print(f"=== {summary['method']} ===")
    for k in ("n", "bias_ms", "mae_ms", "rmse_ms", "median_abs_ms",
              "hit_2ms", "hit_5ms", "hit_10ms", "hit_20ms"):
        v = summary[k]
        if isinstance(v, float):
            if k.startswith("hit_"):
                print(f"  {k:<16s} {100*v:>6.1f}%")
            else:
                print(f"  {k:<16s} {v:>7.3f}")
        else:
            print(f"  {k:<16s} {v}")
