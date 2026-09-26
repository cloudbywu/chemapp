from __future__ import annotations

import numpy as np
from scipy import signal, stats


def find_peaks(
    y: np.ndarray,
    x: np.ndarray | None = None,
    height: float | None = None,
    prominence: float | None = None,
    distance: int = 3,
    width: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict]:
    if height is None and prominence is None:
        prominence = 0.01 * (y.max() - y.min())
    if height is None:
        height = float(y.max() * 0.005)
    return signal.find_peaks(y, height=height, prominence=prominence, distance=distance, width=width)


def trapezoidal_integrate(x: np.ndarray, y: np.ndarray, start: float, end: float) -> float:
    lo, hi = min(start, end), max(start, end)
    mask = (x >= lo) & (x <= hi)
    if mask.sum() < 2:
        return 0.0
    return abs(float(np.trapezoid(y[mask], x[mask])))


def estimate_baseline(y: np.ndarray, percentile: float = 10.0) -> float:
    return float(np.percentile(y, percentile))


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    result = stats.linregress(x, y)
    return {
        "slope": result.slope,
        "intercept": result.intercept,
        "r_squared": result.rvalue ** 2,
        "std_err": result.stderr,
    }


def normalize_minmax(y: np.ndarray) -> np.ndarray:
    y_min = y.min()
    y_max = y.max()
    if y_max == y_min:
        return np.zeros_like(y)
    return (y - y_min) / (y_max - y_min)


def estimate_noise_region(y: np.ndarray, fraction: float = 0.1) -> float:
    n = max(int(len(y) * fraction), 10)
    segment = y[-n:]  # Take from END of data (typically noise region)
    return float(np.std(segment))


def find_nearest(x: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(x - target)))


def local_maxima(x: np.ndarray, y: np.ndarray, n: int = 10) -> list[dict]:
    peak_indices, props = find_peaks(y, height=0.01 * y.max(), prominence=0.01 * (y.max() - y.min()), distance=n)
    results = []
    for i in range(len(peak_indices)):
        idx = peak_indices[i]
        results.append({
            "position": float(x[idx]),
            "intensity": float(y[idx]),
            "index": int(idx),
            "prominence": float(props.get("prominences", [0])[i]) if "prominences" in props else 0.0,
        })
    return sorted(results, key=lambda p: p["intensity"], reverse=True)
