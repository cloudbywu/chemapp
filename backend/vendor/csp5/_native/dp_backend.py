"""ctypes wrapper for native DP matching backend."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


_LIB_PATH = Path(__file__).resolve().with_name("libmatching_dp.so")
if not _LIB_PATH.exists():
    raise RuntimeError(
        "Native DP backend missing: libmatching_dp.so was not bundled. "
        "Reinstall CSP5 from a wheel built for this platform."
    )

try:
    _LIB = ctypes.CDLL(str(_LIB_PATH))
except OSError as exc:  # pragma: no cover
    raise RuntimeError(f"Failed to load native DP backend: {_LIB_PATH} ({exc})") from exc

_MATCH_DP = _LIB.nmrexp_match_indices_dp
_MATCH_DP.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
_MATCH_DP.restype = ctypes.c_int


def match_indices_dp(
    pred_vals: Sequence[float] | np.ndarray,
    obs_vals: Sequence[float] | np.ndarray,
    *,
    dummy_cost: float,
    row_penalties: Sequence[float] | np.ndarray | None = None,
) -> Tuple[List[int], List[int]]:
    """Run native DP matcher and return matched row/column indices."""
    pred_arr = np.asarray(pred_vals, dtype=np.float64).reshape(-1)
    obs_arr = np.asarray(obs_vals, dtype=np.float64).reshape(-1)
    n_pred = int(pred_arr.shape[0])
    n_obs = int(obs_arr.shape[0])

    if n_pred == 0 or n_obs == 0:
        return [], []

    penalties_arr = None
    penalties_ptr = None
    if row_penalties is not None:
        penalties_arr = np.asarray(row_penalties, dtype=np.float64).reshape(-1)
        if penalties_arr.shape[0] != n_pred:
            raise ValueError(
                f"row_penalties length mismatch: got {penalties_arr.shape[0]}, expected {n_pred}"
            )
        penalties_ptr = penalties_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    out_size = int(min(n_pred, n_obs))
    out_rows = np.empty(out_size, dtype=np.int32)
    out_cols = np.empty(out_size, dtype=np.int32)

    count = _MATCH_DP(
        pred_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_pred),
        obs_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_obs),
        ctypes.c_double(float(dummy_cost)),
        penalties_ptr,
        out_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(out_size),
    )

    if count < 0:
        raise RuntimeError(f"Native DP backend returned invalid count={count}")
    if count == 0:
        return [], []

    return out_rows[:count].tolist(), out_cols[:count].tolist()
