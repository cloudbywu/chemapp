"""ctypes bridge to bundled fastmurty C library."""

from __future__ import annotations

from ctypes import (
    CDLL,
    POINTER,
    RTLD_GLOBAL,
    Structure,
    byref,
    c_bool,
    c_char_p,
    c_double,
    c_int,
)
from pathlib import Path

import numpy as np


_LIB_PATH = Path(__file__).resolve().with_name("mhtda.so")
if not _LIB_PATH.exists():
    raise RuntimeError(
        "Native Murty backend missing: mhtda.so was not bundled. "
        "Reinstall CSP5 from a wheel built for this platform."
    )

try:
    lib = CDLL(str(_LIB_PATH), RTLD_GLOBAL)
except OSError as exc:  # pragma: no cover
    raise RuntimeError(f"Failed to load native Murty backend: {_LIB_PATH} ({exc})") from exc

# fastmurty is built in sparse mode by default
sparse = True


class Solution(Structure):
    _fields_ = [("x", POINTER(c_int)), ("y", POINTER(c_int)), ("v", POINTER(c_double))]


class Subproblem(Structure):
    _fields_ = [
        ("buffer", c_char_p),
        ("m", c_int),
        ("n", c_int),
        ("rows2use", POINTER(c_int)),
        ("cols2use", POINTER(c_int)),
        ("eliminateels", POINTER(c_bool)),
        ("eliminatemiss", c_bool),
        ("solution", Solution),
    ]


class QueueEntry(Structure):
    _fields_ = [("key", c_double), ("val", POINTER(Subproblem))]


class cs_di_sparse(Structure):
    _fields_ = [
        ("nzmax", c_int),
        ("m", c_int),
        ("n", c_int),
        ("p", POINTER(c_int)),
        ("i", POINTER(c_int)),
        ("x", POINTER(c_double)),
        ("nz", c_int),
    ]


class PathTypessp(Structure):
    _fields_ = [("val", c_double), ("i", c_int), ("j", c_int)]


class WVssp(Structure):
    _fields_ = [
        ("Q", POINTER(PathTypessp)),
        ("pathback", POINTER(c_int)),
        ("m", c_int),
        ("n", c_int),
    ]


class WVsplit(Structure):
    _fields_ = [
        ("row_cost_estimates", POINTER(c_double)),
        ("row_best_columns", POINTER(c_int)),
        ("col_used", POINTER(c_bool)),
        ("m", c_int),
        ("n", c_int),
        ("m_start", c_int),
        ("n_start", c_int),
    ]


input_argtype = cs_di_sparse


class WVda(Structure):
    _fields_ = [
        ("buffer", c_char_p),
        ("m", c_int),
        ("n", c_int),
        ("nsols", c_int),
        ("solutionsize", c_int),
        ("subproblemsize", c_int),
        ("currentproblem", POINTER(Subproblem)),
        ("Q", POINTER(QueueEntry)),
        ("sspvars", WVssp),
        ("splitvars", WVsplit),
    ]


lib.da.argtypes = [
    input_argtype,
    c_int,
    POINTER(c_bool),
    POINTER(c_double),
    c_int,
    POINTER(c_bool),
    POINTER(c_double),
    c_int,
    POINTER(c_int),
    POINTER(c_double),
    POINTER(WVda),
]
lib.da.restype = c_int

allocateWorkvarsforDA = lib.allocateWorkvarsforDA
allocateWorkvarsforDA.argtypes = [c_int, c_int, c_int]
allocateWorkvarsforDA.restype = WVda

deallocateWorkvarsforDA = lib.deallocateWorkvarsforDA
deallocateWorkvarsforDA.argtypes = [WVda]


def mhtda(
    c,
    row_sets: np.ndarray,
    row_set_weights: np.ndarray,
    col_sets: np.ndarray,
    col_set_weights: np.ndarray,
    out_assocs: np.ndarray,
    out_costs: np.ndarray,
    workvars: WVda,
) -> None:
    """Feed numpy/sparse inputs to fastmurty C library."""
    if sparse:
        c_c = c[0]
    else:  # pragma: no cover
        c_c = c.ctypes.data_as(POINTER(c_double))

    row_sets_c = row_sets.ctypes.data_as(POINTER(c_bool))
    row_set_weights_c = row_set_weights.ctypes.data_as(POINTER(c_double))
    col_sets_c = col_sets.ctypes.data_as(POINTER(c_bool))
    col_set_weights_c = col_set_weights.ctypes.data_as(POINTER(c_double))
    out_assocs_c = out_assocs.ctypes.data_as(POINTER(c_int))
    out_costs_c = out_costs.ctypes.data_as(POINTER(c_double))

    nrowpriors = c_int(row_sets.shape[0])
    ncolpriors = c_int(col_sets.shape[0])
    nsols = c_int(out_assocs.shape[0])

    err = lib.da(
        c_c,
        nrowpriors,
        row_sets_c,
        row_set_weights_c,
        ncolpriors,
        col_sets_c,
        col_set_weights_c,
        nsols,
        out_assocs_c,
        out_costs_c,
        byref(workvars),
    )
    if err != 0:
        raise RuntimeError(f"fastmurty returned err={err} (not enough valid solutions)")


def sparsifyByRow(c: np.ndarray, nvalsperrow: int):
    """Create row-ordered sparse matrix expected by fastmurty."""
    m, n = c.shape
    nvalsperrow = min(int(n), int(nvalsperrow))
    nvals = int(m * nvalsperrow)

    cp = np.arange(0, nvals + 1, nvalsperrow, dtype=np.int32)
    ci = np.empty(nvals, dtype=np.int32)
    cx = np.empty(nvals, dtype=np.float64)

    for i, crow in enumerate(c):
        if nvalsperrow < n:
            colsbyvalue = np.argpartition(crow, nvalsperrow)
        else:
            colsbyvalue = np.arange(nvalsperrow)
        colsinorder = np.sort(colsbyvalue[:nvalsperrow])
        ci[i * nvalsperrow : (i + 1) * nvalsperrow] = colsinorder
        cx[i * nvalsperrow : (i + 1) * nvalsperrow] = crow[colsinorder]

    cstruct = cs_di_sparse(
        c_int(nvals),
        c_int(m),
        c_int(n),
        cp.ctypes.data_as(POINTER(c_int)),
        ci.ctypes.data_as(POINTER(c_int)),
        cx.ctypes.data_as(POINTER(c_double)),
        c_int(nvals),
    )
    # Return backing arrays to keep memory alive for ctypes call duration.
    return (cstruct, cp, ci, cx)
