"""Compatibility shim: implementation moved to ``research``."""

import sys

from research import nmr_calibration_cases as _impl

sys.modules[__name__] = _impl
