"""Compatibility shim: implementation moved to ``research``."""

import sys

from research import independent_nmr_data as _impl

sys.modules[__name__] = _impl
