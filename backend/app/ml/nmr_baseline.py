"""Compatibility shim: implementation moved to ``research``."""

import sys

from research import nmr_baseline as _impl

sys.modules[__name__] = _impl
