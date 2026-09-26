"""Shared internals for the versioned NMR ML modules.

Status: production.  This package is the single source for the canonical
JSON encodings, strict schema validators, numeric guards, ridge-logistic fit
kernels, content hashing, and the fail-closed JSON-Lines sidecar process
machinery that used to be duplicated across the versioned modules in
``app.ml``.  Each versioned module now keeps only its frozen schema,
feature, and semantic differences and imports the shared behaviour here.

Every helper preserves the exact byte-level and numeric behaviour of the
implementation it replaces; ``tests/test_ml_core_golden.py`` locks those
equivalences with pre-refactor golden values.
"""
