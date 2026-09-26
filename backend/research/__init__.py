"""Research-only NMR analysis modules (not part of the production API).

Modules moved here are used by research scripts, retrospective audits and
their tests.  Compatibility shims at the old ``app.ml.*`` paths re-export the
same module objects, so existing imports keep working while the canonical
source of truth lives under ``research/``.

Frozen research releases (v4/v5, pinned by their manifests) intentionally
remain in ``app.ml`` because moving them would invalidate the release chain.
"""
