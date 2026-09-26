"""Convenience entry point for the ChemApp API."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("CHEMAPP_HOST", "127.0.0.1"),
        port=int(os.environ.get("CHEMAPP_PORT", "8000")),
        reload=os.environ.get("CHEMAPP_RELOAD", "").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
