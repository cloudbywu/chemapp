"""Check a moved standalone runtime with isolated Python, without uv or a venv."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

# Every direct production dependency plus relevant native/scientific extensions.
IMPORTS = (
    "aiofiles", "cryptography", "fastapi", "joblib", "lxml.etree", "nmrglue",
    "numpy", "openai", "pandas", "docx", "multipart", "rdkit.Chem", "requests",
    "sklearn", "scipy", "sentencepiece", "torch", "tqdm", "transformers", "uvicorn",
    "sqlite3", "ssl", "ctypes", "bz2", "lzma",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    prefix = runtime / "python"
    manifest = json.loads((runtime / "manifest.json").read_text(encoding="utf-8"))
    assert sys.flags.isolated and sys.flags.no_user_site, "Use -I to isolate Python"
    assert sys.maxsize > 2**32 and sys.platform == manifest["platform"]
    assert Path(sys.prefix).resolve() == prefix == Path(sys.base_prefix).resolve(), "Not a standalone prefix"
    assert Path(sys.executable).resolve().is_relative_to(prefix), "Interpreter escaped the bundle"
    assert not (prefix / "pyvenv.cfg").exists(), "A venv is not relocatable"
    if manifest.get("python_version"):
        assert sys.version.split()[0] == manifest["python_version"]
    for search_path in sys.path:
        assert Path(search_path).resolve().is_relative_to(prefix), f"External import path: {search_path}"
    for name in IMPORTS:
        module = importlib.import_module(name)
        if getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(prefix), f"External module: {name}"
    # Check *every* active requirement exported from uv.lock, not just the imports.
    # pip's bundled parser is from the clean standalone CPython distribution.
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
    requirement_bytes = (runtime / "requirements.txt").read_bytes()
    assert hashlib.sha256(requirement_bytes).hexdigest() == manifest["requirements_sha256"]
    requirements = requirement_bytes.decode("utf-8")
    checked = {}
    for line in requirements.splitlines():
        text = line.strip()
        if not text or text.startswith(("#", "--")):
            continue
        requirement = Requirement(text.removesuffix("\\").rstrip())
        if requirement.marker and not requirement.marker.evaluate():
            continue
        installed = importlib.metadata.distribution(requirement.name)
        assert installed.version in requirement.specifier, f"Wrong locked version: {requirement}"
        assert Path(installed.locate_file("")).resolve().is_relative_to(prefix)
        checked[canonicalize_name(requirement.name)] = installed.version
    assert checked, "No locked runtime requirements were checked"
    print(json.dumps({"python": sys.version.split()[0], "prefix": str(prefix), "locked_distributions": len(checked), "imports": len(IMPORTS), "torch": checked.get("torch")}, indent=2))


if __name__ == "__main__":
    main()
