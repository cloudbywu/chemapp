"""Build a native Windows or Linux x64 relocatable desktop runtime.

Uses uv's managed standalone CPython, never a copied venv or system installation.
Build on the target OS: native Python wheels cannot be cross-packaged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "desktop" / "build"
RUNTIME = BUILD / "runtime"
SOURCE_PATHS = (
    "backend/app", "backend/vendor", "backend/scripts", "dataexample",
    "LICENSE", "NOTICE", "THIRD_PARTY_DATA.md",
)
PLATFORMS = {
    "win32": {"python": "python/python.exe", "site_packages": "Lib/site-packages"},
    "linux": {"python": "python/bin/python3", "site_packages": "lib/python3.11/site-packages"},
}


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def target_config(target, host=sys.platform, machine=None):
    machine = (machine or platform.machine()).lower()
    if target not in PLATFORMS or target != host or machine not in {"amd64", "x86_64"}:
        raise ValueError("Stage on the matching Windows or Linux x64 host; cross-packaging Python is not supported.")
    return PLATFORMS[target]


def distribution_root(executable, install, target):
    executable, install = executable.resolve(strict=True), install.resolve(strict=True)
    if not executable.is_relative_to(install):
        raise ValueError("uv did not return Python from the requested managed install directory")
    if target == "win32":
        if executable.name.lower() != "python.exe":
            raise ValueError("Expected a managed Windows python.exe")
        return executable.parent
    if executable.parent.name != "bin" or executable.name != "python3.11":
        raise ValueError("Expected managed Linux CPython 3.11")
    return executable.parent.parent


def check_symlinks(directory):
    """Only relative links whose real target stays within the distribution travel."""
    root = directory.resolve(strict=True)
    for entry in directory.rglob("*"):
        if entry.is_symlink():
            if Path(os.readlink(entry)).is_absolute() or not entry.resolve(strict=True).is_relative_to(root):
                raise ValueError(f"Runtime contains a non-relocatable symlink: {entry}")


def file_hash(source):
    with source.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=PLATFORMS, default=sys.platform)
    parser.add_argument("--python-version", default="3.11.16", help="Managed CPython 3.11 patch release (default: 3.11.16)")
    parser.add_argument("--python-install-dir", type=Path, default=Path(os.environ.get("UV_PYTHON_INSTALL_DIR", BUILD / "managed-python")))
    parser.add_argument("--with-csp5", action="store_true", help="Fetch pinned, checksum-verified CSP5 weights into the staged copy")
    args = parser.parse_args()
    try:
        config = target_config(args.platform)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not args.python_version.startswith("3.11.") or not args.python_version[5:].isdigit():
        raise SystemExit("Use an explicit CPython 3.11 patch version, such as 3.11.16.")
    if RUNTIME.exists():
        raise SystemExit(f"Staging destination already exists: {RUNTIME}. Move it aside before rebuilding.")
    BUILD.mkdir(parents=True, exist_ok=True)
    install = args.python_install_dir.resolve()
    if install == RUNTIME or install.is_relative_to(RUNTIME):
        raise SystemExit("The managed Python cache must be outside the staged runtime.")
    # Avoid changing global Python shortcuts/registration. An existing managed
    # install may be reused as a source, but nothing links back to it after copy.
    env = {**os.environ, "UV_PYTHON_INSTALL_DIR": str(install)}
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    run("uv", "python", "install", args.python_version, "--install-dir", install, "--no-bin", "--no-registry", env=env)
    executable = Path(run("uv", "python", "find", args.python_version, "--managed-python", "--no-project", "--resolve-links", env=env, capture_output=True, text=True).stdout.strip())
    source_python = distribution_root(executable, install, args.platform)
    check_symlinks(source_python)
    if (source_python / "pyvenv.cfg").exists():
        raise SystemExit("A virtual environment cannot be used as a standalone runtime.")
    runtime_python = RUNTIME / "python"
    shutil.copytree(source_python, runtime_python, symlinks=args.platform != "win32")
    python = RUNTIME / config["python"]
    requirements = BUILD / "runtime-requirements.txt"
    run("uv", "export", "--locked", "--no-dev", "--no-emit-project", "--output-file", requirements, cwd=ROOT / "backend", env=env)
    # Full native locked dependency set, including the Linux CUDA wheels. Copy
    # files rather than linking to a build cache or depending on an existing venv.
    run("uv", "pip", "install", "--python", python, "--target", runtime_python / config["site_packages"], "--link-mode", "copy", "--require-hashes", "--requirements", requirements, env=env)
    check_symlinks(runtime_python)

    # Only versioned assets/source are packaged; no local database, tokens, .env,
    # private models, caches or imports are harvested from the working tree.
    tracked = run("git", "ls-files", "-z", *SOURCE_PATHS, cwd=ROOT, capture_output=True).stdout.decode().split("\0")
    sources = {}
    for relative in sorted(set(filter(None, tracked)) | {"backend/desktop_server.py"}):
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise SystemExit(f"Refusing a missing or symlinked runtime source: {relative}")
        destination = RUNTIME / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sources[relative] = file_hash(source)
    if args.with_csp5:
        run(python, "-I", "-B", RUNTIME / "backend/scripts/fetch_csp5_weights.py", "--target", RUNTIME / "backend/vendor/csp5", env=env)
    manifest = {
        "schema_version": 2,
        "sources": sources,
        "platform": args.platform, "arch": "x64", "python": config["python"],
        "python_version": args.python_version,
        "lock_sha256": file_hash(ROOT / "backend/uv.lock"),
        "requirements_sha256": file_hash(requirements),
        "csp5_weights": args.with_csp5,
        "optional_models": "Only tracked metadata and explicitly requested CSP5 weights are included. External weights, indexes, T5 and DP5q runtimes are not bundled.",
    }
    shutil.copy2(requirements, RUNTIME / "requirements.txt")
    (RUNTIME / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Test from a different prefix with an unrelated cwd. No source venv or
    # original managed prefix is on Python's import search path (-I).
    with tempfile.TemporaryDirectory(prefix="runtime-relocated-", dir=BUILD) as moved_dir:
        moved = Path(moved_dir) / "ChemApp runtime"
        RUNTIME.rename(moved)
        try:
            run(moved / config["python"], "-I", "-B", ROOT / "desktop/scripts/check-python-runtime.py", "--runtime", moved, cwd=moved_dir, env=env)
        finally:
            moved.rename(RUNTIME)
    print(f"Staged relocatable {args.platform} x64 runtime: {RUNTIME}")
    print(f"Next: npm run pack:{'win' if args.platform == 'win32' else 'linux'}, then test the unpacked app.")


if __name__ == "__main__":
    main()
