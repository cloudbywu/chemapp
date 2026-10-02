"""Offline tests for native runtime layout and relocation guardrails."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("staging", Path(__file__).parents[1] / "scripts/stage-runtime.py")
staging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(staging)


class StagingTests(unittest.TestCase):
    def test_native_targets_keep_their_own_layout(self):
        self.assertEqual(staging.target_config("linux", "linux", "x86_64")["python"], "python/bin/python3")
        self.assertEqual(staging.target_config("win32", "win32", "AMD64")["python"], "python/python.exe")

    def test_cross_platform_and_wrong_architecture_are_rejected(self):
        for target, host, machine in [("win32", "linux", "x86_64"), ("linux", "win32", "AMD64"), ("linux", "linux", "aarch64"), ("darwin", "darwin", "x86_64")]:
            with self.subTest(target=target, host=host, machine=machine), self.assertRaises(ValueError):
                staging.target_config(target, host, machine)

    def test_linux_copies_the_whole_standalone_distribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            install = Path(tmp)
            distribution = install / "cpython-3.11.16-linux-x86_64-gnu"
            executable = distribution / "bin/python3.11"
            executable.parent.mkdir(parents=True)
            executable.touch()
            self.assertEqual(staging.distribution_root(executable, install, "linux"), distribution)
            (install / "elsewhere").mkdir()
            with self.assertRaises(ValueError):
                staging.distribution_root(executable, install / "elsewhere", "linux")

    def test_windows_layout_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            install = Path(tmp)
            distribution = install / "cpython-windows"
            distribution.mkdir()
            executable = distribution / "python.exe"
            executable.touch()
            self.assertEqual(staging.distribution_root(executable, install, "win32"), distribution)

    def test_only_internal_relative_symlinks_are_relocatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "real").write_text("data")
            link = directory / "link"
            try:
                link.symlink_to("real")
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation requires host permission")
            staging.check_symlinks(directory)
            link.unlink()
            link.symlink_to(directory / "real")
            with self.assertRaises(ValueError):
                staging.check_symlinks(directory)


if __name__ == "__main__":
    unittest.main()
