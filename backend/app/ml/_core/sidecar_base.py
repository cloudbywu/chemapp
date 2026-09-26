"""Fail-closed JSON-Lines sidecar process machinery shared by the DP5q adapters.

Status: production.  This module is the single source for the forward
protocol error types, the persistent sidecar process lifecycle (stdout/stderr
pump threads, abort, graceful close, deadline-bound reads), the spawn
template, and the request transport that used to be duplicated between
``nmr_forward`` (mean model) and ``nmr_quantile_forward`` (99-quantile
model).

Adapter subclasses freeze their pinned contract by implementing
:meth:`SidecarAdapterBase._verify_install` and
:meth:`SidecarAdapterBase._validate_handshake` and by overriding the small
``_sidecar_*`` hooks (environment, spawn command, working directory, thread
names, message labels).  Handshake payloads, response validation, timeouts,
environment allowlists, and the quantile-disable policy remain in the
subclasses, unchanged.  The protocol exception types and the wire protocol
version are imported from the frozen :mod:`app.ml.nmr_forward` module so
every adapter raises exactly the class objects the rest of the application
catches.
"""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
from typing import Any, Mapping
import uuid

from app.ml.nmr_forward import (
    PROTOCOL_VERSION,
    NMRForwardConfigurationError,
    NMRForwardError,
    NMRForwardProtocolError,
    NMRForwardTimeoutError,
    NMRForwardUnavailableError,
)

from .hashing import sha256_file

_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_STDERR_LINES = 40
_EOF = object()


def resolve_executable(value: str | Path) -> str:
    raw = str(value)
    as_path = Path(raw).expanduser()
    if as_path.is_file():
        return str(as_path.resolve())
    located = shutil.which(raw)
    if located:
        return located
    raise NMRForwardConfigurationError(f"DP5q Python executable does not exist: {raw}")


class SidecarAdapterBase:
    """Own one persistent, serialised JSON-Lines sidecar process.

    Subclasses freeze one pinned sidecar contract; everything else --
    process lifecycle, transport framing, timed reads, shutdown -- is shared
    and behaviour-identical for every adapter.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_queue: queue.Queue[Any] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=_MAX_STDERR_LINES)
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._handshake: dict[str, Any] | None = None
        self._sidecar_code_sha256: str | None = None
        self._runtime_attestation: dict[str, str] | None = None
        self._closed = False

    # -- contract hooks --------------------------------------------------
    def _sidecar_label(self) -> str:
        """Return the frozen message label ("" or "quantile ")."""

        raise NotImplementedError

    def _verify_install(self) -> tuple[str, Path, Path]:
        raise NotImplementedError

    def _validate_handshake(self, value: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _sidecar_environment(self) -> dict[str, str]:
        raise NotImplementedError

    def _spawn_arguments(
        self, executable: str, sidecar_script: Path, repository: Path
    ) -> list[str]:
        raise NotImplementedError

    def _spawn_cwd(self, sidecar_script: Path) -> str:
        raise NotImplementedError

    def _stdout_thread_name(self) -> str:
        raise NotImplementedError

    def _stderr_thread_name(self) -> str:
        raise NotImplementedError

    # -- shared lifecycle -------------------------------------------------
    def __enter__(self) -> "SidecarAdapterBase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def handshake(self) -> dict[str, Any] | None:
        """Return a defensive copy of the verified startup metadata."""

        return json.loads(json.dumps(self._handshake)) if self._handshake else None

    def _closed_error_message(self) -> str:
        return f"DP5q {self._sidecar_label()}adapter has been closed."

    def _start_failure_message(self, exc: OSError) -> str:
        return f"Could not start the DP5q {self._sidecar_label()}sidecar: {exc}"

    def _pipe_failure_prefix(self) -> str:
        return f"DP5q {self._sidecar_label()}sidecar pipe failed: "

    def _resolved_install_paths(self) -> tuple[str, Path, Path]:
        """Resolve the executable/repository/script and validate both paths."""

        executable = resolve_executable(self.config.python_executable)
        repository = self.config.repository.expanduser().resolve()
        sidecar_script = self.config.sidecar_script.expanduser().resolve()
        if not repository.is_dir():
            raise NMRForwardConfigurationError(
                f"DP5q repository does not exist: {repository}"
            )
        if not sidecar_script.is_file():
            raise NMRForwardConfigurationError(
                f"DP5q {self._sidecar_label()}sidecar script does not exist: "
                f"{sidecar_script}"
            )
        return executable, repository, sidecar_script

    @staticmethod
    def _pump_stdout(stream: Any, output_queue: queue.Queue[Any]) -> None:
        try:
            while True:
                line = stream.readline(_MAX_RESPONSE_BYTES + 1)
                if not line:
                    output_queue.put(_EOF)
                    return
                if len(line) > _MAX_RESPONSE_BYTES:
                    output_queue.put(
                        NMRForwardProtocolError(
                            "DP5q sidecar response exceeded the size limit."
                        )
                    )
                    return
                output_queue.put(line)
        except Exception as exc:  # pragma: no cover - OS pipe edge case
            output_queue.put(exc)

    @staticmethod
    def _pump_stderr(stream: Any, stderr_tail: deque[str]) -> None:
        try:
            while True:
                line = stream.readline(4096)
                if not line:
                    return
                stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())
        except Exception:  # pragma: no cover - diagnostic channel only
            return

    def _stderr_summary(self) -> str:
        summary = "\n".join(self._stderr_tail).strip()
        return summary[-2000:] if summary else "no stderr output"

    def _abort(self) -> None:
        process = self._process
        self._process = None
        self._handshake = None
        self._sidecar_code_sha256 = None
        self._runtime_attestation = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=0.75)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.75)
        except (OSError, subprocess.SubprocessError):
            pass
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        while True:
            try:
                self._stdout_queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        with self._lock:
            self._closed = True
            process = self._process
            if process and process.poll() is None and process.stdin:
                try:
                    payload = {
                        "protocol_version": PROTOCOL_VERSION,
                        "op": "shutdown",
                        "request_id": uuid.uuid4().hex,
                        "candidates": [],
                    }
                    process.stdin.write(
                        (json.dumps(payload, separators=(",", ":")) + "\n").encode(
                            "utf-8"
                        )
                    )
                    process.stdin.flush()
                except OSError:
                    pass
            self._abort()

    def _read_json(self, timeout: float) -> dict[str, Any]:
        try:
            item = self._stdout_queue.get(timeout=timeout)
        except queue.Empty as exc:
            self._abort()
            raise NMRForwardTimeoutError(
                f"DP5q sidecar exceeded the {timeout:.3g}s deadline."
            ) from exc
        if item is _EOF:
            message = self._stderr_summary()
            self._abort()
            raise NMRForwardUnavailableError(
                "DP5q sidecar exited before responding: " + message
            )
        if isinstance(item, Exception):
            self._abort()
            if isinstance(item, NMRForwardError):
                raise item
            raise NMRForwardProtocolError(
                f"Could not read DP5q sidecar output: {item}"
            ) from item
        try:
            value = json.loads(item.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._abort()
            raise NMRForwardProtocolError(
                "DP5q sidecar returned invalid JSON."
            ) from exc
        if not isinstance(value, dict):
            self._abort()
            raise NMRForwardProtocolError(
                "DP5q sidecar response must be a JSON object."
            )
        return value

    def _start(self) -> None:
        if self._closed:
            raise NMRForwardUnavailableError(self._closed_error_message())
        if self._process is not None and self._process.poll() is None:
            return
        self._abort()
        executable, repository, sidecar_script = self._verify_install()
        self._sidecar_code_sha256 = sha256_file(sidecar_script)
        self._stdout_queue = queue.Queue()
        self._stderr_tail = deque(maxlen=_MAX_STDERR_LINES)
        env = self._sidecar_environment()
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                self._spawn_arguments(executable, sidecar_script, repository),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._spawn_cwd(sidecar_script),
                env=env,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise NMRForwardUnavailableError(self._start_failure_message(exc)) from exc
        self._process = process
        assert process.stdout is not None
        assert process.stderr is not None
        self._stdout_thread = threading.Thread(
            target=self._pump_stdout,
            args=(process.stdout, self._stdout_queue),
            name=self._stdout_thread_name(),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr,
            args=(process.stderr, self._stderr_tail),
            name=self._stderr_thread_name(),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self._handshake = self._validate_handshake(
                self._read_json(self.config.startup_timeout_seconds)
            )
        except Exception:
            self._abort()
            raise

    def _transact(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Write one JSON-Lines request and return the decoded response object.

        The caller validates the response payload; when its own validation
        raises :class:`NMRForwardProtocolError` it must ``self._abort()`` and
        re-raise, preserving the frozen per-adapter abort semantics.
        """

        encoded = (
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with self._lock:
            self._start()
            process = self._process
            assert process is not None
            assert process.stdin is not None
            try:
                process.stdin.write(encoded)
                process.stdin.flush()
            except OSError as exc:
                summary = self._stderr_summary()
                self._abort()
                raise NMRForwardUnavailableError(
                    self._pipe_failure_prefix() + summary
                ) from exc
            return self._read_json(self.config.timeout_seconds)
