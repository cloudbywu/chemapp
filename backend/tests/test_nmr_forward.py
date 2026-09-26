from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import textwrap

import pytest

from app.ml import dp5q_runtime_pin
from app.ml import nmr_forward as forward_module
from app.ml.nmr_forward import (
    DP5Q_CONFORMER_PROTOCOL_VERSION,
    DP5Q_MEAN_MODEL_SHA256,
    DP5Q_PREPROCESSOR_SHA256,
    DP5Q_REPOSITORY_COMMIT,
    NMRForwardAdapter,
    NMRForwardConfigurationError,
    NMRForwardConfig,
    NMRForwardInputError,
    NMRForwardProtocolError,
    NMRForwardTimeoutError,
    NMRForwardUnavailableError,
    dp5q_conformer_policy,
)


_FAKE_SIDECAR = """
import hashlib
import json
import os
from pathlib import Path
import sys
import time

mode = os.environ.get("FAKE_DP5Q_MODE", "ok")
handshake = {
    "type": "handshake",
    "protocol_version": 1,
    "status": "ready",
    "repository_commit": "b79968cf63cb282e8871d5595ea6cef5b4dc0d49",
    "sidecar": {
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    },
    "assets": {
        "mean_model_sha256":
            "2d453b9c340a45b7e3c8d789a0c6071167e677ccbfe972eec5a5e9c26033d095",
        "preprocessor_sha256":
            "6d143a468595797a05434a32da76cdcf57cb8b0cc929bfe27181acc297fde1b0",
    },
    "model": {
        "name": "DP5q-CASCADE-mean",
        "nucleus": "13C",
        "output": "boltzmann_weighted_mean_shift_ppm",
    },
    "capabilities": {
        "accepted_atomic_numbers": [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35],
        "quantile_enabled": False,
        "calibrated_probability": False,
        "operations": ["conformer_preflight", "predict_13c_mean"],
    },
    "runtime": {"rdkit_version": "2026.03.4"},
    "conformer_generation": {
        "protocol_version": "chemapp.dp5q-conformer.v2",
        "requested_conformers": 20,
        "random_seed": 12648430,
        "num_threads": 1,
        "rms_prune_angstrom": 0.5,
        "attempts": [
            {
                "attempt_id": "etkdgv3_standard",
                "use_random_coords": False,
                "ignore_smoothing_failures": False,
                "use_basic_knowledge": True,
                "use_experimental_torsions": True,
                "enforce_chirality": True,
                "eligibility": "all_supported_candidates",
            },
            {
                "attempt_id": "etkdgv3_random_coordinates",
                "use_random_coords": True,
                "ignore_smoothing_failures": False,
                "use_basic_knowledge": True,
                "use_experimental_torsions": True,
                "enforce_chirality": True,
                "eligibility": "all_supported_candidates",
            },
            {
                "attempt_id": "etkdgv3_relaxed_topology",
                "use_random_coords": False,
                "ignore_smoothing_failures": True,
                "use_basic_knowledge": False,
                "use_experimental_torsions": False,
                "enforce_chirality": False,
                "eligibility": "no_defined_atom_or_bond_stereochemistry",
            },
        ],
        "optimiser": "MMFF94s",
        "mmff_max_iterations": 1000,
        "energy_window_kj_mol": 10.0,
        "temperature_k": 298.15,
        "gas_constant_kj_mol_k": 0.00831446261815324,
        "population_weighting": "boltzmann",
        "terminal_failure": "fail_closed_no_uff",
    },
    "conformer_preflight": {
        "protocol_version": "chemapp.dp5q-conformer-preflight.v1",
        "candidate_failures_are_results": True,
        "operational_failures_abort_request": True,
        "uses_same_prepare_candidate_path_as_prediction": True,
    },
}
if mode == "unexpected_conformer_policy":
    handshake["conformer_generation"]["random_seed"] = 1
print(json.dumps({} if mode == "malformed_handshake" else handshake), flush=True)

for line in sys.stdin:
    request = json.loads(line)
    if request["op"] == "shutdown":
        break
    if request["op"] == "conformer_preflight":
        results = []
        for candidate in request["candidates"]:
            rejected = candidate["candidate_id"] == "reject"
            results.append({
                "candidate_id": candidate["candidate_id"],
                "status": "rejected" if rejected else "passed",
                "reason_code": "conformer_generation_failed" if rejected else None,
                "canonical_smiles": None if rejected else candidate["smiles"],
                "conformer_count": 0 if rejected else 1,
                "warnings": [],
            })
        print(json.dumps({
            "type": "conformer_preflight",
            "protocol_version": 1,
            "request_id": request["request_id"],
            "status": "ok",
            "results": results,
            "model_outputs_created": False,
        }), flush=True)
        continue
    if mode == "timeout":
        time.sleep(2)
    if mode == "malformed_response":
        print("{}", flush=True)
        continue

    predictions = []
    for candidate in request["candidates"]:
        candidate_id = candidate["candidate_id"]
        smiles = candidate["smiles"]
        carbon_count = smiles.count("C")
        if candidate_id == "sparse":
            shifts = [10.0, 20.0]
        elif candidate_id == "complete":
            shifts = [12.0, 22.0, 32.0]
        elif candidate_id == "far":
            shifts = [50.0 + 10.0 * index for index in range(carbon_count)]
        else:
            shifts = [10.0 + 10.0 * index for index in range(carbon_count)]
        atoms = [
            {"atom_index": index, "shift_ppm": shift}
            for index, shift in enumerate(shifts)
        ]
        if mode == "bad_atoms":
            atoms = [
                {"atom_index": 0, "shift_ppm": 10.0},
                {"atom_index": 0, "shift_ppm": 20.0},
            ]
        elif mode == "nonfinite":
            atoms[-1]["shift_ppm"] = float("nan")
        predictions.append(
            {
                "candidate_id": candidate_id,
                "canonical_smiles": smiles,
                "conformer_count": 2,
                "atom_predictions": atoms,
                "warnings": [f"pid:{os.getpid()}"],
            }
        )
    response = {
        "type": "prediction",
        "protocol_version": 1,
        "request_id": request["request_id"],
        "status": "ok",
        "predictions": predictions,
        "evidence_semantics": {
            "kind": "relative_13c_forward_evidence",
            "calibrated_probability": False,
            "quantile_enabled": False,
        },
    }
    print(json.dumps(response), flush=True)
"""


def _adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "ok",
    timeout_seconds: float = 1.0,
) -> NMRForwardAdapter:
    sidecar = tmp_path / "fake_dp5q_sidecar.py"
    sidecar.write_text(textwrap.dedent(_FAKE_SIDECAR), encoding="utf-8")
    monkeypatch.setenv("FAKE_DP5Q_MODE", mode)
    return NMRForwardAdapter(
        NMRForwardConfig(
            python_executable=sys.executable,
            repository=tmp_path,
            sidecar_script=sidecar,
            timeout_seconds=timeout_seconds,
            startup_timeout_seconds=1.0,
            verify_local_install=False,
        )
    )


def test_external_runtime_probe_uses_selected_python_and_strict_canonical_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "untrusted-shadow-path")
    monkeypatch.setenv("PYTHONHOME", "untrusted-python-home")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-propagate")
    monkeypatch.setenv("DP5Q_TEST_SETTING", "preserved")
    expected = forward_module._expected_runtime_versions()
    canonical = json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return forward_module.subprocess.CompletedProcess(
            command,
            0,
            stdout=canonical,
            stderr="",
        )

    monkeypatch.setattr(forward_module.subprocess, "run", fake_run)
    assert forward_module._probe_external_runtime(
        "selected-python",
        timeout_seconds=12.0,
    ) == expected
    assert calls[0][0][:3] == ["selected-python", "-I", "-c"]
    probe_env = calls[0][1]["env"]
    assert "PYTHONPATH" not in probe_env
    assert "PYTHONHOME" not in probe_env
    assert "UNRELATED_SECRET" not in probe_env
    assert probe_env["PYTHONNOUSERSITE"] == "1"
    assert probe_env["DP5Q_TEST_SETTING"] == "preserved"

    mismatched = {**expected, "numpy": "0.0.0"}
    mismatch_line = (
        json.dumps(mismatched, sort_keys=True, separators=(",", ":")) + "\n"
    )
    monkeypatch.setattr(
        forward_module.subprocess,
        "run",
        lambda command, **kwargs: forward_module.subprocess.CompletedProcess(
            command,
            0,
            stdout=mismatch_line,
            stderr="",
        ),
    )
    with pytest.raises(NMRForwardConfigurationError, match="numpy"):
        forward_module._probe_external_runtime(
            "selected-python",
            timeout_seconds=12.0,
        )


def test_sidecar_start_uses_isolated_python_and_clean_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "untrusted-shadow-path")
    monkeypatch.setenv("PYTHONHOME", "untrusted-python-home")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-propagate")
    monkeypatch.setenv("DP5Q_TEST_SETTING", "preserved")
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        raise OSError("stop after argv/env capture")

    monkeypatch.setattr(forward_module.subprocess, "Popen", fake_popen)
    adapter = _adapter(tmp_path, monkeypatch)
    with pytest.raises(NMRForwardUnavailableError, match="Could not start"):
        adapter._start()

    command = captured["command"]
    assert isinstance(command, list)
    assert command[1:3] == ["-I", "-u"]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    sidecar_env = kwargs["env"]
    assert "PYTHONPATH" not in sidecar_env
    assert "PYTHONHOME" not in sidecar_env
    assert "UNRELATED_SECRET" not in sidecar_env
    assert sidecar_env["PYTHONNOUSERSITE"] == "1"
    assert sidecar_env["DP5Q_TEST_SETTING"] == "preserved"


def test_verified_install_binds_source_bundle_and_external_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "DP5"
    model = repository / "dp5/neural_net/model.hdf5"
    preprocessor = repository / "dp5/neural_net/preprocessor.p"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    preprocessor.write_bytes(b"preprocessor")
    sidecar = tmp_path / "sidecar.py"
    sidecar.write_text("pass\n", encoding="utf-8")
    source_calls = []
    monkeypatch.setattr(
        forward_module,
        "_repository_commit",
        lambda _repository: DP5Q_REPOSITORY_COMMIT,
    )
    monkeypatch.setattr(
        dp5q_runtime_pin,
        "verified_source_bytes",
        lambda selected: source_calls.append(selected) or {"dp5/__init__.py": b""},
    )
    monkeypatch.setattr(
        forward_module,
        "_probe_external_runtime",
        lambda executable, timeout_seconds: (
            forward_module._expected_runtime_versions()
        ),
    )
    config = NMRForwardConfig(
        python_executable=sys.executable,
        repository=repository,
        sidecar_script=sidecar,
        expected_model_sha256=hashlib.sha256(b"model").hexdigest(),
        expected_preprocessor_sha256=hashlib.sha256(b"preprocessor").hexdigest(),
    )
    # The production relative paths remain fixed; redirect only this fixture's
    # module constants rather than weakening the install verifier.
    monkeypatch.setattr(
        forward_module,
        "DP5Q_MODEL_RELATIVE_PATH",
        Path("dp5/neural_net/model.hdf5"),
    )
    monkeypatch.setattr(
        forward_module,
        "DP5Q_PREPROCESSOR_RELATIVE_PATH",
        Path("dp5/neural_net/preprocessor.p"),
    )
    adapter = NMRForwardAdapter(config)
    executable, selected_repository, selected_sidecar = adapter._verify_install()

    assert Path(executable).resolve() == Path(sys.executable).resolve()
    assert selected_repository == repository.resolve()
    assert selected_sidecar == sidecar.resolve()
    assert source_calls == [repository.resolve()]
    assert adapter._runtime_attestation == {
        **forward_module._expected_runtime_versions(),
        "source_bundle_sha256": (
            dp5q_runtime_pin.DP5Q_UPSTREAM_SOURCE_BUNDLE_SHA256
        ),
        "runtime_pin_code_sha256": hashlib.sha256(
            Path(dp5q_runtime_pin.__file__).read_bytes()
        ).hexdigest(),
    }
    monkeypatch.setattr(
        dp5q_runtime_pin,
        "verified_source_bytes",
        lambda _selected: (_ for _ in ()).throw(
            dp5q_runtime_pin.DP5qRuntimePinError("changed")
        ),
    )
    with pytest.raises(NMRForwardConfigurationError, match="source bundle"):
        NMRForwardAdapter(config)._verify_install()


def test_predict_validates_formula_and_reuses_persistent_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    with _adapter(tmp_path, monkeypatch) as adapter:
        first = adapter.predict_candidates(
            [{"candidate_id": "close", "smiles": "CCO"}],
            formula="C2H6O",
        )
        second = adapter.predict_candidates(
            [{"candidate_id": "close", "smiles": "CCO"}],
            formula="C2H6O",
        )

    assert first["status"] == "ok"
    assert first["nucleus"] == "13C"
    assert first["calibrated_probability"] is False
    assert first["quantile_enabled"] is False
    assert first["model"] == {
        "name": "DP5q-CASCADE-mean",
        "repository_commit": DP5Q_REPOSITORY_COMMIT,
        "mean_model_sha256": DP5Q_MEAN_MODEL_SHA256,
        "preprocessor_sha256": DP5Q_PREPROCESSOR_SHA256,
    }
    assert first["runtime"]["conformer_generation"] == dp5q_conformer_policy()
    assert (
        first["runtime"]["conformer_generation"]["protocol_version"]
        == DP5Q_CONFORMER_PROTOCOL_VERSION
    )
    assert len(first["runtime"]["sidecar_code_sha256"]) == 64
    assert first["runtime"]["python"] == dp5q_runtime_pin.DP5Q_PYTHON_VERSION
    assert first["runtime"]["tensorflow"] == (
        dp5q_runtime_pin.DP5Q_TENSORFLOW_VERSION
    )
    assert first["runtime"]["keras"] == dp5q_runtime_pin.DP5Q_KERAS_VERSION
    assert first["runtime"]["source_bundle_sha256"] == (
        dp5q_runtime_pin.DP5Q_UPSTREAM_SOURCE_BUNDLE_SHA256
    )
    assert first["runtime"]["runtime_pin_code_sha256"] == hashlib.sha256(
        Path(dp5q_runtime_pin.__file__).read_bytes()
    ).hexdigest()
    legacy = {
        "sidecar_code_sha256": first["runtime"]["sidecar_code_sha256"],
        "rdkit_version": first["runtime"]["rdkit_version"],
    }
    assert first["runtime"]["legacy_preflight_runtime_sha256"] == (
        hashlib.sha256(
            json.dumps(
                legacy,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    assert adapter.handshake is None
    first_warning = first["predictions"][0]["warnings"][0]
    second_warning = second["predictions"][0]["warnings"][0]
    assert first_warning == second_warning


def test_prediction_rejects_mismatched_canonical_structure() -> None:
    with pytest.raises(NMRForwardProtocolError, match="requested structure"):
        NMRForwardAdapter._validate_predictions(
            [
                {
                    "candidate_id": "candidate",
                    "canonical_smiles": "COC",
                    "conformer_count": 1,
                    "atom_predictions": [
                        {"atom_index": 0, "shift_ppm": 10.0},
                        {"atom_index": 1, "shift_ppm": 20.0},
                    ],
                    "warnings": [],
                }
            ],
            [{"candidate_id": "candidate", "smiles": "CCO"}],
            {"candidate": {0, 1}},
        )


def test_conformer_preflight_returns_candidate_failures_without_model_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _adapter(tmp_path, monkeypatch) as adapter:
        result = adapter.preflight_candidates(
            [
                {"candidate_id": "pass", "smiles": "CCO"},
                {"candidate_id": "reject", "smiles": "COC"},
            ],
            formula="C2H6O",
        )

    assert result["status"] == "ok"
    assert result["model_outputs_created"] is False
    assert result["runtime"]["rdkit_version"] == "2026.03.4"
    assert (
        result["runtime"]["conformer_preflight"]["operational_failures_abort_request"]
        is True
    )
    assert [item["status"] for item in result["results"]] == [
        "passed",
        "rejected",
    ]
    assert result["results"][1]["reason_code"] == "conformer_generation_failed"


def test_empty_13c_is_unsupported_without_starting_sidecar(
    tmp_path: Path,
):
    adapter = NMRForwardAdapter(
        NMRForwardConfig(
            python_executable="does-not-exist",
            repository=tmp_path / "does-not-exist",
            sidecar_script=tmp_path / "does-not-exist.py",
        )
    )

    result = adapter.score_candidates(
        [],
        [
            {
                "candidate_id": "current-1h-candidate",
                "smiles": "CN(C)c1ccc(N(C)C)c(Br)c1",
            }
        ],
        formula="C10H15BrN2",
    )

    assert result == {
        "status": "unsupported_modality",
        "required_nucleus": "13C",
        "reason": "dp5q_mean_requires_observed_13c_resonances",
        "model_called": False,
        "calibrated_probability": False,
        "quantile_enabled": False,
        "candidates": [],
    }
    assert adapter.handshake is None


def test_formula_mismatch_is_rejected_before_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = _adapter(tmp_path, monkeypatch)

    with pytest.raises(NMRForwardInputError, match="formula mismatch"):
        adapter.predict_candidates(
            [{"candidate_id": "ethanol", "smiles": "CCO"}],
            formula="C3H8O",
        )

    assert adapter.handshake is None


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        (
            [
                {"candidate_id": f"candidate-{index}", "smiles": "CC"}
                for index in range(21)
            ],
            "At most 20 candidates",
        ),
        (
            [{"candidate_id": "too-large", "smiles": "C" * 81}],
            "heavy-atom limit",
        ),
        (
            [{"candidate_id": "too-flexible", "smiles": "C" * 25}],
            "rotatable-bond limit",
        ),
    ],
)
def test_resource_limits_are_enforced_before_model_start(
    tmp_path: Path,
    candidates: list[dict[str, str]],
    message: str,
):
    adapter = NMRForwardAdapter(
        NMRForwardConfig(
            python_executable="does-not-exist",
            repository=tmp_path / "does-not-exist",
            sidecar_script=tmp_path / "does-not-exist.py",
        )
    )

    with pytest.raises(NMRForwardInputError, match=message):
        adapter.predict_candidates(candidates)

    assert adapter.handshake is None


def test_non_string_formula_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = _adapter(tmp_path, monkeypatch)

    with pytest.raises(NMRForwardInputError, match="must be a string"):
        adapter.predict_candidates(
            [{"candidate_id": "ethanol", "smiles": "CCO"}],
            formula=123,  # type: ignore[arg-type]
        )


def test_scoring_prioritises_complete_coverage_before_low_partial_mae(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    with _adapter(tmp_path, monkeypatch) as adapter:
        result = adapter.score_candidates(
            [10.0, 20.0, 30.0],
            [
                {"candidate_id": "sparse", "smiles": "CCO"},
                {"candidate_id": "complete", "smiles": "CCC"},
            ],
        )

    assert result["diagnostic_only"] is True
    assert "symmetry-equivalent" in result["assignment_limitation"]
    assert [item["candidate_id"] for item in result["candidates"]] == [
        "complete",
        "sparse",
    ]
    complete, sparse = result["candidates"]
    assert complete["assignment_complete"] is True
    assert sparse["assignment_complete"] is False
    assert sparse["mae_ppm"] == pytest.approx(0.0)
    assert sparse["unmatched_observed_count"] == 1
    assert sparse["bidirectional_coverage"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "mode",
    ["bad_atoms", "nonfinite", "malformed_response"],
)
def test_malformed_sidecar_output_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
):
    with _adapter(tmp_path, monkeypatch, mode=mode) as adapter:
        with pytest.raises(NMRForwardProtocolError):
            adapter.predict_candidates([{"candidate_id": "ethanol", "smiles": "CCO"}])
        assert adapter.handshake is None


def test_malformed_handshake_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    with _adapter(tmp_path, monkeypatch, mode="malformed_handshake") as adapter:
        with pytest.raises(NMRForwardProtocolError):
            adapter.predict_candidates([{"candidate_id": "ethanol", "smiles": "CCO"}])
        assert adapter.handshake is None


def test_unexpected_conformer_policy_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    with _adapter(
        tmp_path,
        monkeypatch,
        mode="unexpected_conformer_policy",
    ) as adapter:
        with pytest.raises(
            NMRForwardProtocolError,
            match="conformer-generation policy",
        ):
            adapter.predict_candidates([{"candidate_id": "ethanol", "smiles": "CCO"}])
        assert adapter.handshake is None


def test_sidecar_timeout_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    with _adapter(
        tmp_path,
        monkeypatch,
        mode="timeout",
        timeout_seconds=0.05,
    ) as adapter:
        with pytest.raises(NMRForwardTimeoutError):
            adapter.predict_candidates([{"candidate_id": "ethanol", "smiles": "CCO"}])
        assert adapter.handshake is None


def test_sidecar_recovers_after_timeout_with_fresh_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    with _adapter(
        tmp_path,
        monkeypatch,
        mode="timeout",
        timeout_seconds=0.05,
    ) as adapter:
        with pytest.raises(NMRForwardTimeoutError):
            adapter.predict_candidates([{"candidate_id": "ethanol", "smiles": "CCO"}])

        monkeypatch.setenv("FAKE_DP5Q_MODE", "ok")
        result = adapter.predict_candidates(
            [{"candidate_id": "ethanol", "smiles": "CCO"}]
        )

    assert result["status"] == "ok"
