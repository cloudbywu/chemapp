from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import textwrap

import pytest

from app.ml.nmr_forward import (
    NMRForwardInputError,
    NMRForwardProtocolError,
)
from app.ml import dp5q_runtime_pin as runtime_pin
from app.ml.nmr_quantile_forward import (
    DP5Q_QUANTILE_ARCHIVE_SHA256,
    DP5Q_QUANTILE_LEVELS,
    DP5Q_QUANTILE_SCORE_SEMANTICS,
    DP5Q_UNASSIGNED_SHADOW_SEMANTICS,
    NMRQuantileForwardAdapter,
    NMRQuantileForwardConfig,
    dp5q_distribution_contract,
    score_assigned_prediction,
)


_FAKE_SIDECAR = """
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

mode = "__FAKE_MODE__"
levels = [index / 100 for index in range(1, 100)]
contract = __DISTRIBUTION_CONTRACT__
handshake = {
    "type": "handshake",
    "protocol_version": 1,
    "status": "ready",
    "repository_commit": "b79968cf63cb282e8871d5595ea6cef5b4dc0d49",
    "sidecar": {
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mean_contract_code_sha256": "__MEAN_CONTRACT_SHA__",
        "runtime_pin_code_sha256": "__RUNTIME_PIN_SHA__",
    },
    "assets": {
        "quantile_archive_sha256": "__QUANTILE_SHA__",
        "preprocessor_sha256":
            "6d143a468595797a05434a32da76cdcf57cb8b0cc929bfe27181acc297fde1b0",
        "upstream_source_bundle_sha256": "__SOURCE_BUNDLE_SHA__",
    },
    "model": {
        "name": "DP5q-CASCADE-99quantiles",
        "nucleus": "13C",
        "output": "per_conformer_normal_fit_and_boltzmann_quantile_summary",
        "quantile_levels": levels,
        "raw_quantile_tensor_digest":
            "sha256-sorted-population-tensor-records-float64-le-v1",
    },
    "capabilities": {
        "accepted_atomic_numbers": [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35],
        "quantile_enabled": True,
        "calibrated_probability": False,
        "operations": ["conformer_preflight", "predict_13c_quantiles"],
    },
    "runtime": {
        "archive_checkpoint_path_separator": "posix",
        "keras_version": "2.14.0",
        "numpy_version": "1.26.4",
        "pandas_version": "2.2.3",
        "python_version": "3.11.15",
        "rdkit_version": "2026.03.4",
        "scikit_learn_version": "1.3.2",
        "scipy_version": "1.11.4",
        "tensorflow_version": "2.14.0",
        "tqdm_version": "4.67.3",
    },
    "conformer_generation": __CONFORMER_POLICY__,
    "conformer_preflight": {
        "protocol_version": "chemapp.dp5q-conformer-preflight.v1",
        "candidate_failures_are_results": True,
        "operational_failures_abort_request": True,
        "uses_same_prepare_candidate_path_as_prediction": True,
    },
    "distribution_fit": contract,
}
if mode == "bad_distribution_contract":
    handshake["distribution_fit"]["family"] = "logistic"
if mode == "malformed_handshake":
    handshake = {}
print(json.dumps(handshake), flush=True)

for line in sys.stdin:
    request = json.loads(line)
    if request["op"] == "shutdown":
        break
    if mode == "timeout":
        time.sleep(2)
    if request["op"] == "conformer_preflight":
        results = []
        for candidate in request["candidates"]:
            rejected = (
                mode == "preflight_rejection"
                and candidate == request["candidates"][0]
            )
            results.append({
                "candidate_id": candidate["candidate_id"],
                "status": "rejected" if rejected else "passed",
                "reason_code": "mmff_parameters_unavailable" if rejected else None,
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
    predictions = []
    for candidate in request["candidates"]:
        atom_count = candidate["smiles"].count("C")
        base = 60.0 if candidate["candidate_id"] == "far" else 10.0
        atoms = []
        for atom_index in range(atom_count):
            median = base + 10.0 * atom_index
            values = [
                median + (quantile_index - 49) * 0.1
                for quantile_index in range(99)
            ]
            if mode == "crossing_quantiles" and atom_index == 0:
                values[50] = values[49] - 10.0
            atoms.append({
                "atom_index": atom_index,
                "quantiles_ppm": values,
                "conformer_mu_ppm": [median],
                "conformer_sigma_ppm": [5.0],
                "conformer_quantile_crossing_counts": [0],
                "conformer_max_quantile_crossing_ppm": [0.0],
            })
        populations = [0.5] if mode == "bad_populations" else [1.0]
        predictions.append({
            "candidate_id": candidate["candidate_id"],
            "canonical_smiles": (
                "C" if mode == "wrong_canonical" else candidate["smiles"]
            ),
            "conformer_count": 1,
            "conformer_populations": populations,
            "raw_conformer_quantiles_shape": [1, atom_count, 99],
            "raw_conformer_quantiles_atom_indices": list(range(atom_count)),
            "raw_conformer_quantiles_sha256": (
                "g" * 64 if mode == "bad_raw_digest" else "0" * 64
            ),
            "atom_predictions": atoms,
            "warnings": [
                (
                    "secret:" + os.environ.get("OPENAI_API_KEY", "missing")
                    if mode == "environment_probe"
                    else f"pid:{os.getpid()}"
                )
            ],
        })
    semantics = {
        "kind": "dp5q_13c_quantile_components",
        "calibrated_probability": False,
        "quantile_enabled": True,
        "assigned_observations_required_for_equation_parity_score": True,
    }
    if mode == "bad_semantics":
        semantics["calibrated_probability"] = True
    print(json.dumps({
        "type": "prediction",
        "protocol_version": 1,
        "request_id": request["request_id"],
        "status": "ok",
        "predictions": predictions,
        "evidence_semantics": semantics,
    }), flush=True)
"""

_FAKE_MEAN_CONTRACT = b"# fixed fake mean contract for protocol tests\n"


def _fake_sidecar_source(mode: str) -> str:
    from app.ml.nmr_forward import dp5q_conformer_policy

    return (
        textwrap.dedent(_FAKE_SIDECAR)
        .replace("__QUANTILE_SHA__", DP5Q_QUANTILE_ARCHIVE_SHA256)
        .replace(
            "__DISTRIBUTION_CONTRACT__",
            repr(dp5q_distribution_contract()),
        )
        .replace("__CONFORMER_POLICY__", repr(dp5q_conformer_policy()))
        .replace(
            "__MEAN_CONTRACT_SHA__",
            hashlib.sha256(_FAKE_MEAN_CONTRACT).hexdigest(),
        )
        .replace(
            "__RUNTIME_PIN_SHA__",
            hashlib.sha256(Path(runtime_pin.__file__).read_bytes()).hexdigest(),
        )
        .replace(
            "__SOURCE_BUNDLE_SHA__",
            runtime_pin.DP5Q_UPSTREAM_SOURCE_BUNDLE_SHA256,
        )
        .replace("__FAKE_MODE__", mode)
    )


def _adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "ok",
    timeout_seconds: float = 1.0,
) -> NMRQuantileForwardAdapter:
    sidecar = tmp_path / "fake_dp5q_quantile_sidecar.py"
    sidecar.write_text(_fake_sidecar_source(mode), encoding="utf-8")
    (tmp_path / "dp5q_sidecar.py").write_bytes(_FAKE_MEAN_CONTRACT)
    return NMRQuantileForwardAdapter(
        NMRQuantileForwardConfig(
            python_executable=sys.executable,
            repository=tmp_path,
            sidecar_script=sidecar,
            timeout_seconds=timeout_seconds,
            startup_timeout_seconds=1.0,
            verify_local_install=False,
        )
    )


def _prediction(
    *,
    populations: list[float] | None = None,
) -> dict[str, object]:
    populations = populations or [0.25, 0.75]
    return {
        "candidate_id": "candidate",
        "canonical_smiles": "C",
        "conformer_count": len(populations),
        "conformer_populations": populations,
        "atom_predictions": [
            {
                "atom_index": 0,
                "quantiles_ppm": [float(index) for index in range(99)],
                "conformer_mu_ppm": [10.0, 12.0],
                "conformer_sigma_ppm": [2.0, 3.0],
                "conformer_quantile_crossing_counts": [0, 0],
                "conformer_max_quantile_crossing_ppm": [0.0, 0.0],
            }
        ],
        "warnings": [],
    }


def test_assigned_score_matches_released_dp5q_equations() -> None:
    result = score_assigned_prediction(_prediction(), {0: 11.0})

    cdf_first = 0.5 * math.erfc(-(11.0 - 10.0) / (2.0 * math.sqrt(2.0)))
    cdf_second = 0.5 * math.erfc(-(11.0 - 12.0) / (3.0 * math.sqrt(2.0)))
    expected_atom = 1.0 - (
        0.25 * abs(1.0 - 2.0 * cdf_first)
        + 0.75 * abs(1.0 - 2.0 * cdf_second)
    )

    assert result["score_semantics"] == DP5Q_QUANTILE_SCORE_SEMANTICS
    assert result["official_assignment_semantics"] is True
    assert result["official_equation_parity"] is True
    assert result["official_workflow_parity"] is False
    assert result["conformer_protocol"] == "chemapp.dp5q-conformer.v2"
    assert result["calibrated_probability"] is False
    assert result["atom_scores"][0]["dp5q_atom_score"] == pytest.approx(
        expected_atom
    )
    assert result["dp5q_method_score"] == pytest.approx(expected_atom + 1e-6)


def test_assigned_score_rejects_unknown_or_malformed_assignments() -> None:
    with pytest.raises(NMRForwardInputError, match="absent"):
        score_assigned_prediction(_prediction(), {5: 10.0})
    with pytest.raises(NMRForwardInputError, match="At least one"):
        score_assigned_prediction(_prediction(), {})
    with pytest.raises(NMRForwardInputError, match="outside"):
        score_assigned_prediction(_prediction(), {0: float("nan")})


def test_predict_validates_quantiles_and_reuses_persistent_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _adapter(tmp_path, monkeypatch) as adapter:
        first = adapter.predict_candidates(
            [{"candidate_id": "close", "smiles": "CCC"}],
            formula="C3H8",
        )
        second = adapter.predict_candidates(
            [{"candidate_id": "close", "smiles": "CCC"}],
            formula="C3H8",
        )

    assert first["status"] == "ok"
    assert first["quantile_enabled"] is True
    assert first["calibrated_probability"] is False
    assert first["model"]["quantile_archive_sha256"] == (
        DP5Q_QUANTILE_ARCHIVE_SHA256
    )
    assert first["runtime"]["distribution_fit"] == dp5q_distribution_contract()
    assert first["predictions"][0]["raw_conformer_quantiles_shape"] == [
        1,
        3,
        99,
    ]
    assert (
        first["predictions"][0]["raw_conformer_quantiles_sha256"]
        == "0" * 64
    )
    assert first["predictions"][0]["atom_predictions"][0]["quantiles_ppm"][49] == (
        10.0
    )
    assert (
        first["predictions"][0]["warnings"][0]
        == second["predictions"][0]["warnings"][0]
    )


def test_unassigned_shadow_is_separate_from_official_assignment_and_probability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _adapter(tmp_path, monkeypatch) as adapter:
        result = adapter.score_candidates(
            [10.0, 20.0, 30.0],
            [
                {"candidate_id": "far", "smiles": "CCC"},
                {"candidate_id": "close", "smiles": "CCC"},
            ],
            formula="C3H8",
        )

    assert result["evidence_kind"] == DP5Q_UNASSIGNED_SHADOW_SEMANTICS
    assert result["official_assignment_semantics"] is False
    assert result["official_equation_parity"] is True
    assert result["official_workflow_parity"] is False
    assert result["conformer_protocol"] == "chemapp.dp5q-conformer.v2"
    assert result["used_for_ranking"] is False
    assert result["calibrated_probability"] is False
    assert [item["candidate_id"] for item in result["candidates"]] == [
        "close",
        "far",
    ]
    close = result["candidates"][0]
    assert close["assignment_mode"] == "unassigned_hungarian_q50_atom_level"
    assert close["official_equation_parity"] is True
    assert close["official_workflow_parity"] is False
    assert close["used_for_ranking"] is False
    assert close["dp5q_shadow_score"] > result["candidates"][1][
        "dp5q_shadow_score"
    ]
    assert close["equation_details"]["official_equation_parity"] is True
    assert close["equation_details"]["official_workflow_parity"] is False
    assert close["equation_details"]["official_assignment_semantics"] is False
    assert (
        close["equation_details"]["conformer_protocol"]
        == "chemapp.dp5q-conformer.v2"
    )
    assert close["q50_assignment_mae_ppm"] == pytest.approx(0.0)


def test_empty_observations_do_not_start_quantile_sidecar(tmp_path: Path) -> None:
    adapter = NMRQuantileForwardAdapter(
        NMRQuantileForwardConfig(
            python_executable="does-not-exist",
            repository=tmp_path / "does-not-exist",
            sidecar_script=tmp_path / "does-not-exist.py",
        )
    )

    result = adapter.score_candidates(
        [],
        [{"candidate_id": "candidate", "smiles": "CC"}],
    )

    assert result["status"] == "unsupported_modality"
    assert result["model_called"] is False
    assert result["used_for_ranking"] is False
    assert result["quantile_enabled"] is True
    assert result["calibrated_probability"] is False
    assert adapter.handshake is None


@pytest.mark.parametrize(
    "mode",
    [
        "crossing_quantiles",
        "bad_populations",
        "bad_semantics",
        "bad_raw_digest",
        "malformed_handshake",
        "bad_distribution_contract",
    ],
)
def test_malformed_or_semantically_different_output_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    with _adapter(tmp_path, monkeypatch, mode=mode) as adapter:
        with pytest.raises(NMRForwardProtocolError):
            adapter.predict_candidates(
                [{"candidate_id": "candidate", "smiles": "CC"}]
            )
        assert adapter.handshake is None


def test_formula_mismatch_is_rejected_before_quantile_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(tmp_path, monkeypatch)

    with pytest.raises(NMRForwardInputError, match="formula mismatch"):
        adapter.predict_candidates(
            [{"candidate_id": "ethanol", "smiles": "CCO"}],
            formula="C3H8O",
        )

    assert adapter.handshake is None


def test_conformer_preflight_rejection_is_input_error_not_sidecar_outage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _adapter(
        tmp_path,
        monkeypatch,
        mode="preflight_rejection",
    ) as adapter:
        with pytest.raises(
            NMRForwardInputError,
            match="mmff_parameters_unavailable",
        ):
            adapter.predict_candidates(
                [{"candidate_id": "candidate", "smiles": "CC"}]
            )
        assert adapter.handshake is not None


def test_quantile_response_must_preserve_canonical_structure_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _adapter(
        tmp_path,
        monkeypatch,
        mode="wrong_canonical",
    ) as adapter:
        with pytest.raises(NMRForwardProtocolError, match="identity"):
            adapter.predict_candidates(
                [{"candidate_id": "candidate", "smiles": "CC"}]
            )
        assert adapter.handshake is None


def test_quantile_sidecar_environment_does_not_inherit_api_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-sidecar")
    with _adapter(
        tmp_path,
        monkeypatch,
        mode="environment_probe",
    ) as adapter:
        result = adapter.predict_candidates(
            [{"candidate_id": "candidate", "smiles": "CC"}]
        )

    assert result["predictions"][0]["warnings"] == ["secret:missing"]


def test_quantile_levels_are_exactly_one_through_ninety_nine_percent() -> None:
    assert len(DP5Q_QUANTILE_LEVELS) == 99
    assert DP5Q_QUANTILE_LEVELS[0] == 0.01
    assert DP5Q_QUANTILE_LEVELS[49] == 0.5
    assert DP5Q_QUANTILE_LEVELS[-1] == 0.99
    assert DP5Q_QUANTILE_LEVELS == tuple(
        sorted(set(DP5Q_QUANTILE_LEVELS))
    )


def test_checked_in_golden_reproduces_assigned_equation_parity_score() -> None:
    golden_path = (
        Path(__file__).parent
        / "fixtures"
        / "dp5q_quantile_golden_v1.json"
    )
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    candidate = golden["candidate"]
    prediction = {
        "candidate_id": candidate["candidate_id"],
        "canonical_smiles": candidate["canonical_smiles"],
        "conformer_count": candidate["conformer_count"],
        "conformer_populations": candidate["conformer_populations"],
        "atom_predictions": [
            {
                "atom_index": atom["atom_index"],
                "quantiles_ppm": [atom["q50_ppm"]] * 99,
                "conformer_mu_ppm": atom["conformer_mu_ppm"],
                "conformer_sigma_ppm": atom["conformer_sigma_ppm"],
                "conformer_quantile_crossing_counts": atom[
                    "conformer_quantile_crossing_counts"
                ],
                "conformer_max_quantile_crossing_ppm": atom[
                    "conformer_max_quantile_crossing_ppm"
                ],
            }
            for atom in golden["atoms"]
        ],
        "warnings": [],
    }
    assigned = {
        int(index): shift
        for index, shift in golden["assigned_case"][
            "assigned_shifts_ppm"
        ].items()
    }

    score = score_assigned_prediction(prediction, assigned)

    assert score["dp5q_method_score"] == pytest.approx(
        golden["assigned_case"]["expected_dp5q_method_score"],
        abs=golden["tolerances"]["score_absolute"],
    )
    assert {
        str(atom["atom_index"]): atom["dp5q_atom_score"]
        for atom in score["atom_scores"]
    } == pytest.approx(golden["assigned_case"]["expected_atom_scores"])


# NOTE(public-trim): removed test_upstream_manifest_and_adapter_pin_the_same_quantile_archive — depends on docs/ research manifests not shipped in this repository.

def test_fake_sidecar_source_is_finite_json_contract() -> None:
    source = _fake_sidecar_source("ok")
    assert DP5Q_QUANTILE_ARCHIVE_SHA256 in source
    assert json.dumps(float("nan")) not in source
