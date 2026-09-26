"""Tests for the executable v8 sealed-test state machine."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.ml.sealed_test.cli import run_simulation, run_verify
from app.ml.sealed_test.crypto import (
    canonical_object_bytes,
    generate_private_key_pem,
    load_private_key,
    public_key_from_private_pem,
    sign_object,
    verify_object,
)
from app.ml.sealed_test.ledger import Denylist, Ledger, LedgerError
from app.ml.sealed_test.protocol import (
    SealedTestError,
    SealedTestStateMachine,
    build_reservation_draft,
    sign_reservation,
)


def _load(dir_path: Path, name: str) -> dict:
    return json.loads((dir_path / name).read_text(encoding="utf-8"))


def _holder_pub(dir_path: Path):
    return public_key_from_private_pem((dir_path / "holder.key.pem").read_bytes())


def _executor_pub(dir_path: Path):
    return public_key_from_private_pem((dir_path / "executor.key.pem").read_bytes())


def _evaluator_pub(dir_path: Path):
    return public_key_from_private_pem((dir_path / "evaluator.key.pem").read_bytes())


@pytest.fixture()
def simulated(tmp_path: Path) -> Path:
    summary = run_simulation(tmp_path)
    assert summary["chain_status"] == "verified"
    return tmp_path


def test_full_simulation_and_reverify(simulated: Path) -> None:
    result = run_verify(simulated)
    assert result["status"] == "verified"
    assert result["problems"] == []


def test_attempt_consumption_is_exactly_once(simulated: Path) -> None:
    reservation = _load(simulated, "reservation.json")
    attempt = _load(simulated, "attempt.json")
    ledger = Ledger(simulated / "ledger.jsonl")
    machine = SealedTestStateMachine(simulation=True)
    with pytest.raises(LedgerError, match="already consumed"):
        machine.consume(reservation, attempt, _holder_pub(simulated), ledger)


def test_denylist_blocks_reuse(simulated: Path) -> None:
    denylist = Denylist(simulated / "denylist.jsonl")
    with pytest.raises(LedgerError, match="denylist overlap"):
        denylist.check([f"src-q_{i:02d}" for i in range(10)])


def test_commit_before_consume_is_rejected(simulated: Path) -> None:
    reservation = _load(simulated, "reservation.json")
    attempt = _load(simulated, "attempt.json")
    commitment = _load(simulated, "commitment.json")
    machine = SealedTestStateMachine(simulation=True)
    with pytest.raises(SealedTestError, match="ledger attempt binding mismatch"):
        machine.commit(
            reservation,
            attempt,
            {},
            commitment,
            _holder_pub(simulated),
            _executor_pub(simulated),
        )


def test_missing_query_in_prediction_manifest_is_rejected(simulated: Path) -> None:
    reservation = _load(simulated, "reservation.json")
    attempt = _load(simulated, "attempt.json")
    commitment = _load(simulated, "commitment.json")
    ledger_entry = Ledger(simulated / "ledger.jsonl").read()[0]
    machine = SealedTestStateMachine(simulation=True)
    query_ids = [f"q_{i:02d}" for i in range(10)]
    manifest = {
        "schema_version": "chemapp.nmr.v8-prediction-manifest.v1",
        "attempt_id": attempt["attempt_id"],
        "outcomes": {query_id: "success" for query_id in query_ids[:-1]},
    }
    with pytest.raises(SealedTestError, match="must cover every query exactly once"):
        machine.commit(
            reservation,
            attempt,
            ledger_entry,
            commitment,
            _holder_pub(simulated),
            _executor_pub(simulated),
            query_ids=query_ids,
            prediction_manifest=manifest,
        )


def test_empty_prediction_denominator_is_rejected(simulated: Path) -> None:
    reservation = _load(simulated, "reservation.json")
    attempt = _load(simulated, "attempt.json")
    commitment = copy.deepcopy(_load(simulated, "commitment.json"))
    ledger_entry = Ledger(simulated / "ledger.jsonl").read()[0]
    commitment["query_count"] = 0
    commitment["prediction_count"] = 0
    machine = SealedTestStateMachine(simulation=True)
    with pytest.raises(SealedTestError, match="query_count invalid"):
        machine.commit(
            reservation, attempt, ledger_entry, commitment,
            _holder_pub(simulated), _executor_pub(simulated),
        )


def test_signature_tamper_is_rejected(simulated: Path) -> None:
    reservation = _load(simulated, "reservation.json")
    attempt = _load(simulated, "attempt.json")
    commitment = copy.deepcopy(_load(simulated, "commitment.json"))
    ledger_entry = Ledger(simulated / "ledger.jsonl").read()[0]
    commitment["executor"] = {"executor_id": "tampered"}
    machine = SealedTestStateMachine(simulation=True)
    with pytest.raises(SealedTestError, match="signed payload hash mismatch"):
        machine.commit(
            reservation, attempt, ledger_entry, commitment,
            _holder_pub(simulated), _executor_pub(simulated),
        )


def test_gold_before_commitment_is_rejected(simulated: Path) -> None:
    reservation = _load(simulated, "reservation.json")
    attempt = _load(simulated, "attempt.json")
    commitment = _load(simulated, "commitment.json")
    receipt = copy.deepcopy(_load(simulated, "receipt.json"))
    receipt["gold_release_receipt"]["gold_released_at_utc"] = "2000-01-01T00:00:00Z"
    machine = SealedTestStateMachine(simulation=True)
    with pytest.raises(SealedTestError, match="Gold must follow"):
        machine.evaluate(
            reservation, attempt, commitment, receipt,
            _holder_pub(simulated), _evaluator_pub(simulated),
        )


def test_real_reservation_thresholds_are_enforced(tmp_path: Path) -> None:
    key = public_key_from_private_pem(generate_private_key_pem())
    draft = build_reservation_draft(
        holder_id="holder", holder_public_key=key, record_count=10,
        connected_group_count=5, protocol_sha256="0" * 64,
        task_manifest_sha256="0" * 64, model_release_sha256="0" * 64,
        generator_release_sha256="0" * 64, cohort_commitment_sha256="0" * 64,
        review_commitment_sha256="0" * 64, license_review_commitment_sha256="0" * 64,
        similarity_audit_commitment_sha256="0" * 64, query_bundle_sha256="0" * 64,
        prior_influence_index_sha256="0" * 64, denylist_commitment_sha256="0" * 64,
        created_at_utc="2026-08-13T00:00:00Z",
    )
    private = load_private_key(_write_key(tmp_path, "holder", generate_private_key_pem()))
    reservation = sign_reservation(draft, private, anchor_timestamp_utc="2026-08-13T00:00:01Z")
    machine = SealedTestStateMachine(simulation=False)
    with pytest.raises(SealedTestError, match="record_count below threshold"):
        machine.verify_reservation(reservation, key)


def test_simulation_reservation_requires_simulation_flag(tmp_path: Path) -> None:
    pem = generate_private_key_pem()
    key = public_key_from_private_pem(pem)
    draft = build_reservation_draft(
        holder_id="holder", holder_public_key=key, record_count=10,
        connected_group_count=5, protocol_sha256="0" * 64,
        task_manifest_sha256="0" * 64, model_release_sha256="0" * 64,
        generator_release_sha256="0" * 64, cohort_commitment_sha256="0" * 64,
        review_commitment_sha256="0" * 64, license_review_commitment_sha256="0" * 64,
        similarity_audit_commitment_sha256="0" * 64, query_bundle_sha256="0" * 64,
        prior_influence_index_sha256="0" * 64, denylist_commitment_sha256="0" * 64,
        created_at_utc="2026-08-13T00:00:00Z", simulation=True,
    )
    draft.pop("simulation")
    private = load_private_key(_write_key(tmp_path, "holder", pem))
    reservation = sign_reservation(draft, private, anchor_timestamp_utc="2026-08-13T00:00:01Z")
    machine = SealedTestStateMachine(simulation=True)
    with pytest.raises(SealedTestError, match="simulation mode requires simulation=true"):
        machine.verify_reservation(reservation, key)


def test_canonical_json_is_deterministic() -> None:
    left = {"b": 1, "a": [2, 3], "c": {"x": True}}
    right = {"c": {"x": True}, "a": [2, 3], "b": 1}
    assert canonical_object_bytes(left) == canonical_object_bytes(right)


def test_sign_verify_roundtrip(tmp_path: Path) -> None:
    pem = generate_private_key_pem()
    private = load_private_key(_write_key(tmp_path, "roundtrip", pem))
    key = public_key_from_private_pem(pem)
    payload = {"attempt_id": "att_" + "a" * 32}
    signature = sign_object(payload, private)
    assert verify_object(payload, signature, key) is True


def test_verify_chain_rejects_missing_ledger(simulated: Path, tmp_path: Path) -> None:
    reservation = _load(simulated, "reservation.json")
    attempt = _load(simulated, "attempt.json")
    commitment = _load(simulated, "commitment.json")
    receipt = _load(simulated, "receipt.json")
    empty_ledger = Ledger(tmp_path / "empty-ledger.jsonl")
    machine = SealedTestStateMachine(simulation=True)
    result = machine.verify_chain(
        reservation, attempt, commitment, receipt,
        _holder_pub(simulated), _executor_pub(simulated), _evaluator_pub(simulated),
        ledger=empty_ledger,
    )
    assert result["status"] == "rejected"
    assert any("exactly one consumed entry" in problem for problem in result["problems"])


def _write_key(dir_path: Path, name: str, pem: bytes) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{name}.key.pem"
    path.write_bytes(pem)
    return path
