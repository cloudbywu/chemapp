from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Event, Thread

import pytest

from app.ml import nmr_phase6_independent_test as phase6
from app.ml.nmr_phase6_independent_test import (
    CONSUMED_SET_SCHEMA,
    EXTERNAL_HOLDER_REQUIRED,
    GENERATOR_SCHEMA,
    GOLD_SCHEMA,
    HOLDER_AUDIT_SCHEMA,
    RANKING_SCHEMA,
    ROLELESS_SCHEMA,
    RUN_SPEC_SCHEMA,
    NMRPhase6Error,
    canonical_json_bytes,
    evaluate_phase6_once,
    execution_status,
    reserve_phase6_once,
    sha256_bytes,
)


def _digest(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _write(path: Path, value: object) -> Path:
    path.write_bytes(canonical_json_bytes(value))
    return path


def _key_pair(tmp_path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    public_path = tmp_path / "holder-public.pem"
    spki = private.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private, public_path, sha256_bytes(spki)


def _artifacts(
    tmp_path: Path,
    *,
    group_count: int = 200,
    cohort_tag: str = "private",
    run_id: str = "phase6-synthetic-run",
) -> dict[str, Path | object]:
    private, public_key, public_key_digest = _key_pair(tmp_path)
    records = [
        {
            "query_id": f"query-{index:04d}",
            "connected_group_id": f"group-{index:04d}",
            "query_commitment_sha256": _digest(f"{cohort_tag}-query-{index}"),
            "connected_group_commitment_sha256": _digest(f"{cohort_tag}-group-{index}"),
            "formula": "C2H6O",
            "artifacts": [
                {
                    "kind": "carbon-13-spectrum",
                    "sha256": _digest(f"{cohort_tag}-roleless-spectrum-{index}"),
                    "bytes": 128,
                }
            ],
        }
        for index in range(group_count)
    ]
    commitment = phase6._cohort_commitment(records)
    roleless = {
        "schema_version": ROLELESS_SCHEMA,
        "protocol_version": 1,
        "protocol_id": "phase6-synthetic-protocol",
        "run_id": run_id,
        "cohort_commitment_sha256": commitment,
        "records": records,
    }
    roleless_path = _write(tmp_path / "roleless.json", roleless)
    consumed = {
        "schema_version": CONSUMED_SET_SCHEMA,
        "updated_at": "2026-08-01T00:00:00+00:00",
        "cohort_commitment_sha256s": [],
        "query_commitment_sha256s": [],
        "connected_group_commitment_sha256s": [],
    }
    consumed_path = _write(tmp_path / "consumed.json", consumed)
    spec = {
        "schema_version": RUN_SPEC_SCHEMA,
        "protocol_version": 1,
        "protocol_id": roleless["protocol_id"],
        "run_id": roleless["run_id"],
        "registered_at": "2026-08-01T00:01:00+00:00",
        "external_holder": {
            "status": "external_holder_verified",
            "holder_id": "independent-holder-1",
            "attestation": phase6.HOLDER_ATTESTATION,
            "attestation_sha256": _digest("signed-holder-attestation"),
        },
        "minimum_connected_groups": 200,
        "recall_ks": list(phase6.RECALL_KS),
        "primary_endpoint": phase6.PRIMARY_ENDPOINT,
        "secondary_endpoint": phase6.SECONDARY_ENDPOINT,
        "connected_group_aggregation": phase6.GROUP_AGGREGATION,
        "gold_absent_policy": phase6.GOLD_ABSENT_POLICY,
        "one_time_policy": phase6.ONE_TIME_POLICY,
        "v5_model_release": {
            "release_id": "frozen-v5-model",
            "manifest_sha256": _digest("v5-model-release-manifest"),
        },
        "v5_calibrator_release": {
            "release_id": "frozen-v5-calibrator",
            "manifest_sha256": _digest("v5-calibrator-release-manifest"),
        },
        "holder_public_key_spki_sha256": public_key_digest,
        "roleless_queries_sha256": sha256_bytes(roleless_path.read_bytes()),
        "cohort_commitment_sha256": commitment,
        "consumed_set_exclusion": {
            "registry_sha256": sha256_bytes(consumed_path.read_bytes()),
            "checked_at": "2026-08-01T00:00:30+00:00",
            "attestation": phase6.CONSUMED_ATTESTATION,
            "current_cohort_absent": True,
        },
    }
    spec_path = _write(tmp_path / "run-spec.json", spec)
    generator = {
        "schema_version": GENERATOR_SCHEMA,
        "protocol_version": 1,
        "protocol_id": spec["protocol_id"],
        "run_id": spec["run_id"],
        "created_at": "2026-08-01T01:00:00+00:00",
        "run_spec_sha256": sha256_bytes(spec_path.read_bytes()),
        "roleless_queries_sha256": sha256_bytes(roleless_path.read_bytes()),
        "generator_release_manifest_sha256": _digest("generator-release-manifest"),
        "records": [
            {"query_id": record["query_id"], "candidates": []} for record in records
        ],
    }
    generator_path = _write(tmp_path / "generator.json", generator)
    ranking = {
        "schema_version": RANKING_SCHEMA,
        "protocol_version": 1,
        "protocol_id": spec["protocol_id"],
        "run_id": spec["run_id"],
        "created_at": "2026-08-01T02:00:00+00:00",
        "run_spec_sha256": sha256_bytes(spec_path.read_bytes()),
        "roleless_queries_sha256": sha256_bytes(roleless_path.read_bytes()),
        "generator_output_sha256": sha256_bytes(generator_path.read_bytes()),
        "v5_model_release_manifest_sha256": spec["v5_model_release"]["manifest_sha256"],
        "v5_calibrator_release_manifest_sha256": spec["v5_calibrator_release"][
            "manifest_sha256"
        ],
        "records": [
            {"query_id": record["query_id"], "ranked_candidate_ids": []}
            for record in records
        ],
    }
    ranking_path = _write(tmp_path / "ranking.json", ranking)
    holder_audit = {
        "schema_version": HOLDER_AUDIT_SCHEMA,
        "protocol_version": 1,
        "protocol_id": spec["protocol_id"],
        "run_id": spec["run_id"],
        "cohort_commitment_sha256": commitment,
        "completed_at": "2026-08-01T00:00:45+00:00",
        "dual_review": {
            "status": (
                "two_independent_reviews_with_third_adjudicator_on_disagreement"
            ),
            "evidence_sha256": _digest("dual-review-evidence"),
        },
        "rights_review": {
            "status": "rights_cleared_for_test_and_release",
            "evidence_sha256": _digest("rights-evidence"),
        },
        "overlap_review": {
            "status": "no_consumed_cohort_query_or_connected_group_overlap",
            "evidence_sha256": _digest("overlap-evidence"),
        },
    }
    holder_audit_path = _write(tmp_path / "holder-audit.json", holder_audit)
    gold = {
        "schema_version": GOLD_SCHEMA,
        "protocol_version": 1,
        "protocol_id": spec["protocol_id"],
        "run_id": spec["run_id"],
        "cohort_commitment_sha256": commitment,
        "released_at": "2026-08-01T04:00:00+00:00",
        "holder_audit_sha256": sha256_bytes(holder_audit_path.read_bytes()),
        "records": [
            {
                "query_id": record["query_id"],
                "connected_group_id": record["connected_group_id"],
                "query_commitment_sha256": record["query_commitment_sha256"],
                "connected_group_commitment_sha256": record[
                    "connected_group_commitment_sha256"
                ],
                "exact_structure_sha256": _digest(f"{cohort_tag}-gold-exact-{index}"),
                "connectivity_sha256": _digest(
                    f"{cohort_tag}-gold-connectivity-{index}"
                ),
            }
            for index, record in enumerate(records)
        ],
    }
    gold_path = _write(tmp_path / "holder-gold.json", gold)
    signature_path = tmp_path / "holder-gold.ed25519"
    signature_path.write_bytes(private.sign(gold_path.read_bytes()))
    return {
        "spec": spec_path,
        "roleless": roleless_path,
        "generator": generator_path,
        "ranking": ranking_path,
        "consumed": consumed_path,
        "ledger": tmp_path / "reservation.jsonl",
        "gold": gold_path,
        "audit": holder_audit_path,
        "signature": signature_path,
        "public_key": public_key,
        "private_key": private,
    }


def _rewrite_public_bindings(paths: dict[str, Path | object]) -> None:
    roleless_path = Path(paths["roleless"])
    roleless = json.loads(roleless_path.read_text(encoding="utf-8"))
    roleless["cohort_commitment_sha256"] = phase6._cohort_commitment(
        roleless["records"]
    )
    _write(roleless_path, roleless)

    spec_path = Path(paths["spec"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["cohort_commitment_sha256"] = roleless["cohort_commitment_sha256"]
    spec["roleless_queries_sha256"] = sha256_bytes(roleless_path.read_bytes())
    _write(spec_path, spec)

    generator_path = Path(paths["generator"])
    generator = json.loads(generator_path.read_text(encoding="utf-8"))
    generator["run_spec_sha256"] = sha256_bytes(spec_path.read_bytes())
    generator["roleless_queries_sha256"] = sha256_bytes(roleless_path.read_bytes())
    _write(generator_path, generator)

    ranking_path = Path(paths["ranking"])
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranking["run_spec_sha256"] = sha256_bytes(spec_path.read_bytes())
    ranking["roleless_queries_sha256"] = sha256_bytes(roleless_path.read_bytes())
    ranking["generator_output_sha256"] = sha256_bytes(generator_path.read_bytes())
    _write(ranking_path, ranking)


def _reserve(paths: dict[str, Path | object]):
    return reserve_phase6_once(
        run_spec_path=paths["spec"],
        roleless_path=paths["roleless"],
        generator_output_path=paths["generator"],
        ranking_output_path=paths["ranking"],
        consumed_set_path=paths["consumed"],
        reservation_ledger_path=paths["ledger"],
        clock=lambda: datetime(2026, 8, 1, 3, tzinfo=timezone.utc),
    )


def test_status_is_fail_closed_without_external_holder() -> None:
    status = execution_status()
    assert status["status"] == EXTERNAL_HOLDER_REQUIRED
    assert status["executed"] is False
    assert status["real_external_results"] is None


def test_empty_candidate_lists_are_full_denominator_generator_misses(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    reservation = _reserve(paths)
    assert reservation["connected_group_count"] == 200

    result = evaluate_phase6_once(
        run_spec_path=paths["spec"],
        roleless_path=paths["roleless"],
        generator_output_path=paths["generator"],
        ranking_output_path=paths["ranking"],
        reservation_ledger_path=paths["ledger"],
        holder_gold_path=paths["gold"],
        holder_audit_path=paths["audit"],
        holder_gold_signature_path=paths["signature"],
        holder_public_key_path=paths["public_key"],
        expected_reservation_entry_sha256=reservation["entry_sha256"],
        expected_holder_public_key_spki_sha256=(
            json.loads(Path(paths["spec"]).read_text(encoding="utf-8"))[
                "holder_public_key_spki_sha256"
            ]
        ),
    )

    assert result["denominators"] == {
        "queries": 200,
        "connected_groups": 200,
        "conditional_exact_queries": 0,
        "conditional_connectivity_queries": 0,
    }
    assert result["generator_misses"]["exact_queries"] == 200
    assert result["primary"]["full_denominator"] is True
    assert set(result["primary"]["exact_recall_at_k"].values()) == {0.0}
    assert result["secondary"]["not_a_primary_claim"] is True


def test_reservation_rejects_cohort_reuse(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    _reserve(paths)
    with pytest.raises(NMRPhase6Error, match="already reserved"):
        _reserve(paths)


def test_primary_uses_all_groups_while_conditional_ranker_is_secondary(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    gold = json.loads(Path(paths["gold"]).read_text(encoding="utf-8"))
    generator_path = Path(paths["generator"])
    generator = json.loads(generator_path.read_text(encoding="utf-8"))
    generator["records"][0]["candidates"] = [
        {
            "candidate_id": "candidate-hit",
            "exact_structure_sha256": gold["records"][0]["exact_structure_sha256"],
            "connectivity_sha256": gold["records"][0]["connectivity_sha256"],
        }
    ]
    _write(generator_path, generator)
    ranking_path = Path(paths["ranking"])
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranking["generator_output_sha256"] = sha256_bytes(generator_path.read_bytes())
    ranking["records"][0]["ranked_candidate_ids"] = ["candidate-hit"]
    _write(ranking_path, ranking)
    reservation = _reserve(paths)

    spec = json.loads(Path(paths["spec"]).read_text(encoding="utf-8"))
    result = evaluate_phase6_once(
        run_spec_path=paths["spec"],
        roleless_path=paths["roleless"],
        generator_output_path=paths["generator"],
        ranking_output_path=paths["ranking"],
        reservation_ledger_path=paths["ledger"],
        holder_gold_path=paths["gold"],
        holder_audit_path=paths["audit"],
        holder_gold_signature_path=paths["signature"],
        holder_public_key_path=paths["public_key"],
        expected_reservation_entry_sha256=reservation["entry_sha256"],
        expected_holder_public_key_spki_sha256=spec["holder_public_key_spki_sha256"],
    )

    assert result["primary"]["exact_recall_at_k"]["1"] == pytest.approx(1 / 200)
    assert result["secondary"]["exact_recall_at_k"]["1"] == 1.0
    assert result["denominators"]["conditional_exact_queries"] == 1
    assert result["generator_misses"]["exact_queries"] == 199


def test_evaluation_requires_independently_supplied_key_digest(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    reservation = _reserve(paths)
    with pytest.raises(NMRPhase6Error, match="trust anchor"):
        evaluate_phase6_once(
            run_spec_path=paths["spec"],
            roleless_path=paths["roleless"],
            generator_output_path=paths["generator"],
            ranking_output_path=paths["ranking"],
            reservation_ledger_path=paths["ledger"],
            holder_gold_path=paths["gold"],
            holder_audit_path=paths["audit"],
            holder_gold_signature_path=paths["signature"],
            holder_public_key_path=paths["public_key"],
            expected_reservation_entry_sha256=reservation["entry_sha256"],
            expected_holder_public_key_spki_sha256="f" * 64,
        )


def test_evaluation_does_not_touch_gold_before_reservation(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    Path(paths["gold"]).unlink()
    with pytest.raises(NMRPhase6Error, match="reservation"):
        evaluate_phase6_once(
            run_spec_path=paths["spec"],
            roleless_path=paths["roleless"],
            generator_output_path=paths["generator"],
            ranking_output_path=paths["ranking"],
            reservation_ledger_path=paths["ledger"],
            holder_gold_path=paths["gold"],
            holder_audit_path=paths["audit"],
            holder_gold_signature_path=paths["signature"],
            holder_public_key_path=paths["public_key"],
            expected_reservation_entry_sha256="f" * 64,
            expected_holder_public_key_spki_sha256=(
                json.loads(Path(paths["spec"]).read_text(encoding="utf-8"))[
                    "holder_public_key_spki_sha256"
                ]
            ),
        )


def test_reservation_rejects_less_than_200_groups(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path, group_count=199)
    with pytest.raises(NMRPhase6Error, match="fewer than 200"):
        _reserve(paths)


def test_generator_requires_every_roleless_query_in_original_order(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    generator_path = Path(paths["generator"])
    generator = json.loads(generator_path.read_text(encoding="utf-8"))
    generator["records"].pop()
    _write(generator_path, generator)
    with pytest.raises(NMRPhase6Error, match="exactly cover"):
        _reserve(paths)


@pytest.mark.parametrize("overlap_level", ["query", "group"])
def test_reservation_rejects_partial_overlap_with_prior_reservation(
    tmp_path: Path, overlap_level: str
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _artifacts(first_dir, cohort_tag="first", run_id="phase6-run-first")
    second = _artifacts(second_dir, cohort_tag="second", run_id="phase6-run-second")
    shared_ledger = tmp_path / "authoritative-reservations.jsonl"
    first["ledger"] = shared_ledger
    second["ledger"] = shared_ledger
    _reserve(first)

    first_roleless = json.loads(Path(first["roleless"]).read_text(encoding="utf-8"))
    second_roleless_path = Path(second["roleless"])
    second_roleless = json.loads(second_roleless_path.read_text(encoding="utf-8"))
    source_field = (
        "query_commitment_sha256"
        if overlap_level == "query"
        else "connected_group_commitment_sha256"
    )
    second_roleless["records"][0][source_field] = first_roleless["records"][0][
        source_field
    ]
    _write(second_roleless_path, second_roleless)
    _rewrite_public_bindings(second)

    expected_message = (
        "queries overlap" if overlap_level == "query" else "groups overlap"
    )
    with pytest.raises(NMRPhase6Error, match=expected_message):
        _reserve(second)


def test_roleless_rejects_group_commitment_aliases(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path, group_count=201)
    roleless_path = Path(paths["roleless"])
    roleless = json.loads(roleless_path.read_text(encoding="utf-8"))
    roleless["records"][1]["connected_group_commitment_sha256"] = roleless["records"][
        0
    ]["connected_group_commitment_sha256"]
    roleless["cohort_commitment_sha256"] = phase6._cohort_commitment(
        roleless["records"]
    )
    _write(roleless_path, roleless)

    with pytest.raises(NMRPhase6Error, match="commitment maps to multiple group ids"):
        _reserve(paths)


def test_evaluation_is_recorded_once_and_replay_is_rejected_before_gold(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    reservation = _reserve(paths)
    spec = json.loads(Path(paths["spec"]).read_text(encoding="utf-8"))
    arguments = {
        "run_spec_path": paths["spec"],
        "roleless_path": paths["roleless"],
        "generator_output_path": paths["generator"],
        "ranking_output_path": paths["ranking"],
        "reservation_ledger_path": paths["ledger"],
        "holder_gold_path": paths["gold"],
        "holder_audit_path": paths["audit"],
        "holder_gold_signature_path": paths["signature"],
        "holder_public_key_path": paths["public_key"],
        "expected_reservation_entry_sha256": reservation["entry_sha256"],
        "expected_holder_public_key_spki_sha256": spec["holder_public_key_spki_sha256"],
    }
    result = evaluate_phase6_once(**arguments)
    assert result["status"] == "evaluated_once"
    entries = phase6._read_ledger(Path(paths["ledger"]))
    assert [entry["event"] for entry in entries] == [
        "one_time_reservation",
        "one_time_evaluation",
    ]
    assert entries[1]["metric_result_sha256"] == sha256_bytes(
        canonical_json_bytes(result)
    )

    Path(paths["gold"]).unlink()
    with pytest.raises(NMRPhase6Error, match="already been evaluated once"):
        evaluate_phase6_once(**arguments)


def test_concurrent_evaluation_is_rejected_before_its_gold_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _artifacts(tmp_path)
    reservation = _reserve(paths)
    spec = json.loads(Path(paths["spec"]).read_text(encoding="utf-8"))
    arguments = {
        "run_spec_path": paths["spec"],
        "roleless_path": paths["roleless"],
        "generator_output_path": paths["generator"],
        "ranking_output_path": paths["ranking"],
        "reservation_ledger_path": paths["ledger"],
        "holder_gold_path": paths["gold"],
        "holder_audit_path": paths["audit"],
        "holder_gold_signature_path": paths["signature"],
        "holder_public_key_path": paths["public_key"],
        "expected_reservation_entry_sha256": reservation["entry_sha256"],
        "expected_holder_public_key_spki_sha256": spec["holder_public_key_spki_sha256"],
    }
    holder_boundary_entered = Event()
    release_first_evaluation = Event()
    original_key_loader = phase6._public_key_spki_sha256

    def paused_key_loader(path: str | Path):
        holder_boundary_entered.set()
        if not release_first_evaluation.wait(timeout=5):
            raise AssertionError("test did not release first evaluation")
        return original_key_loader(path)

    monkeypatch.setattr(phase6, "_public_key_spki_sha256", paused_key_loader)
    outcome: list[object] = []

    def run_first() -> None:
        try:
            outcome.append(evaluate_phase6_once(**arguments))
        except BaseException as exc:  # pragma: no cover - reported by assertion below
            outcome.append(exc)

    first = Thread(target=run_first)
    first.start()
    assert holder_boundary_entered.wait(timeout=5)
    second_arguments = dict(arguments)
    second_arguments["holder_gold_path"] = tmp_path / "must-not-be-opened.json"
    try:
        with pytest.raises(NMRPhase6Error, match="locked by another process"):
            evaluate_phase6_once(**second_arguments)
    finally:
        release_first_evaluation.set()
        first.join(timeout=5)
    assert not first.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], dict)
    assert outcome[0]["status"] == "evaluated_once"


def test_evaluation_requires_independently_anchored_reservation_before_gold(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    _reserve(paths)
    spec = json.loads(Path(paths["spec"]).read_text(encoding="utf-8"))
    Path(paths["gold"]).unlink()
    with pytest.raises(NMRPhase6Error, match="independently anchored digest"):
        evaluate_phase6_once(
            run_spec_path=paths["spec"],
            roleless_path=paths["roleless"],
            generator_output_path=paths["generator"],
            ranking_output_path=paths["ranking"],
            reservation_ledger_path=paths["ledger"],
            holder_gold_path=paths["gold"],
            holder_audit_path=paths["audit"],
            holder_gold_signature_path=paths["signature"],
            holder_public_key_path=paths["public_key"],
            expected_reservation_entry_sha256="f" * 64,
            expected_holder_public_key_spki_sha256=spec[
                "holder_public_key_spki_sha256"
            ],
        )


@pytest.mark.parametrize(
    "payload, message",
    [
        (b'{"value":NaN}\n', "non-finite"),
        (b'{"value":1,"value":2}\n', "duplicate JSON key"),
    ],
)
def test_strict_json_rejects_nan_and_duplicate_keys(
    payload: bytes, message: str
) -> None:
    with pytest.raises(NMRPhase6Error, match=message):
        phase6.strict_json_bytes(payload, location="adversarial")


def test_ranking_rejects_non_string_ids_without_typeerror(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    ranking_path = Path(paths["ranking"])
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranking["records"][0]["ranked_candidate_ids"] = [{}]
    _write(ranking_path, ranking)
    with pytest.raises(NMRPhase6Error, match="invalid non-empty text"):
        _reserve(paths)


def test_roleless_identifier_cannot_escape_as_a_path(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    roleless_path = Path(paths["roleless"])
    roleless = json.loads(roleless_path.read_text(encoding="utf-8"))
    roleless["records"][0]["query_id"] = "../holder-gold"
    _write(roleless_path, roleless)
    with pytest.raises(NMRPhase6Error, match="invalid identifier"):
        _reserve(paths)
