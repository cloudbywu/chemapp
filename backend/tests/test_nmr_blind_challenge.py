from __future__ import annotations

from datetime import datetime, timezone
import builtins
import json
import os
from pathlib import Path

import pytest

from app.ml import nmr_blind_challenge as blind_module
from app.ml.nmr_blind_challenge import (
    GOLD_FILENAME,
    HOLDER_MANIFEST_FILENAME,
    PUBLIC_MANIFEST_FILENAME,
    RELEASE_GOLD_FILENAME,
    RELEASE_MANIFEST_FILENAME,
    RELEASE_SIGNATURE_FILENAME,
    REVIEW_AUDIT_FILENAME,
    ROLELESS_FILENAME,
    RUN_SPEC_FILENAME,
    NMRBlindChallengeError,
    NMRBlindChallengeDependencyError,
    build_blind_challenge,
    canonical_json_bytes,
    release_blind_gold,
    sha256_bytes,
    submit_blind_predictions,
    verify_blind_release,
)


def _digest(text: str) -> str:
    return sha256_bytes(text.encode())


def _review(
    reviewer: str,
    decision: str,
    candidate: str | None,
    hour: int,
) -> dict:
    return {
        "reviewer_id": reviewer,
        "reviewed_at": f"2026-08-01T{hour:02d}:00:00+00:00",
        "decision": decision,
        "selected_source_candidate_id": candidate,
        "evidence_sha256": _digest(f"{reviewer}-{hour}"),
        "independence_attestation": ("completed_without_access_to_the_other_review"),
    }


def _source_record(index: int, mode: str) -> dict:
    candidates = [
        {"source_candidate_id": f"src-{index}-a", "smiles": f"CC{'C' * index}"},
        {"source_candidate_id": f"src-{index}-b", "smiles": f"CO{'C' * index}"},
    ]
    if mode == "agree_include":
        reviews = [
            _review(f"reviewer-{index}-1", "include", f"src-{index}-a", 1),
            _review(f"reviewer-{index}-2", "include", f"src-{index}-a", 2),
        ]
        adjudication = None
    elif mode == "disagree_include":
        reviews = [
            _review(f"reviewer-{index}-1", "include", f"src-{index}-a", 1),
            _review(f"reviewer-{index}-2", "include", f"src-{index}-b", 2),
        ]
        adjudication = {
            "adjudicator_id": f"adjudicator-{index}",
            "adjudicated_at": "2026-08-01T03:00:00+00:00",
            "decision": "include",
            "selected_source_candidate_id": f"src-{index}-b",
            "rationale_sha256": _digest(f"adjudication-{index}"),
        }
    elif mode == "disagree_exclude":
        reviews = [
            _review(f"reviewer-{index}-1", "include", f"src-{index}-a", 1),
            _review(f"reviewer-{index}-2", "exclude", None, 2),
        ]
        adjudication = {
            "adjudicator_id": f"adjudicator-{index}",
            "adjudicated_at": "2026-08-01T03:00:00+00:00",
            "decision": "exclude",
            "selected_source_candidate_id": None,
            "rationale_sha256": _digest(f"adjudication-{index}"),
        }
    else:
        raise AssertionError(mode)
    return {
        "source": {
            "collection_id": "synthetic-holder-fixture",
            "record_id": f"source-record-{index}",
            "record_locator_sha256": _digest(f"locator-{index}"),
        },
        "spectrum": {
            "nucleus": "13C",
            "shifts_ppm": [10.0 + index, 20.0 + index],
            "spectrum_fingerprint_sha256": _digest(f"spectrum-{index}"),
        },
        "candidates": candidates,
        "reviews": reviews,
        "adjudication": adjudication,
    }


def _bundle() -> dict:
    return {
        "schema_version": "chemapp.nmr.blind-review-bundle.v1",
        "dataset_id": "synthetic-test-only",
        "records": [
            _source_record(1, "agree_include"),
            _source_record(2, "disagree_include"),
            _source_record(3, "disagree_exclude"),
        ],
    }


def _run_spec(*, holder_public_key_spki_sha256: str = "d" * 64) -> dict:
    analysis_plan = {
        "schema_version": "synthetic-analysis-plan.v1",
        "primary_test": "group-level exact top-1 accuracy",
    }
    model_manifest = {
        "schema_version": "synthetic-model-release.v1",
        "artifacts": {
            "weights.bin": {"sha256": _digest("synthetic-weights"), "bytes": 17}
        },
    }
    return {
        "schema_version": "chemapp.nmr.blind-run-spec.v2",
        "protocol_id": "synthetic-protocol-v1",
        "registered_at": "2026-08-01T00:00:00+00:00",
        "primary_endpoint": "group-level top-1 accuracy",
        "holder_public_key_spki_sha256": holder_public_key_spki_sha256,
        "model_release_ids": ["frozen-model-v5"],
        "model_release_manifests": {"frozen-model-v5": model_manifest},
        "analysis_plan": analysis_plan,
        "analysis_plan_sha256": sha256_bytes(canonical_json_bytes(analysis_plan)),
    }


class _Tokens:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, size: int) -> str:
        assert size == 16
        self.value += 1
        return f"{self.value:032x}"


def _clock(hour: int = 0):
    return lambda: datetime(2026, 8, 1, hour, tzinfo=timezone.utc)


def _write_json(path: Path, value: object, *, canonical: bool = False) -> None:
    payload = (
        canonical_json_bytes(value)
        if canonical
        else json.dumps(value, allow_nan=True).encode("utf-8")
    )
    path.write_bytes(payload)


def _build(
    tmp_path: Path,
    *,
    holder_public_key_spki_sha256: str = "d" * 64,
) -> dict[str, Path]:
    source = tmp_path / "reviewed.json"
    spec = tmp_path / "input-run-spec.json"
    _write_json(source, _bundle())
    _write_json(
        spec,
        _run_spec(holder_public_key_spki_sha256=holder_public_key_spki_sha256),
    )
    public = tmp_path / "public"
    holder = tmp_path / "holder"
    build_blind_challenge(
        source,
        spec,
        public,
        holder,
        _clock=_clock(),
        _token_hex=_Tokens(),
        _shuffle=lambda rows: rows.reverse(),
    )
    return {
        "source": source,
        "public": public,
        "holder": holder,
        "public_manifest": public / PUBLIC_MANIFEST_FILENAME,
        "roleless": public / ROLELESS_FILENAME,
        "run_spec": public / RUN_SPEC_FILENAME,
        "holder_manifest": holder / HOLDER_MANIFEST_FILENAME,
        "gold": holder / GOLD_FILENAME,
        "audit": holder / REVIEW_AUDIT_FILENAME,
    }


def _predictions(paths: dict[str, Path], *, model: str = "frozen-model-v5") -> Path:
    roleless_payload = paths["roleless"].read_bytes()
    roleless = json.loads(roleless_payload)
    spec_payload = paths["run_spec"].read_bytes()
    spec = json.loads(spec_payload)
    model_manifest = spec["model_release_manifests"].get(
        model,
        {"schema_version": "unregistered-model-release.v1"},
    )
    value = {
        "schema_version": "chemapp.nmr.blind-predictions.v1",
        "protocol_version": 1,
        "challenge_id": roleless["challenge_id"],
        "roleless_sha256": sha256_bytes(roleless_payload),
        "run_spec_sha256": sha256_bytes(spec_payload),
        "model_release_id": model,
        "model_release_manifest_sha256": sha256_bytes(
            canonical_json_bytes(model_manifest)
        ),
        "created_at": "2026-08-01T04:00:00+00:00",
        "records": [
            {
                "record_id": row["record_id"],
                "ranked_candidate_ids": [
                    candidate["candidate_id"] for candidate in row["candidates"]
                ],
            }
            for row in roleless["records"]
        ],
    }
    target = paths["public"].parent / "predictions.json"
    _write_json(target, value, canonical=True)
    return target


def _keys(tmp_path: Path) -> tuple[Path, Path]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "holder-private.pem"
    public_path = tmp_path / "holder-public.pem"
    private_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    if os.name != "nt":
        private_path.chmod(0o600)
    public_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def _public_key_spki_sha256(public_key_path: Path) -> str:
    from cryptography.hazmat.primitives import serialization

    public_key = serialization.load_pem_public_key(public_key_path.read_bytes())
    return sha256_bytes(
        public_key.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def _build_with_keys(tmp_path: Path) -> tuple[dict[str, Path], Path, Path, str]:
    private_key, public_key = _keys(tmp_path)
    expected_key_sha = _public_key_spki_sha256(public_key)
    paths = _build(
        tmp_path,
        holder_public_key_spki_sha256=expected_key_sha,
    )
    return paths, private_key, public_key, expected_key_sha


def _release_args(
    paths: dict[str, Path],
    predictions: Path,
    ledger: Path,
    private_key: Path,
    release: Path,
) -> dict:
    return {
        "holder_manifest_path": paths["holder_manifest"],
        "public_manifest_path": paths["public_manifest"],
        "roleless_path": paths["roleless"],
        "run_spec_path": paths["run_spec"],
        "holder_gold_path": paths["gold"],
        "review_audit_path": paths["audit"],
        "predictions_path": predictions,
        "receipt_ledger_path": ledger,
        "private_key_path": private_key,
        "release_directory": release,
        "_clock": _clock(6),
    }


def test_end_to_end_build_submit_release_verify_with_excluded_row(
    tmp_path: Path,
) -> None:
    paths, private_key, public_key, expected_key_sha = _build_with_keys(tmp_path)
    predictions = _predictions(paths)
    ledger = tmp_path / "receipts.jsonl"
    release = tmp_path / "release"

    submission = submit_blind_predictions(
        paths["public_manifest"],
        paths["roleless"],
        paths["run_spec"],
        predictions,
        ledger,
        _clock=_clock(5),
        _token_hex=_Tokens(),
    )
    assert submission["event"] == "predictions_submitted_before_gold_release"
    released = release_blind_gold(
        **_release_args(paths, predictions, ledger, private_key, release)
    )
    assert released["submission_receipt_sha256"] == submission["receipt_sha256"]
    verified = verify_blind_release(
        release_manifest_path=release / RELEASE_MANIFEST_FILENAME,
        signature_path=release / RELEASE_SIGNATURE_FILENAME,
        public_key_path=public_key,
        expected_public_key_spki_sha256=expected_key_sha,
        public_manifest_path=paths["public_manifest"],
        holder_manifest_path=paths["holder_manifest"],
        roleless_path=paths["roleless"],
        run_spec_path=paths["run_spec"],
        gold_path=release / RELEASE_GOLD_FILENAME,
        predictions_path=predictions,
        review_audit_path=paths["audit"],
        receipt_ledger_path=ledger,
    )
    assert verified["status"] == "verified"
    assert verified["record_count"] == 2
    assert verified["receipt_ledger_is_worm"] is False
    assert verified["holder_public_key_spki_sha256"] == expected_key_sha
    with pytest.raises(NMRBlindChallengeError, match="trust anchor"):
        verify_blind_release(
            release_manifest_path=release / RELEASE_MANIFEST_FILENAME,
            signature_path=release / RELEASE_SIGNATURE_FILENAME,
            public_key_path=public_key,
            expected_public_key_spki_sha256="0" * 64,
            public_manifest_path=paths["public_manifest"],
            holder_manifest_path=paths["holder_manifest"],
            roleless_path=paths["roleless"],
            run_spec_path=paths["run_spec"],
            gold_path=release / RELEASE_GOLD_FILENAME,
            predictions_path=predictions,
            review_audit_path=paths["audit"],
            receipt_ledger_path=ledger,
        )
    audit = json.loads(paths["audit"].read_bytes())
    excluded = next(row for row in audit["records"] if not row["included"])
    assert excluded["record_id"] is None
    assert excluded["truth_candidate_id"] is None
    assert all(item["candidate_id"] is None for item in excluded["candidate_map"])


def test_release_rejects_key_not_committed_before_submission(tmp_path: Path) -> None:
    paths, _, _, _ = _build_with_keys(tmp_path)
    predictions = _predictions(paths)
    ledger = tmp_path / "ledger.jsonl"
    submit_blind_predictions(
        paths["public_manifest"],
        paths["roleless"],
        paths["run_spec"],
        predictions,
        ledger,
        _clock=_clock(5),
    )
    other = tmp_path / "other-key"
    other.mkdir()
    wrong_private_key, _ = _keys(other)
    release = tmp_path / "release"
    with pytest.raises(NMRBlindChallengeError, match="committed before prediction"):
        release_blind_gold(
            **_release_args(paths, predictions, ledger, wrong_private_key, release)
        )
    assert not release.exists()


def test_gold_cannot_be_released_before_prediction_receipt(tmp_path: Path) -> None:
    paths, private_key, _, _ = _build_with_keys(tmp_path)
    predictions = _predictions(paths)
    release = tmp_path / "release"
    with pytest.raises(NMRBlindChallengeError, match="Gold remains sealed"):
        release_blind_gold(
            **_release_args(
                paths,
                predictions,
                tmp_path / "empty-ledger.jsonl",
                private_key,
                release,
            )
        )
    assert not release.exists()


def test_ed25519_dependency_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def blocked_import(name: str, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(NMRBlindChallengeDependencyError, match="cryptography>=44.0"):
        blind_module._require_ed25519()


def test_public_pack_has_no_truth_source_or_reviewer_leak(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    roleless = json.loads(paths["roleless"].read_bytes())
    serialized = paths["roleless"].read_text(encoding="utf-8")
    assert "source-record" not in serialized
    assert "reviewer-" not in serialized
    assert "truth" not in serialized.lower()

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert keys(roleless).isdisjoint(
        {"source", "reviews", "reviewer_id", "adjudication", "truth_candidate_id"}
    )


@pytest.mark.parametrize(
    "fault",
    [
        "same_reviewer",
        "missing_adjudication",
        "early_adjudication",
        "simultaneous_adjudication",
    ],
)
def test_dual_review_and_disagreement_rules_are_enforced(
    tmp_path: Path,
    fault: str,
) -> None:
    bundle = _bundle()
    if fault == "same_reviewer":
        bundle["records"][0]["reviews"][1]["reviewer_id"] = bundle["records"][0][
            "reviews"
        ][0]["reviewer_id"]
    elif fault == "missing_adjudication":
        bundle["records"][1]["adjudication"] = None
    elif fault == "early_adjudication":
        bundle["records"][1]["adjudication"]["adjudicated_at"] = (
            "2026-08-01T00:00:00+00:00"
        )
    else:
        bundle["records"][1]["adjudication"]["adjudicated_at"] = (
            "2026-08-01T02:00:00+00:00"
        )
    source = tmp_path / "bad.json"
    spec = tmp_path / "spec.json"
    _write_json(source, bundle)
    _write_json(spec, _run_spec())
    with pytest.raises(NMRBlindChallengeError):
        build_blind_challenge(source, spec, tmp_path / "public", tmp_path / "holder")


def test_unregistered_model_and_incomplete_ranking_are_rejected(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    unregistered = _predictions(paths, model="post-hoc-model")
    with pytest.raises(NMRBlindChallengeError, match="not frozen"):
        submit_blind_predictions(
            paths["public_manifest"],
            paths["roleless"],
            paths["run_spec"],
            unregistered,
            tmp_path / "ledger.jsonl",
        )

    spec = _run_spec()
    spec["model_release_ids"] = [{}]
    bad_spec = tmp_path / "bad-model-types.json"
    _write_json(bad_spec, spec)
    with pytest.raises(NMRBlindChallengeError, match="invalid non-empty text"):
        build_blind_challenge(
            paths["source"],
            bad_spec,
            tmp_path / "bad-types-public",
            tmp_path / "bad-types-holder",
        )
    predictions = json.loads(unregistered.read_bytes())
    predictions["model_release_id"] = "frozen-model-v5"
    _write_json(unregistered, predictions, canonical=True)
    with pytest.raises(NMRBlindChallengeError, match="manifest"):
        submit_blind_predictions(
            paths["public_manifest"],
            paths["roleless"],
            paths["run_spec"],
            unregistered,
            tmp_path / "ledger.jsonl",
        )

    predictions["model_release_manifest_sha256"] = sha256_bytes(
        canonical_json_bytes(_run_spec()["model_release_manifests"]["frozen-model-v5"])
    )
    predictions["records"][0]["ranked_candidate_ids"] = [[]]
    _write_json(unregistered, predictions, canonical=True)
    with pytest.raises(NMRBlindChallengeError, match="permutation|opaque"):
        submit_blind_predictions(
            paths["public_manifest"],
            paths["roleless"],
            paths["run_spec"],
            unregistered,
            tmp_path / "ledger.jsonl",
        )


def test_run_spec_binds_embedded_analysis_plan_and_model_manifests(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reviewed.json"
    _write_json(source, _bundle())

    plan_tamper = _run_spec()
    plan_tamper["analysis_plan"]["primary_test"] = "post-hoc endpoint"
    plan_path = tmp_path / "plan-tamper.json"
    _write_json(plan_path, plan_tamper)
    with pytest.raises(NMRBlindChallengeError, match="embedded analysis_plan"):
        build_blind_challenge(
            source,
            plan_path,
            tmp_path / "plan-public",
            tmp_path / "plan-holder",
        )

    manifest_gap = _run_spec()
    manifest_gap["model_release_manifests"] = {}
    manifest_path = tmp_path / "manifest-gap.json"
    _write_json(manifest_path, manifest_gap)
    with pytest.raises(NMRBlindChallengeError, match="exactly cover"):
        build_blind_challenge(
            source,
            manifest_path,
            tmp_path / "manifest-public",
            tmp_path / "manifest-holder",
        )

    legacy = _run_spec()
    legacy["schema_version"] = "chemapp.nmr.blind-run-spec.v1"
    legacy.pop("holder_public_key_spki_sha256")
    legacy_path = tmp_path / "legacy-v1.json"
    _write_json(legacy_path, legacy)
    with pytest.raises(NMRBlindChallengeError, match="legacy run-spec v1"):
        build_blind_challenge(
            source,
            legacy_path,
            tmp_path / "legacy-public",
            tmp_path / "legacy-holder",
        )


def test_event_time_order_is_enforced(tmp_path: Path) -> None:
    source = tmp_path / "reviewed.json"
    spec_path = tmp_path / "spec.json"
    _write_json(source, _bundle())
    spec = _run_spec()
    spec["registered_at"] = "2035-01-01T00:00:00+00:00"
    _write_json(spec_path, spec)
    public = tmp_path / "public"
    holder = tmp_path / "holder"
    build_blind_challenge(source, spec_path, public, holder)
    paths = {
        "public": public,
        "public_manifest": public / PUBLIC_MANIFEST_FILENAME,
        "roleless": public / ROLELESS_FILENAME,
        "run_spec": public / RUN_SPEC_FILENAME,
    }
    predictions = _predictions(paths)
    with pytest.raises(NMRBlindChallengeError, match="predate"):
        submit_blind_predictions(
            paths["public_manifest"],
            paths["roleless"],
            paths["run_spec"],
            predictions,
            tmp_path / "ledger.jsonl",
        )


def test_synchronized_manifest_count_forgery_is_rejected_by_review_audit(
    tmp_path: Path,
) -> None:
    paths, private_key, _, _ = _build_with_keys(tmp_path)
    public_manifest = json.loads(paths["public_manifest"].read_bytes())
    public_manifest["counts"]["excluded_records"] = 7
    _write_json(paths["public_manifest"], public_manifest, canonical=True)
    holder_manifest = json.loads(paths["holder_manifest"].read_bytes())
    holder_manifest["counts"]["excluded_records"] = 7
    public_payload = paths["public_manifest"].read_bytes()
    holder_manifest["artifacts"]["public_manifest"]["sha256"] = sha256_bytes(
        public_payload
    )
    holder_manifest["artifacts"]["public_manifest"]["size_bytes"] = len(public_payload)
    _write_json(paths["holder_manifest"], holder_manifest, canonical=True)
    predictions = _predictions(paths)
    ledger = tmp_path / "ledger.jsonl"
    submit_blind_predictions(
        paths["public_manifest"],
        paths["roleless"],
        paths["run_spec"],
        predictions,
        ledger,
        _clock=_clock(5),
    )
    with pytest.raises(NMRBlindChallengeError, match="review audit"):
        release_blind_gold(
            **_release_args(
                paths,
                predictions,
                ledger,
                private_key,
                tmp_path / "release",
            )
        )


def test_gold_is_committed_before_predictions(tmp_path: Path) -> None:
    paths, private_key, _, _ = _build_with_keys(tmp_path)
    gold = json.loads(paths["gold"].read_bytes())
    gold["sealing_nonce"] = "f" * 64
    _write_json(paths["gold"], gold, canonical=True)
    gold_payload = paths["gold"].read_bytes()
    holder_manifest = json.loads(paths["holder_manifest"].read_bytes())
    holder_manifest["artifacts"]["gold"]["sha256"] = sha256_bytes(gold_payload)
    holder_manifest["artifacts"]["gold"]["size_bytes"] = len(gold_payload)
    _write_json(paths["holder_manifest"], holder_manifest, canonical=True)
    predictions = _predictions(paths)
    ledger = tmp_path / "ledger.jsonl"
    submit_blind_predictions(
        paths["public_manifest"],
        paths["roleless"],
        paths["run_spec"],
        predictions,
        ledger,
        _clock=_clock(5),
    )
    with pytest.raises(NMRBlindChallengeError, match="pre-prediction commitment"):
        release_blind_gold(
            **_release_args(
                paths,
                predictions,
                ledger,
                private_key,
                tmp_path / "release",
            )
        )


def test_prediction_submission_and_gold_release_are_single_use(tmp_path: Path) -> None:
    paths, private_key, _, _ = _build_with_keys(tmp_path)
    predictions = _predictions(paths)
    ledger = tmp_path / "ledger.jsonl"
    submit_blind_predictions(
        paths["public_manifest"],
        paths["roleless"],
        paths["run_spec"],
        predictions,
        ledger,
        _clock=_clock(5),
    )
    with pytest.raises(NMRBlindChallengeError, match="replay"):
        submit_blind_predictions(
            paths["public_manifest"],
            paths["roleless"],
            paths["run_spec"],
            predictions,
            ledger,
        )
    release_blind_gold(
        **_release_args(paths, predictions, ledger, private_key, tmp_path / "release")
    )
    with pytest.raises(NMRBlindChallengeError, match="already exists"):
        release_blind_gold(
            **_release_args(
                paths, predictions, ledger, private_key, tmp_path / "release"
            )
        )


def test_detached_signature_and_smiles_mapping_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    paths, private_key, public_key, expected_key_sha = _build_with_keys(tmp_path)
    predictions = _predictions(paths)
    ledger = tmp_path / "ledger.jsonl"
    release = tmp_path / "release"
    submit_blind_predictions(
        paths["public_manifest"],
        paths["roleless"],
        paths["run_spec"],
        predictions,
        ledger,
        _clock=_clock(5),
    )
    release_blind_gold(
        **_release_args(paths, predictions, ledger, private_key, release)
    )
    bad_signature = tmp_path / "bad.sig"
    signature = bytearray((release / RELEASE_SIGNATURE_FILENAME).read_bytes())
    signature[0] ^= 1
    bad_signature.write_bytes(signature)
    with pytest.raises(NMRBlindChallengeError, match="signature is invalid"):
        verify_blind_release(
            release_manifest_path=release / RELEASE_MANIFEST_FILENAME,
            signature_path=bad_signature,
            public_key_path=public_key,
            expected_public_key_spki_sha256=expected_key_sha,
            public_manifest_path=paths["public_manifest"],
            holder_manifest_path=paths["holder_manifest"],
            roleless_path=paths["roleless"],
            run_spec_path=paths["run_spec"],
            gold_path=release / RELEASE_GOLD_FILENAME,
            predictions_path=predictions,
            review_audit_path=paths["audit"],
            receipt_ledger_path=ledger,
        )

    audit = json.loads(paths["audit"].read_bytes())
    included = next(row for row in audit["records"] if row["included"])
    included["candidate_map"][0]["candidate_smiles_sha256"] = _digest("wrong")
    tampered_audit = tmp_path / "tampered-audit.json"
    _write_json(tampered_audit, audit, canonical=True)
    with pytest.raises(NMRBlindChallengeError, match="binding|SHA-256"):
        verify_blind_release(
            release_manifest_path=release / RELEASE_MANIFEST_FILENAME,
            signature_path=release / RELEASE_SIGNATURE_FILENAME,
            public_key_path=public_key,
            expected_public_key_spki_sha256=expected_key_sha,
            public_manifest_path=paths["public_manifest"],
            holder_manifest_path=paths["holder_manifest"],
            roleless_path=paths["roleless"],
            run_spec_path=paths["run_spec"],
            gold_path=release / RELEASE_GOLD_FILENAME,
            predictions_path=predictions,
            review_audit_path=tampered_audit,
            receipt_ledger_path=ledger,
        )


def test_duplicate_keys_nonfinite_and_unexpected_fields_are_rejected(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "spec.json"
    _write_json(spec, _run_spec())
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"chemapp.nmr.blind-review-bundle.v1",'
        '"dataset_id":"a","dataset_id":"b","records":[]}',
        encoding="utf-8",
    )
    with pytest.raises(NMRBlindChallengeError, match="duplicate JSON key"):
        build_blind_challenge(
            duplicate,
            spec,
            tmp_path / "public-a",
            tmp_path / "holder-a",
        )
    bundle = _bundle()
    bundle["records"][0]["spectrum"]["shifts_ppm"][0] = float("nan")
    nonfinite = tmp_path / "nan.json"
    _write_json(nonfinite, bundle)
    with pytest.raises(NMRBlindChallengeError, match="non-finite"):
        build_blind_challenge(
            nonfinite,
            spec,
            tmp_path / "public-b",
            tmp_path / "holder-b",
        )
    bad_spec = _run_spec() | {"truth": "leak"}
    _write_json(spec, bad_spec)
    good = tmp_path / "good.json"
    _write_json(good, _bundle())
    with pytest.raises(NMRBlindChallengeError, match="field allowlist"):
        build_blind_challenge(
            good,
            spec,
            tmp_path / "public-c",
            tmp_path / "holder-c",
        )


def test_destinations_refuse_overwrite_and_nesting(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    with pytest.raises(NMRBlindChallengeError, match="already exists"):
        build_blind_challenge(
            paths["source"],
            tmp_path / "input-run-spec.json",
            paths["public"],
            tmp_path / "holder-2",
        )
    with pytest.raises(NMRBlindChallengeError, match="non-nested"):
        build_blind_challenge(
            paths["source"],
            tmp_path / "input-run-spec.json",
            tmp_path / "new-public",
            tmp_path / "new-public" / "holder",
        )


def test_symlinked_parent_is_rejected_when_supported(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("creating directory symlinks is not permitted on this host")
    source = real / "reviewed.json"
    spec = real / "spec.json"
    _write_json(source, _bundle())
    _write_json(spec, _run_spec())
    with pytest.raises(NMRBlindChallengeError, match="symlinked path component"):
        build_blind_challenge(
            linked / "reviewed.json",
            linked / "spec.json",
            tmp_path / "public",
            tmp_path / "holder",
        )
