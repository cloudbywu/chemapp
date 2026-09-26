"""Command-line toolkit for the v8 sealed-test state machine.

Primary flows:

- ``simulate --scratch-dir DIR``: run the full four-stage protocol end-to-end
  with a small synthetic cohort in simulation mode and write every artifact
  (reservation/attempt/commitment/receipt, ledgers, public keys) to DIR.
  Simulation artifacts are marked ``simulation: true`` and can never be
  mistaken for a real gate.
- ``verify --dir DIR``: load the artifacts written by ``simulate`` (or by a
  real holder using the same layout) and verify the whole chain.
- ``gen-keys --out-dir DIR``: generate holder/executor/evaluator keypairs.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .crypto import (
    generate_private_key_pem,
    load_private_key,
    public_key_from_private_pem,
    public_key_pem,
)
from .ledger import Denylist, Ledger
from .protocol import (
    SealedTestStateMachine,
    build_attempt,
    build_commitment,
    build_receipt,
    build_reservation_draft,
    sign_attempt,
    sign_commitment,
    sign_receipt,
    sign_reservation,
)

DUMMY_HASH = "0" * 64
FILENAMES = (
    "reservation.json",
    "attempt.json",
    "commitment.json",
    "receipt.json",
    "ledger.jsonl",
    "denylist.jsonl",
    "simulation-summary.json",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _gen_keypair(out_dir: Path, label: str) -> Path:
    pem = generate_private_key_pem()
    private = out_dir / f"{label}.key.pem"
    public = out_dir / f"{label}.pub.pem"
    private.write_bytes(pem)
    public.write_bytes(public_key_pem(public_key_from_private_pem(pem)))
    return private


def run_simulation(scratch_dir: Path) -> dict[str, Any]:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    holder_key = _gen_keypair(scratch_dir, "holder")
    executor_key = _gen_keypair(scratch_dir, "executor")
    evaluator_key = _gen_keypair(scratch_dir, "evaluator")

    machine = SealedTestStateMachine(simulation=True)
    record_count = 10
    group_count = 5
    query_ids = [f"q_{i:02d}" for i in range(record_count)]
    identities = [f"src-{query_id}" for query_id in query_ids]

    reservation_draft = build_reservation_draft(
        holder_id="simulation-holder",
        holder_public_key=public_key_from_private_pem(holder_key.read_bytes()),
        record_count=record_count,
        connected_group_count=group_count,
        protocol_sha256=DUMMY_HASH,
        task_manifest_sha256=DUMMY_HASH,
        model_release_sha256=DUMMY_HASH,
        generator_release_sha256=DUMMY_HASH,
        cohort_commitment_sha256=DUMMY_HASH,
        review_commitment_sha256=DUMMY_HASH,
        license_review_commitment_sha256=DUMMY_HASH,
        similarity_audit_commitment_sha256=DUMMY_HASH,
        query_bundle_sha256=DUMMY_HASH,
        prior_influence_index_sha256=DUMMY_HASH,
        denylist_commitment_sha256=DUMMY_HASH,
        created_at_utc=_now(),
        simulation=True,
    )
    reservation = sign_reservation(
        reservation_draft,
        load_private_key(holder_key),
        anchor_timestamp_utc=_now(),
    )
    machine.verify_reservation(
        reservation, public_key_from_private_pem(holder_key.read_bytes())
    )

    ledger = Ledger(scratch_dir / "ledger.jsonl")
    denylist = Denylist(scratch_dir / "denylist.jsonl")
    attempt = sign_attempt(
        build_attempt(
            reservation=reservation,
            attempt_id="att_" + secrets.token_hex(16),
            holder_nonce=secrets.token_hex(32),
            consumed_at_utc=_now(),
            ledger_receipt={
                "type": "worm",
                "provider": "simulation-local-ledger",
                "timestamp_utc": _now(),
                "receipt_sha256": DUMMY_HASH,
                "receipt_uri": "https://example.invalid/simulation-ledger",
                "validator": "simulation",
                "reservation_uniqueness_enforced": True,
            },
        ),
        load_private_key(holder_key),
    )
    consume_result = machine.consume(
        reservation,
        attempt,
        public_key_from_private_pem(holder_key.read_bytes()),
        ledger,
        denylist=denylist,
        identities=identities,
    )

    prediction_manifest = {
        "schema_version": "chemapp.nmr.v8-prediction-manifest.v1",
        "attempt_id": attempt["attempt_id"],
        "outcomes": {query_id: "success" for query_id in query_ids},
    }
    from .crypto import canonical_object_bytes, payload_sha256
    manifest_payload = canonical_object_bytes(prediction_manifest)
    prediction_manifest_sha256 = payload_sha256(manifest_payload)
    committed_at = _now()
    commitment = sign_commitment(
        build_commitment(
            reservation=reservation,
            attempt=attempt,
            ledger_entry=consume_result["ledger_entry"],
            query_manifest_sha256=DUMMY_HASH,
            prediction_manifest_sha256=prediction_manifest_sha256,
            query_count=record_count,
            prediction_count=record_count,
            succeeded_count=record_count,
            failed_count=0,
            prediction_committed_at_utc=committed_at,
            prediction_anchor={
                "type": "worm",
                "provider": "simulation-local-ledger",
                "timestamp_utc": _now(),
                "receipt_sha256": DUMMY_HASH,
                "receipt_uri": "https://example.invalid/simulation-anchor",
                "validator": "simulation",
                "anchored_payload_rule": "SHA-256 of the exact immutable prediction artifact bytes",
                "anchored_payload_sha256": prediction_manifest_sha256,
            },
            executor_id="simulation-executor",
        ),
        load_private_key(executor_key),
        load_private_key(holder_key),
    )
    machine.commit(
        reservation,
        attempt,
        consume_result["ledger_entry"],
        commitment,
        public_key_from_private_pem(holder_key.read_bytes()),
        public_key_from_private_pem(executor_key.read_bytes()),
        query_ids=query_ids,
        prediction_manifest=prediction_manifest,
    )

    gold_released_at = _now()
    evaluation_artifact = {
        "schema_version": "chemapp.nmr.v8-evaluation-artifact.v1",
        "attempt_id": attempt["attempt_id"],
        "outcomes": {query_id: {"top1": True, "status": "success"} for query_id in query_ids},
    }
    evaluation_payload = canonical_object_bytes(evaluation_artifact)
    receipt = sign_receipt(
        build_receipt(
            reservation=reservation,
            commitment=commitment,
            evaluation_mode="holder_private",
            total_queries=record_count,
            evaluated_queries=record_count,
            succeeded_queries=record_count,
            failed_queries=0,
            gold_released_at_utc=gold_released_at,
            gold_anchor={
                "type": "worm",
                "provider": "simulation-local-ledger",
                "timestamp_utc": _now(),
                "receipt_sha256": DUMMY_HASH,
                "receipt_uri": "https://example.invalid/simulation-gold",
                "validator": "simulation",
            },
            evaluation_completed_at_utc=_now(),
            evaluation_artifact_sha256=payload_sha256(evaluation_payload),
            evaluator_id="simulation-evaluator",
        ),
        load_private_key(evaluator_key),
        load_private_key(holder_key),
    )
    machine.evaluate(
        reservation,
        attempt,
        commitment,
        receipt,
        public_key_from_private_pem(holder_key.read_bytes()),
        public_key_from_private_pem(evaluator_key.read_bytes()),
        evaluation_artifact_bytes=evaluation_payload,
    )

    chain = machine.verify_chain(
        reservation,
        attempt,
        commitment,
        receipt,
        public_key_from_private_pem(holder_key.read_bytes()),
        public_key_from_private_pem(executor_key.read_bytes()),
        public_key_from_private_pem(evaluator_key.read_bytes()),
        ledger=ledger,
        denylist=denylist,
        identities=identities,
    )

    artifacts = {
        "reservation": reservation,
        "attempt": attempt,
        "commitment": commitment,
        "receipt": receipt,
    }
    for name, value in artifacts.items():
        _write(scratch_dir / f"{name}.json", value)
    summary = {
        "schema_version": "chemapp.nmr.v8-sealed-test-simulation.v1",
        "simulation": True,
        "generated_at_utc": _now(),
        "record_count": record_count,
        "connected_group_count": group_count,
        "chain_status": chain["status"],
        "problems": chain["problems"],
        "artifacts": FILENAMES,
        "notes": "Simulation artifacts are marked simulation=true and never unlock release gates.",
    }
    _write(scratch_dir / "simulation-summary.json", summary)
    return summary


def run_verify(scratch_dir: Path) -> dict[str, Any]:
    def load(name: str) -> dict[str, Any]:
        return json.loads((scratch_dir / name).read_text(encoding="utf-8"))

    reservation = load("reservation.json")
    attempt = load("attempt.json")
    commitment = load("commitment.json")
    receipt = load("receipt.json")
    ledger = Ledger(scratch_dir / "ledger.jsonl")
    denylist = Denylist(scratch_dir / "denylist.jsonl")
    machine = SealedTestStateMachine(simulation=True)
    holder_pub = public_key_from_private_pem((scratch_dir / "holder.key.pem").read_bytes())
    executor_pub = public_key_from_private_pem((scratch_dir / "executor.key.pem").read_bytes())
    evaluator_pub = public_key_from_private_pem((scratch_dir / "evaluator.key.pem").read_bytes())
    identities = [f"src-{query_id}" for query_id in [f"q_{i:02d}" for i in range(reservation["record_count"])]]
    return machine.verify_chain(
        reservation,
        attempt,
        commitment,
        receipt,
        holder_pub,
        executor_pub,
        evaluator_pub,
        ledger=ledger,
        denylist=denylist,
        identities=identities,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v8 sealed-test state machine toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run the full four-stage protocol as a simulation dry run")
    sim.add_argument("--scratch-dir", type=Path, default=Path(tempfile.gettempdir()) / "chemapp-sealed-test-sim")

    ver = sub.add_parser("verify", help="verify a simulated or holder-produced artifact directory")
    ver.add_argument("--dir", type=Path, required=True)

    keys = sub.add_parser("gen-keys", help="generate holder/executor/evaluator keypairs")
    keys.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "simulate":
            result = run_simulation(args.scratch_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["chain_status"] == "verified" else 2
        if args.command == "verify":
            result = run_verify(args.dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "verified" else 2
        if args.command == "gen-keys":
            args.out_dir.mkdir(parents=True, exist_ok=True)
            for label in ("holder", "executor", "evaluator"):
                path = _gen_keypair(args.out_dir, label)
                print(path)
            return 0
    except Exception as exc:  # pragma: no cover - CLI error surface
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
