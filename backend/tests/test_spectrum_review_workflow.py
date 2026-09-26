from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.store as store_module
from app.core.models import Peak, SampleInfo, Spectrum, Technique
from app.main import app


ALICE_TOKEN = "alice-review-token-0001"
BOB_TOKEN = "bob-review-token-0000002"


def _spectrum(*, solvent: str = "CDCl3") -> Spectrum:
    return Spectrum(
        technique=Technique.NMR,
        x_data=np.asarray([8.0, 7.0, 6.0], dtype=np.float64),
        y_data=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        x_label="Chemical shift",
        y_label="Intensity",
        x_unit="ppm",
        parameters={"nucleus": "1H"},
        metadata=SampleInfo(
            name="review sample",
            formula="C2H6O",
            solvent=solvent,
        ),
        peaks=[
            Peak(position=7.0, intensity=1.0),
            Peak(position=6.5, intensity=0.5),
        ],
        source_file="review-sample.jdx",
    )


def _client(tmp_path, monkeypatch, token: str = ALICE_TOKEN) -> TestClient:
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.setenv(
        "CHEMAPP_REVIEWER_TOKENS",
        json.dumps({"alice": ALICE_TOKEN, "bob": BOB_TOKEN}),
    )
    monkeypatch.setenv(
        "CHEMAPP_REVIEW_ADMIN_SUBJECT",
        "curation-admin",
    )
    monkeypatch.delenv("CHEMAPP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
    store_module._store = None
    return TestClient(
        app,
        headers={"X-ChemApp-Reviewer-Token": token},
    )


def _enqueue_payload(
    spectrum_id: str,
    *,
    spectrum_revision: int = 1,
    result_revision: int = 0,
):
    return {
        "spectrum_id": spectrum_id,
        "structure_smiles": "CCO",
        "structure_source": "curator record",
        "molecule_id": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "source_collection": "independent-lab-2026",
        "source_record_id": f"record-{spectrum_id}",
        "independence_group": "lab-batch-001",
        "license_id": "CC-BY-4.0",
        "provenance_uri": "https://example.invalid/record",
        "rights_confirmed": True,
        "expected_spectrum_revision": spectrum_revision,
        "expected_result_revision": result_revision,
    }


def _accepted_review(item: dict):
    return {
        "verdict": "accept",
        "checks": {
            "structure": "pass",
            "nucleus": "pass",
            "axis": "pass",
            "peaks": "pass",
            "solvent": "pass",
        },
        "observations": {
            "structure_smiles": item["structure_smiles"],
            "nucleus": item["facts"]["nucleus"],
            "axis_unit": item["facts"]["axis_unit"],
            "axis_direction": item["facts"]["axis_direction"],
            "solvent": item["facts"]["solvent"],
            "peak_count": item["facts"]["peak_count"],
        },
        "notes": "independently checked",
        "expected_queue_revision": item["queue_revision"],
        "expected_snapshot_sha256": item["snapshot_sha256"],
    }


def test_review_identity_is_server_bound_and_fail_closed(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        identity = client.get("/api/reviews/me")
        assert identity.status_code == 200
        assert identity.json() == {"reviewer_id": "alice"}
        capabilities = client.get("/api/reviews/capabilities")
        assert capabilities.status_code == 200
        assert capabilities.json()["reviewer"] == {
            "configured": True,
            "authenticated": True,
            "subject": "alice",
        }
        assert capabilities.json()["admin"] == {
            "configured": True,
            "authenticated": True,
            "subject": "curation-admin",
        }

        forged = client.post(
            "/api/reviews/not-queued/submit",
            json={
                **_accepted_review(
                    {
                        "structure_smiles": "CCO",
                        "facts": {
                            "nucleus": "1H",
                            "axis_unit": "ppm",
                            "axis_direction": "descending",
                            "solvent": "CDCl3",
                            "peak_count": 2,
                        },
                        "queue_revision": 1,
                        "snapshot_sha256": "a" * 64,
                    }
                ),
                "reviewer_id": "bob",
            },
        )
        assert forged.status_code == 422

        wrong_token = client.get(
            "/api/reviews/me",
            headers={"X-ChemApp-Reviewer-Token": "wrong-review-token-0000"},
        )
        assert wrong_token.status_code == 401
        assert wrong_token.headers["www-authenticate"] == "ChemAppReviewer"

        monkeypatch.setenv("CHEMAPP_ACCESS_TOKEN", "general-access-token-000")
        reviewer_is_not_access = client.get("/api/spectra")
        assert reviewer_is_not_access.status_code == 401
        monkeypatch.delenv("CHEMAPP_ACCESS_TOKEN", raising=False)

        monkeypatch.setenv("CHEMAPP_ADMIN_TOKEN", "admin-token-correct-000")
        wrong_admin = client.get(
            "/api/reviews/gold-manifest",
            headers={"X-ChemApp-Admin-Token": "admin-token-wrong-00000"},
        )
        assert wrong_admin.status_code == 401
        assert wrong_admin.headers["www-authenticate"] == "ChemAppAdmin"
        monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)

    monkeypatch.delenv("CHEMAPP_REVIEWER_TOKENS", raising=False)
    store_module._store = None
    with TestClient(app) as client:
        disabled = client.get("/api/reviews/me")
        assert disabled.status_code == 503


def test_two_independent_reviews_export_gold_and_immutable_audit(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch) as alice:
        spectrum_id = store_module.get_store().add(_spectrum()).id
        queued_response = alice.post(
            "/api/reviews/queue",
            json=_enqueue_payload(spectrum_id),
        )
        assert queued_response.status_code == 200, queued_response.text
        queued = queued_response.json()
        assert queued["status"] == "pending"
        assert len(queued["snapshot_sha256"]) == 64

        first = alice.post(
            f"/api/reviews/{spectrum_id}/submit",
            json=_accepted_review(queued),
        )
        assert first.status_code == 200, first.text
        first_item = first.json()["item"]
        assert first_item["status"] == "awaiting_second"
        assert first_item["has_submitted"] is True

        with _client(tmp_path, monkeypatch, BOB_TOKEN) as bob:
            private_view = bob.get(f"/api/reviews/{spectrum_id}")
            assert private_view.status_code == 200
            bob_item = private_view.json()["item"]
            assert bob_item["review_count"] == 1
            assert bob_item["own_review"] is None

            second = bob.post(
                f"/api/reviews/{spectrum_id}/submit",
                json=_accepted_review(bob_item),
            )
            assert second.status_code == 200, second.text
            accepted = second.json()["item"]
            assert accepted["status"] == "accepted"
            assert accepted["final_decision"] == "accept"

            manifest_response = bob.get("/api/reviews/gold-manifest")
            assert manifest_response.status_code == 200
            manifest = manifest_response.json()
            assert manifest["item_count"] == 1
            assert manifest["benchmark_eligible"] is False
            assert manifest["requires_group_leakage_audit"] is True
            assert manifest["items"][0]["snapshot_sha256"] == (
                accepted["snapshot_sha256"]
            )
            assert set(manifest["items"][0]["review"]["reviewers"]) == {
                "alice",
                "bob",
            }
            assert manifest_response.headers["x-content-sha256"] == (
                manifest["manifest_sha256"]
            )

            audit = bob.get(f"/api/reviews/{spectrum_id}/audit")
            assert audit.status_code == 200
            assert [event["event_type"] for event in audit.json()["events"]] == [
                "enqueued",
                "review_submitted",
                "review_submitted",
            ]
            assert audit.json()["events"][0]["actor_id"] == "curation-admin"

        store = store_module.get_store()
        with store._connect() as conn:
            review_id = conn.execute(
                "SELECT id FROM spectrum_review_submissions LIMIT 1"
            ).fetchone()[0]
            try:
                conn.execute(
                    """
                    UPDATE spectrum_review_submissions
                    SET notes='changed'
                    WHERE id=?
                    """,
                    (review_id,),
                )
            except sqlite3.IntegrityError as error:
                assert "immutable" in str(error)
            else:
                raise AssertionError("Immutable review row was updated")
            queue_id = conn.execute(
                "SELECT id FROM spectrum_review_queue LIMIT 1"
            ).fetchone()[0]
            try:
                conn.execute(
                    """
                    UPDATE spectrum_review_queue
                    SET snapshot_sha256=?
                    WHERE id=?
                    """,
                    ("0" * 64, queue_id),
                )
            except sqlite3.IntegrityError as error:
                assert "immutable" in str(error)
            else:
                raise AssertionError("Immutable queue snapshot was updated")


def test_disagreement_requires_admin_adjudication(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as alice:
        spectrum_id = store_module.get_store().add(_spectrum()).id
        queued = alice.post(
            "/api/reviews/queue",
            json=_enqueue_payload(spectrum_id),
        ).json()
        first = alice.post(
            f"/api/reviews/{spectrum_id}/submit",
            json=_accepted_review(queued),
        ).json()["item"]

        with _client(tmp_path, monkeypatch, BOB_TOKEN) as bob:
            rejected_payload = {
                **_accepted_review(first),
                "verdict": "reject",
                "checks": {
                    "structure": "fail",
                    "nucleus": "pass",
                    "axis": "pass",
                    "peaks": "pass",
                    "solvent": "pass",
                },
                "notes": "structure identity does not match",
            }
            second = bob.post(
                f"/api/reviews/{spectrum_id}/submit",
                json=rejected_payload,
            )
            assert second.status_code == 200, second.text
            conflict = second.json()["item"]
            assert conflict["status"] == "conflict"

            monkeypatch.setenv("CHEMAPP_REVIEW_ADMIN_SUBJECT", "bob")
            overlapping = bob.post(
                f"/api/reviews/{spectrum_id}/adjudicate",
                json={
                    "decision": "reject",
                    "checks": {
                        "structure": "fail",
                        "nucleus": "pass",
                        "axis": "pass",
                        "peaks": "pass",
                        "solvent": "pass",
                    },
                    "reason": "The same reviewer may not adjudicate.",
                    "expected_queue_revision": conflict["queue_revision"],
                    "expected_snapshot_sha256": conflict["snapshot_sha256"],
                },
            )
            assert overlapping.status_code == 503
            monkeypatch.setenv(
                "CHEMAPP_REVIEW_ADMIN_SUBJECT",
                "curation-admin",
            )

            adjudicated = bob.post(
                f"/api/reviews/{spectrum_id}/adjudicate",
                json={
                    "decision": "accept",
                    "checks": {
                        "structure": "pass",
                        "nucleus": "pass",
                        "axis": "pass",
                        "peaks": "pass",
                        "solvent": "pass",
                    },
                    "reason": "Primary source record confirms the structure.",
                    "expected_queue_revision": conflict["queue_revision"],
                    "expected_snapshot_sha256": conflict["snapshot_sha256"],
                },
            )
            assert adjudicated.status_code == 200, adjudicated.text
            assert adjudicated.json()["status"] == "accepted"
            assert adjudicated.json()["final_reason"].startswith("Primary")

            audit = bob.get(f"/api/reviews/{spectrum_id}/audit").json()
            assert audit["events"][-1]["event_type"] == "conflict_adjudicated"
            assert audit["events"][-1]["actor_id"] == "curation-admin"


def test_spectrum_change_marks_review_stale_and_allows_new_cycle(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch) as alice:
        store = store_module.get_store()
        spectrum_id = store.add(_spectrum()).id
        queued = alice.post(
            "/api/reviews/queue",
            json=_enqueue_payload(spectrum_id),
        ).json()
        first = alice.post(
            f"/api/reviews/{spectrum_id}/submit",
            json=_accepted_review(queued),
        ).json()["item"]
        with _client(tmp_path, monkeypatch, BOB_TOKEN) as bob:
            accepted = bob.post(
                f"/api/reviews/{spectrum_id}/submit",
                json=_accepted_review(first),
            )
            assert accepted.status_code == 200

            changed = _spectrum()
            changed.y_data = np.asarray([0.0, 2.0, 0.0], dtype=np.float64)
            store = store_module.get_store()
            updated = store.set_spectrum(
                spectrum_id,
                changed,
                expected_revision=1,
            )
            assert updated is not None
            assert updated.spectrum_revision == 2

            stale = bob.get(f"/api/reviews/{spectrum_id}")
            assert stale.status_code == 200
            assert stale.json()["item"]["status"] == "stale"
            assert bob.get("/api/reviews/gold-manifest").json()["item_count"] == 0

            new_cycle = bob.post(
                "/api/reviews/queue",
                json=_enqueue_payload(
                    spectrum_id,
                    spectrum_revision=2,
                    result_revision=1,
                ),
            )
            assert new_cycle.status_code == 200, new_cycle.text
            assert new_cycle.json()["cycle"] == 2
            assert new_cycle.json()["snapshot_sha256"] != (
                queued["snapshot_sha256"]
            )
            assert store_module.get_store().remove(spectrum_id) is True
            deleted_audit = bob.get(
                f"/api/reviews/{spectrum_id}/audit"
            ).json()
            assert deleted_audit["queue"]["status"] == "stale"
            assert deleted_audit["events"][-1]["details"]["reason"] == (
                "spectrum_deleted"
            )


def test_enqueue_requires_current_revisions_and_confirmed_rights(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch) as client:
        spectrum_id = store_module.get_store().add(_spectrum()).id
        wrong_revision = client.post(
            "/api/reviews/queue",
            json=_enqueue_payload(spectrum_id, spectrum_revision=2),
        )
        assert wrong_revision.status_code == 409
        assert wrong_revision.json()["detail"]["code"] == (
            "review_revision_conflict"
        )

        without_rights = {
            **_enqueue_payload(spectrum_id),
            "rights_confirmed": False,
        }
        refused = client.post("/api/reviews/queue", json=without_rights)
        assert refused.status_code == 409
        assert "rights" in refused.json()["detail"]["message"].lower()


def test_enqueue_canonicalizes_structure_and_rejects_bad_governance(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch) as client:
        spectrum_id = store_module.get_store().add(_spectrum()).id
        base = _enqueue_payload(spectrum_id)

        mismatch = client.post(
            "/api/reviews/queue",
            json={**base, "molecule_id": "AAAAAAAAAAAAAA-BBBBBBBBBB-C"},
        )
        assert mismatch.status_code == 409
        assert "does not match" in mismatch.json()["detail"]["message"]

        fragments = client.post(
            "/api/reviews/queue",
            json={**base, "structure_smiles": "CCO.Cl"},
        )
        assert fragments.status_code == 409
        assert "one connected fragment" in fragments.json()["detail"]["message"]

        insecure_provenance = client.post(
            "/api/reviews/queue",
            json={**base, "provenance_uri": "http://example.test/record"},
        )
        assert insecure_provenance.status_code == 409
        assert "HTTPS" in insecure_provenance.json()["detail"]["message"]

        unknown_license = client.post(
            "/api/reviews/queue",
            json={**base, "license_id": "UNKNOWN"},
        )
        assert unknown_license.status_code == 409
        assert "not explicitly allowed" in (
            unknown_license.json()["detail"]["message"]
        )

        blank_source = client.post(
            "/api/reviews/queue",
            json={**base, "structure_source": "   "},
        )
        assert blank_source.status_code == 409

        canonical = client.post(
            "/api/reviews/queue",
            json={**base, "structure_smiles": "OCC"},
        )
        assert canonical.status_code == 200, canonical.text
        assert canonical.json()["structure_smiles"] == "CCO"
        assert canonical.json()["molecule_id"] == (
            "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        )


def test_gold_export_revalidates_evidence_instead_of_trusting_status(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch) as client:
        spectrum_id = store_module.get_store().add(_spectrum()).id
        queued = client.post(
            "/api/reviews/queue",
            json=_enqueue_payload(spectrum_id),
        )
        assert queued.status_code == 200
        store = store_module.get_store()
        with store._connect() as conn:
            conn.execute(
                """
                UPDATE spectrum_review_queue
                SET status='accepted', final_decision='accept',
                    final_reason='independent_reviews_agree'
                WHERE spectrum_id=?
                """,
                (spectrum_id,),
            )
            conn.commit()

        manifest = client.get("/api/reviews/gold-manifest")
        assert manifest.status_code == 200
        assert manifest.json()["item_count"] == 0


@pytest.mark.parametrize(
    ("trigger_name", "mutation_sql"),
    [
        (
            "trg_review_queue_snapshot_no_update",
            "UPDATE spectrum_review_queue SET license_id='UNKNOWN'",
        ),
        (
            "trg_review_queue_snapshot_no_update",
            "UPDATE spectrum_review_queue SET structure_smiles='not-smiles'",
        ),
        (
            "trg_review_queue_snapshot_no_update",
            """
            UPDATE spectrum_review_queue
            SET facts_json=json_set(facts_json, '$.nucleus', '13C')
            """,
        ),
        (
            "trg_review_event_no_update",
            """
            UPDATE spectrum_review_events
            SET actor_id='mallory'
            WHERE event_type='review_submitted'
            """,
        ),
    ],
)
def test_gold_export_rejects_corrupt_frozen_evidence_and_event_chain(
    tmp_path,
    monkeypatch,
    trigger_name,
    mutation_sql,
):
    with _client(tmp_path, monkeypatch) as alice:
        spectrum_id = store_module.get_store().add(_spectrum()).id
        queued = alice.post(
            "/api/reviews/queue",
            json=_enqueue_payload(spectrum_id),
        ).json()
        first = alice.post(
            f"/api/reviews/{spectrum_id}/submit",
            json=_accepted_review(queued),
        ).json()["item"]

        with _client(tmp_path, monkeypatch, BOB_TOKEN) as bob:
            accepted = bob.post(
                f"/api/reviews/{spectrum_id}/submit",
                json=_accepted_review(first),
            )
            assert accepted.status_code == 200, accepted.text
            assert bob.get("/api/reviews/gold-manifest").json()["item_count"] == 1

            store = store_module.get_store()
            with store._connect() as conn:
                # Simulate a legacy/corrupt database that predates immutable
                # triggers. Export must still fail closed on the evidence.
                conn.execute(f"DROP TRIGGER {trigger_name}")
                conn.execute(mutation_sql)
                conn.commit()

            manifest = bob.get("/api/reviews/gold-manifest")
            assert manifest.status_code == 200
            assert manifest.json()["item_count"] == 0


def test_review_admin_subject_is_required_even_on_loopback(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch) as client:
        spectrum_id = store_module.get_store().add(_spectrum()).id
        monkeypatch.delenv("CHEMAPP_REVIEW_ADMIN_SUBJECT", raising=False)
        response = client.post(
            "/api/reviews/queue",
            json=_enqueue_payload(spectrum_id),
        )
        assert response.status_code == 503
        assert "CHEMAPP_REVIEW_ADMIN_SUBJECT" in response.text
