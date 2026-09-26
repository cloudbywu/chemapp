from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import io

from fastapi.testclient import TestClient

import app.api.store as store_module
from app.main import app


DATA_DIR = Path(__file__).parent.parent.parent / "dataexample"


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    store_module._store = None
    return TestClient(app)


def _preview_and_execute(client, name: str, args: dict):
    preview = client.post(
        "/api/ai/actions/preview",
        json={"name": name, "args": args},
    )
    assert preview.status_code == 200, preview.text
    return client.post(
        "/api/ai/actions/execute",
        json={
            "name": name,
            "args": args,
            "preview_token": preview.json()["preview_token"],
        },
    )


def test_example_list_and_load(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        examples = client.get("/api/spectra/examples")
        assert examples.status_code == 200
        assert any(item["technique"] == "UV-Vis" for item in examples.json())

        loaded = client.post("/api/spectra/examples/load", json={"path": "紫外example/LA.txt"})
        assert loaded.status_code == 200, loaded.text
        data = loaded.json()
        assert data["id"]
        assert data["technique"] == "UV-Vis"


def test_analyze_options_add_quality_report(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        with (DATA_DIR / "紫外example" / "LA.txt").open("rb") as f:
            upload = client.post("/api/upload", files={"file": ("LA.txt", f, "text/plain")})
        assert upload.status_code == 200, upload.text
        sid = upload.json()["id"]

        analysis = client.post(
            f"/api/analyze/{sid}",
            json={
                "options": {"baseline_correct": True, "height_fraction": 0.02},
                "expected_revision": 0,
            },
        )
        assert analysis.status_code == 200, analysis.text
        result = analysis.json()
        assert result["metrics"]["baseline_corrected"] is True
        assert result["metrics"]["quality"]["status"] in {"good", "review", "poor"}


def test_markdown_report_export(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        loaded = client.post("/api/spectra/examples/load", json={"path": "紫外example/SE.txt"})
        assert loaded.status_code == 200, loaded.text
        sid = loaded.json()["id"]

        report = client.post("/api/reports/markdown", json={"ids": [sid], "title": "Test Report"})
        assert report.status_code == 200, report.text
        assert "text/markdown" in report.headers["content-type"]
        assert "# Test Report" in report.text
        assert "Data Quality" in report.text or "Quality:" in report.text


def test_manual_result_save_and_integrate(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        with (DATA_DIR / "紫外example" / "LA.txt").open("rb") as f:
            upload = client.post("/api/upload", files={"file": ("LA.txt", f, "text/plain")})
        assert upload.status_code == 200, upload.text
        sid = upload.json()["id"]

        analysis = client.post(f"/api/analyze/{sid}", json={"expected_revision": 0})
        assert analysis.status_code == 200, analysis.text
        result = analysis.json()
        assert result["peaks"]

        integration = client.post(
            f"/api/results/{sid}/integrate",
            json={"ranges": [{"start": 250, "end": 300, "center": 275}]},
        )
        assert integration.status_code == 200, integration.text
        assert integration.json()["integrals"][0]["area"] >= 0

        peaks = result["peaks"][:1]
        peaks.append({
            "position": 333.3,
            "intensity": 0.42,
            "area": None,
            "width": None,
            "assignment": "manual",
            "multiplicity": "",
            "coupling_constant": None,
        })
        saved = client.put(
            f"/api/results/{sid}/manual",
            json={
                "peaks": peaks,
                "metrics": result["metrics"],
                "summary": result["summary"],
                "expected_revision": result["result_revision"],
            },
        )
        assert saved.status_code == 200, saved.text
        data = saved.json()
        assert data["metrics"]["manual_confirmed"] is True
        assert data["metrics"]["n_peaks"] == 2

        versions = client.get(f"/api/results/{sid}/versions")
        assert versions.status_code == 200, versions.text
        assert versions.json()["versions"][0]["version"] == 1

        restored = client.post(
            f"/api/results/{sid}/versions/1/restore",
            json={"expected_revision": data["result_revision"]},
        )
        assert restored.status_code == 200, restored.text
        assert restored.json()["metrics"]["restored_from_version"] == 1


def test_batch_analyze_endpoint(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        first = client.post("/api/spectra/examples/load", json={"path": "紫外example/LA.txt"})
        second = client.post("/api/spectra/examples/load", json={"path": "紫外example/SE.txt"})
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text

        ids = [first.json()["id"], second.json()["id"]]
        batch = client.post(
            "/api/analyze/batch",
            json={"ids": ids, "expected_revisions": {sid: 0 for sid in ids}},
        )
        assert batch.status_code == 200, batch.text
        data = batch.json()
        assert len(data["results"]) == 2
        assert data["errors"] == []
        assert all(item["quality"]["status"] in {"good", "review", "poor"} for item in data["results"])


def test_batch_csv_zip_export(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        first = client.post("/api/spectra/examples/load", json={"path": "紫外example/LA.txt"})
        second = client.post("/api/spectra/examples/load", json={"path": "液相色谱example/-S-001.sirslt/-S-001.dx"})
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        client.post(
            f"/api/analyze/{first.json()['id']}",
            json={"expected_revision": 0},
        )

        response = client.post("/api/spectra/export/csv.zip", json={"ids": [first.json()["id"], second.json()["id"]]})
        assert response.status_code == 200, response.text
        assert "application/zip" in response.headers["content-type"]
        with ZipFile(io.BytesIO(response.content)) as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert any(name.endswith(".csv") for name in names)
            assert any(name.endswith(".result.json") for name in names)


def test_hplc_linear_baseline_integration_and_manual_channel_save(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        loaded = client.post("/api/spectra/examples/load", json={"path": "液相色谱example/-S-001.sirslt/-S-001.dx"})
        assert loaded.status_code == 200, loaded.text
        sid = loaded.json()["id"]

        analysis = client.post(f"/api/analyze/{sid}", json={"expected_revision": 0})
        assert analysis.status_code == 200, analysis.text
        result = analysis.json()
        first_peak = result["metrics"]["channel_peaks"]["DAD1A"]["peaks"][0]

        integration = client.post(
            f"/api/results/{sid}/integrate",
            json={"ranges": [{
                "start": first_peak["begin_time"],
                "end": first_peak["end_time"],
                "center": first_peak["position"],
                "channel": "DAD1A",
                "baseline": "linear",
            }]},
        )
        assert integration.status_code == 200, integration.text
        recalculated = integration.json()["integrals"][0]
        assert recalculated["area"] > 0
        assert recalculated["height"] > 0
        assert recalculated["width"] > 0

        channel_peaks = result["metrics"]["channel_peaks"]
        channel_peaks["DAD1A"]["peaks"][0]["area"] = recalculated["area"]
        saved = client.put(
            f"/api/results/{sid}/manual",
            json={
                "peaks": result["peaks"],
                "channel_peaks": channel_peaks,
                "metrics": result["metrics"],
                "summary": result["summary"],
                "expected_revision": result["result_revision"],
            },
        )
        assert saved.status_code == 200, saved.text
        saved_data = saved.json()
        assert saved_data["metrics"]["manual_confirmed"] is True
        assert saved_data["metrics"]["channel_peaks"]["DAD1A"]["source"] == "manual_confirmed"
        assert saved_data["metrics"]["channel_peaks"]["DAD1A"]["peaks"][0]["area_percent"] > 0


def test_inference_evidence_table(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        uv = client.post("/api/spectra/examples/load", json={"path": "紫外example/LA.txt"})
        hplc = client.post("/api/spectra/examples/load", json={"path": "液相色谱example/-S-001.sirslt/-S-001.dx"})
        assert uv.status_code == 200, uv.text
        assert hplc.status_code == 200, hplc.text

        response = client.post("/api/inference", json={"ids": [uv.json()["id"], hplc.json()["id"]]})
        assert response.status_code == 200, response.text
        inference = response.json()["inference"]
        evidence = inference["technique_results"]["_evidence_table"]["items"]
        assert len(evidence) >= 2
        assert {item["technique"] for item in evidence} >= {"UV-Vis", "HPLC"}
        assert "Evidence Table" in response.json()["report_markdown"]


def test_hplc_batch_peak_matching(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        first = client.post("/api/spectra/examples/load", json={"path": "液相色谱example/-S-001.sirslt/-S-001.dx"})
        second = client.post("/api/spectra/examples/load", json={"path": "液相色谱example/-S-002.sirslt/-S-002.dx"})
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text

        comparison = client.post("/api/compare/hplc", json={"ids": [first.json()["id"], second.json()["id"]], "channel": "DAD1A"})
        assert comparison.status_code == 200, comparison.text
        data = comparison.json()
        assert data["channel"] == "DAD1A"
        assert len(data["drift"]) == 2
        assert len(data["rows"]) > 0
        assert data["drift"][0]["matched_peaks"] > 0

        csv_response = client.post("/api/compare/hplc.csv", json={"ids": [first.json()["id"], second.json()["id"]], "channel": "DAD1A"})
        assert csv_response.status_code == 200, csv_response.text
        assert "text/csv" in csv_response.headers["content-type"]
        assert "HPLC RT Drift" in csv_response.text


def test_nmr_hide_solvent_and_manual_multiplet_ranges(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        loaded = client.post("/api/spectra/examples/load", json={"path": "HNMRexample.zip"})
        assert loaded.status_code == 200, loaded.text
        sid = loaded.json()["id"]

        visible = client.post(
            f"/api/analyze/{sid}",
            json={"options": {"auto_reference": False}, "expected_revision": 0},
        )
        assert visible.status_code == 200, visible.text
        hidden = client.post(
            f"/api/analyze/{sid}",
            json={
                "options": {
                    "hide_solvent_peaks": True,
                    "solvent_tolerance_ppm": 0.08,
                },
                "expected_revision": visible.json()["result_revision"],
            },
        )
        assert hidden.status_code == 200, hidden.text
        assert "solvent_peaks" in hidden.json()["metrics"]
        assert hidden.json()["metrics"]["solvent_hidden"] is True
        assert len(hidden.json()["peaks"]) <= len(visible.json()["peaks"])

        manual = client.post(
            f"/api/analyze/{sid}",
            json={
                "options": {
                    "multiplet_ranges": [
                        {"start": 8.18, "end": 8.06},
                        {"start": 4.10, "end": 3.96},
                    ]
                },
                "expected_revision": hidden.json()["result_revision"],
            },
        )
        assert manual.status_code == 200, manual.text
        assert len(manual.json()["multiplets"]) <= 2

        saved = client.put(
            f"/api/results/{sid}/manual",
            json={
                "peaks": manual.json()["peaks"],
                "integrals": manual.json()["integrals"],
                "multiplets": manual.json()["multiplets"],
                "metrics": manual.json()["metrics"],
                "summary": manual.json()["summary"],
                "note": "NMR multiplet confirmation",
                "expected_revision": manual.json()["result_revision"],
            },
        )
        assert saved.status_code == 200, saved.text
        assert saved.json()["metrics"]["manual_confirmed"] is True
        assert saved.json()["metrics"]["n_multiplets"] == len(manual.json()["multiplets"])


def test_xrd_custom_phase_matching(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        loaded = client.post("/api/spectra/examples/load", json={"path": "XRDexample/CdS-1_Theta_2-Theta.asc"})
        assert loaded.status_code == 200, loaded.text
        sid = loaded.json()["id"]

        custom = [{
            "name": "User CdS-like",
            "formula": "CdS",
            "crystal_system": "hexagonal",
            "peaks": [
                {"two_theta": 24.8, "hkl": "100", "rel_intensity": 65},
                {"two_theta": 26.5, "hkl": "002", "rel_intensity": 100},
                {"two_theta": 28.2, "hkl": "101", "rel_intensity": 78},
            ],
        }]
        analysis = client.post(
            f"/api/analyze/{sid}",
            json={"options": {"custom_phases": custom}, "expected_revision": 0},
        )
        assert analysis.status_code == 200, analysis.text
        result = analysis.json()
        assert result["metrics"]["analysis_options"]["custom_phases"] == 1
        assert any(phase["phase_name"] == "User CdS-like" for phase in result["phase_matches"])


def test_batch_quality_summary(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        uv = client.post("/api/spectra/examples/load", json={"path": "紫外example/LA.txt"})
        hplc = client.post("/api/spectra/examples/load", json={"path": "液相色谱example/-S-001.sirslt/-S-001.dx"})
        assert uv.status_code == 200, uv.text
        assert hplc.status_code == 200, hplc.text

        response = client.post("/api/quality/batch", json={"ids": [uv.json()["id"], hplc.json()["id"]]})
        assert response.status_code == 200, response.text
        data = response.json()
        assert len(data["items"]) == 2
        assert sum(data["counts"].values()) == 2
        assert all(item["status"] in {"good", "review", "poor"} for item in data["items"])


def test_ai_actions_modify_results(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        nmr = client.post("/api/spectra/examples/load", json={"path": "HNMRexample.zip"})
        assert nmr.status_code == 200, nmr.text
        nmr_id = nmr.json()["id"]
        analyzed = client.post(
            f"/api/analyze/{nmr_id}", json={"expected_revision": 0}
        )
        assert analyzed.status_code == 200, analyzed.text

        update = _preview_and_execute(
            client,
            "update_nmr_integral_range",
            {
                "spectrum_id": nmr_id,
                "index": 0,
                "start": 8.42,
                "end": 8.37,
                "center": 8.39,
                "expected_revision": analyzed.json()["result_revision"],
            },
        )
        assert update.status_code == 200, update.text
        assert update.json()["result"]["ok"] is True
        assert update.json()["result"]["integral"]["start_ppm"] == 8.42

        delete = _preview_and_execute(
            client,
            "delete_peak",
            {
                "spectrum_id": nmr_id,
                "index": 0,
                "expected_revision": update.json()["result"]["result_revision"],
            },
        )
        assert delete.status_code == 200, delete.text
        assert delete.json()["result"]["removed"]


def test_ai_actions_hplc_and_cross_inference(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        hplc = client.post("/api/spectra/examples/load", json={"path": "液相色谱example/-S-001.sirslt/-S-001.dx"})
        uv = client.post("/api/spectra/examples/load", json={"path": "紫外example/LA.txt"})
        assert hplc.status_code == 200, hplc.text
        assert uv.status_code == 200, uv.text
        hplc_id = hplc.json()["id"]
        uv_id = uv.json()["id"]
        result = client.post(
            f"/api/analyze/{hplc_id}", json={"expected_revision": 0}
        ).json()
        peak = result["metrics"]["channel_peaks"]["DAD1A"]["peaks"][0]

        reintegrate = _preview_and_execute(
            client,
            "hplc_reintegrate_peak",
            {
                "spectrum_id": hplc_id,
                "channel": "DAD1A",
                "index": 0,
                "start": peak["begin_time"],
                "end": peak["end_time"],
                "expected_revision": result["result_revision"],
            },
        )
        assert reintegrate.status_code == 200, reintegrate.text
        assert reintegrate.json()["result"]["peak"]["area"] > 0

        inference = client.post("/api/ai/actions/execute", json={
            "name": "run_cross_inference",
            "args": {"ids": [hplc_id, uv_id]},
        })
        assert inference.status_code == 200, inference.text
        assert inference.json()["result"]["inference"]["technique_results"]["_evidence_table"]["items"]


def test_nmr_processing_updates_spectrum_and_clears_result(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        loaded = client.post("/api/spectra/examples/load", json={"path": "HNMRexample.zip"})
        assert loaded.status_code == 200, loaded.text
        sid = loaded.json()["id"]

        analysis = client.post(f"/api/analyze/{sid}", json={"expected_revision": 0})
        assert analysis.status_code == 200, analysis.text

        processed = client.post(
            f"/api/nmr/{sid}/process",
            json={
                "baseline_correct": True,
                "baseline_percentile": 10,
                "reference_current_ppm": 7.3,
                "reference_target_ppm": 7.26,
                "normalize": True,
                "normalize_ppm": 7.26,
                "smoothing_window": 5,
                "crop_min_ppm": 0,
                "crop_max_ppm": 10,
                "expected_revision": loaded.json()["spectrum_revision"],
            },
        )
        assert processed.status_code == 200, processed.text
        data = processed.json()
        assert data["parameters"]["processed"] is True
        assert len(data["parameters"]["processing_history"]) >= 4
        assert len(data["x_data"]) < 32768

        result = client.get(f"/api/results/{sid}")
        assert result.status_code == 404


def test_ai_preview_suggest_undo_and_workbench(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        nmr = client.post("/api/spectra/examples/load", json={"path": "HNMRexample.zip"})
        assert nmr.status_code == 200, nmr.text
        sid = nmr.json()["id"]
        result = client.post(
            f"/api/analyze/{sid}", json={"expected_revision": 0}
        ).json()
        first = result["integrals"][0]

        preview = client.post("/api/ai/actions/preview", json={
            "name": "update_nmr_integral_range",
            "args": {
                "spectrum_id": sid,
                "index": 0,
                "start": first["start_ppm"],
                "end": first["end_ppm"],
                "expected_revision": result["result_revision"],
            },
        })
        assert preview.status_code == 200, preview.text
        assert preview.json()["preview"]["current"]["center_ppm"] == first["center_ppm"]

        update_args = {
            "spectrum_id": sid,
            "index": 0,
            "start": first["start_ppm"] + 0.01,
            "end": first["end_ppm"] - 0.01,
            "expected_revision": result["result_revision"],
        }
        update = _preview_and_execute(
            client, "update_nmr_integral_range", update_args
        )
        assert update.status_code == 200, update.text
        assert update.json()["result"]["previous_version"] is not None

        undo = _preview_and_execute(
            client,
            "undo_last_ai_action",
            {
                "spectrum_id": sid,
                "expected_revision": update.json()["result"]["result_revision"],
            },
        )
        assert undo.status_code == 200, undo.text
        assert undo.json()["result"]["undone_action"] == "update_nmr_integral_range"

        suggestions = client.post("/api/ai/actions/suggest", json={"ids": [sid]})
        assert suggestions.status_code == 200, suggestions.text
        assert "suggestions" in suggestions.json()

        workbench = client.post("/api/batch/workbench", json={"ids": [sid]})
        assert workbench.status_code == 200, workbench.text
        assert workbench.json()["summary"]["count"] == 1


def test_report_docx_html_standards_hplc_events_xrd_refinement(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        hplc = client.post("/api/spectra/examples/load", json={"path": "液相色谱example/-S-001.sirslt/-S-001.dx"})
        xrd = client.post("/api/spectra/examples/load", json={"path": "XRDexample/CdS-1_Theta_2-Theta.asc"})
        assert hplc.status_code == 200, hplc.text
        assert xrd.status_code == 200, xrd.text
        hplc_id = hplc.json()["id"]
        xrd_id = xrd.json()["id"]

        hplc_result = client.post(
            f"/api/analyze/{hplc_id}",
            json={
                "options": {
                    "integration_events": [
                        {
                            "channel": "DAD1A",
                            "start": 0,
                            "end": 999,
                            "mode": "force_bb",
                        }
                    ]
                },
                "expected_revision": 0,
            },
        )
        assert hplc_result.status_code == 200, hplc_result.text
        first_peak = hplc_result.json()["metrics"]["channel_peaks"]["DAD1A"]["peaks"][0]
        assert first_peak["type"] == "BB"

        xrd_result = client.post(
            f"/api/analyze/{xrd_id}",
            json={"options": {"rietveld_enabled": True}, "expected_revision": 0},
        )
        assert xrd_result.status_code == 200, xrd_result.text
        assert xrd_result.json()["rietveld_refinement"]["rwp"] >= 0

        html = client.post("/api/reports/html", json={"ids": [hplc_id, xrd_id], "title": "Enhanced"})
        assert html.status_code == 200, html.text
        assert "<html" in html.text

        docx = client.post("/api/reports/docx", json={"ids": [hplc_id, xrd_id], "title": "Enhanced"})
        assert docx.status_code == 200, docx.text
        assert "wordprocessingml" in docx.headers["content-type"]

        standards = client.get("/api/standards", params={"technique": "XRD", "query": "CdS"})
        assert standards.status_code == 200, standards.text
        assert standards.json()["records"]

        match = client.post(f"/api/standards/match/{xrd_id}", json={"tolerance": 0.6})
        assert match.status_code == 200, match.text
        assert match.json()["matches"]


def test_nmr_elucidation_retrieval_and_mixture_without_t5(tmp_path, monkeypatch):
    # retrieval_db_v2.pt is a local data asset that is not shipped in the
    # repository (backend/data/*.pt is gitignored); skip on clean checkouts.
    import pytest
    from pathlib import Path

    from app.ml.nmr_structure_elucidation import _default_external_dir

    _candidates = []
    _base = _default_external_dir()
    if _base is not None:
        _candidates.append(_base / "data" / "graphdiff_cache" / "retrieval_db_v2.pt")
    _candidates.append(
        Path(__file__).resolve().parents[1] / "data" / "retrieval_db_v2.pt"
    )
    if not any(p.exists() for p in _candidates):
        pytest.skip("retrieval_db_v2.pt asset not present")
    monkeypatch.setenv("CHEMAPP_NMR_INDEX", str(tmp_path / "nmr_index.sqlite"))
    monkeypatch.setenv("CHEMAPP_NMR_DATA_DIR", str(tmp_path / "external"))
    with _client(tmp_path, monkeypatch) as client:
        status = client.post("/api/ml/elucidate/index/import", json={"source": "quick"})
        assert status.status_code == 200, status.text
        assert status.json()["records"] >= 1

        response = client.post("/api/ml/elucidate/predict", json={
            "peaks_13c": [{"shift": 18.3}, {"shift": 41.8}, {"shift": 203.0}],
            "peaks_1h": [{"shift": 1.2, "integral": 3}, {"shift": 7.2, "integral": 2}],
            "top_k": 5,
            "num_beams": 3,
        })
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["method"].startswith("formula-constrained spectral retrieval")
        assert data["result_type"] == "candidate_ranking_not_identification"
        assert data["candidates"]
        assert "mixture_analysis" in data
        assert data["index"]["total_records"] >= 1


def test_experiment_report_agent_generates_xrd_report(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        loaded = client.post("/api/spectra/examples/load", json={"path": "XRDexample/CdS-1_Theta_2-Theta.asc"})
        assert loaded.status_code == 200, loaded.text
        sid = loaded.json()["id"]

        handout = (
            "# X射线衍射分析\n\n"
            "实验项目名称：X射线衍射分析\n\n"
            "实验目的：\n"
            "1. 了解X射线衍射仪的基本构造及原理。\n"
            "2. 掌握物相分析、晶粒尺寸和晶格常数分析方法。\n\n"
            "实验原理：\n"
            "X射线衍射遵循布拉格方程 2d sinθ = nλ。\n\n"
            "思考题:\n"
            "1. 择优取向是什么意思，它如何影响XRPD图谱？\n"
            "2. 写出常用的Scherrer公式，并解释公式中各物理量的含义及使用注意事项。\n"
        ).encode("utf-8")

        response = client.post(
            "/api/experiment-agent/report",
            data={
                "ids": f'["{sid}"]',
                "name": "测试学生",
                "student_id": "PB00000000",
                "use_llm": "false",
            },
            files={"handout": ("xrd.md", handout, "text/markdown")},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert "X射线衍射分析" in data["title"]
        assert "实验数据与分析" in data["markdown"]
        assert "物相" in data["markdown"] or "XRD" in data["markdown"]
        assert data["used_llm"] is False
        assert data["steps"][-1]["status"] == "done"

        docx = client.post(
            "/api/experiment-agent/report/docx",
            data={"ids": f'["{sid}"]', "use_llm": "false"},
            files={"handout": ("xrd.md", handout, "text/markdown")},
        )
        assert docx.status_code == 200, docx.text
        assert "wordprocessingml" in docx.headers["content-type"]


def test_experiment_report_agent_supports_four_lab_types(tmp_path, monkeypatch):
    cases = [
        {
            "example": "XRDexample/CdS-1_Theta_2-Theta.asc",
            "handout": "实验项目名称：X射线衍射分析\n实验目的：掌握物相分析、RIR定量、Scherrer公式和晶格常数分析。\n思考题:\n1. 写出常用的Scherrer公式，并解释公式中各物理量的含义及使用注意事项。",
            "expected": ["实验类型：xrd", "物相定性分析", "Scherrer"],
        },
        {
            "example": "液相色谱example/-S-001.sirslt/-S-001.dx",
            "handout": "实验项目名称：液相色谱仪分离测定饮料中的咖啡因\n实验目的：掌握反相色谱、保留时间、分离度、理论塔板数和标准曲线定量方法。\n思考题:\n1. 反相分配色谱的分离原理是什么？流动相的改变会对色谱图造成哪些影响及原因。",
            "expected": ["实验类型：hplc", "咖啡因", "反相分配色谱"],
        },
        {
            "example": "紫外example/LA.txt",
            "handout": "实验项目名称：紫外-可见分光光度法&分子荧光分析法\n实验目的：掌握Lambert-Beer定律、紫外吸收光谱和分子荧光分析。\n思考题:\n1. 被测物浓度过大或过小对测量有何影响？应如何调整？",
            "expected": ["实验类型：uv_fluorescence", "Lambert-Beer", "紫外"],
        },
        {
            "example": "伏安法example/50mV_s.txt",
            "handout": "实验项目名称：循环伏安法和交流阻抗测试\n实验目的：掌握循环伏安CV、交流阻抗EIS、Nyquist图和Randles-Sevcik方程。\n思考题:\n1. 循环伏安中峰电流与扫描速率有什么关系？",
            "expected": ["实验类型：electrochem", "循环伏安", "峰电流"],
        },
    ]
    with _client(tmp_path, monkeypatch) as client:
        for case in cases:
            loaded = client.post("/api/spectra/examples/load", json={"path": case["example"]})
            assert loaded.status_code == 200, loaded.text
            sid = loaded.json()["id"]
            response = client.post(
                "/api/experiment-agent/report",
                data={"ids": f'["{sid}"]', "use_llm": "false"},
                files={"handout": ("handout.md", case["handout"].encode("utf-8"), "text/markdown")},
            )
            assert response.status_code == 200, response.text
            markdown = response.json()["markdown"]
            for needle in case["expected"]:
                assert needle in markdown
