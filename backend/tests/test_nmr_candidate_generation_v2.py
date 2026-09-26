"""Tests for the hybrid (index + PubChem) candidate generation module."""

from __future__ import annotations

import json

import pytest

from app.ml.nmr_candidate_generation_v1 import (
    _candidate_id,
    _connectivity_id,
    canonical_sha256,
)
from app.ml.nmr_candidate_generation_v2 import (
    HybridGenerationError,
    HybridOpenWorldProvider,
    PubChemFormulaSource,
    build_hybrid_open_world_bundle,
    evaluate_hybrid_coverage,
)


def _write_cache(cache_dir, formula: str, props: list[dict]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{formula}.json").write_text(
        json.dumps(
            {
                "formula": formula,
                "properties": props,
                "truncated": False,
                "fetched_at": "2026-08-06T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )


def test_pubchem_offline_cache_hit_and_miss(tmp_path) -> None:
    _write_cache(
        tmp_path,
        "C2H6O",
        [{"ConnectivitySMILES": "CCO", "InChIKey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"}],
    )
    source = PubChemFormulaSource(tmp_path, offline=True)
    assert source.candidate_smiles("C2H6O") == ["CCO"]
    with pytest.raises(HybridGenerationError):
        source.candidate_smiles("C3H8O")


def test_hybrid_provider_generates_formula_validated_candidates(tmp_path) -> None:
    _write_cache(
        tmp_path,
        "C2H6O",
        [
            {"ConnectivitySMILES": "CCO", "InChIKey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"},
            {"ConnectivitySMILES": "COC", "InChIKey": "XNWFRZJHXBZDAG-UHFFFAOYSA-N"},
            {"ConnectivitySMILES": "CC", "InChIKey": "WRONG-UHFFFAOYSA-N"},
        ],
    )
    provider = HybridOpenWorldProvider(
        "missing-index.sqlite", PubChemFormulaSource(tmp_path, offline=True)
    )
    candidates = provider.generate(
        formula="C2H6O", constraints={}, limit=10
    )
    smiles = [item["smiles"] for item in candidates]
    assert set(smiles) == {"CCO", "COC"}
    assert all(item["source_id"].startswith("candidate-") for item in candidates)


def test_hybrid_bundle_and_coverage(tmp_path) -> None:
    _write_cache(
        tmp_path,
        "C2H6O",
        [{"ConnectivitySMILES": "CCO", "InChIKey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"}],
    )
    index_bundle = {
        "index_binding": {"schema_version": "test"},
        "rows": [],
    }
    queries = [
        {"case_id": "case-aaaaaaaaaaaaaaaaaaaaaaaa", "formula": "C2H6O", "split": "dev"}
    ]
    bundle = build_hybrid_open_world_bundle(
        queries,
        index_bundle=index_bundle,
        pubchem=PubChemFormulaSource(tmp_path, offline=True),
        pool_limit=10,
    )
    assert bundle["schema_version"].endswith("v2")
    assert bundle["rows"][0]["candidate_count"] == 1
    gold = {
        "schema_version": "chemapp.nmr.holder-candidate-gold.v1",
        "rows": [
            {
                "case_id": "case-aaaaaaaaaaaaaaaaaaaaaaaa",
                "candidate_id": _candidate_id("CCO"),
                "connectivity_id": _connectivity_id("LFQSCWFLJHTTHZ"),
            }
        ],
        "mapping_sha256": "",
    }
    gold["mapping_sha256"] = canonical_sha256(gold["rows"])
    report = evaluate_hybrid_coverage(bundle, gold)
    assert report["overall"]["exact_identity_coverage"]["case_count"] == 1
    assert report["overall"]["connectivity_coverage"]["case_count"] == 1
