from __future__ import annotations

import pytest

from app.ml.nmr_evidence import (
    FormulaError,
    build_generation_prompt,
    canonical_formula,
    parse_formula,
    prepare_query_peaks,
    validate_generated_smiles,
)
from app.ml import nmr_structure_elucidation as elucidation


def test_formula_is_canonicalised_to_hill_order():
    info = parse_formula(" C10 H15 N2 Br ")
    assert info is not None
    assert info.canonical == "C10H15BrN2"
    assert info.elements == {"C": 10, "H": 15, "N": 2, "Br": 1}
    assert info.dbe == 4.0
    assert canonical_formula("NaCl") == "ClNa"


@pytest.mark.parametrize("formula", ["C6H5(OH)", "C6H6+", "13CH4", "C0H2", "C6H6.NaCl"])
def test_ambiguous_or_invalid_formula_is_rejected(formula):
    with pytest.raises(FormulaError):
        parse_formula(formula)


def test_query_preparation_removes_tms_and_declared_solvent_then_clusters_lines():
    peaks = [
        {"shift": -0.0024, "intensity": 0.95},
        {"shift": 7.2483, "intensity": 0.3},
        {"shift": 7.2821, "intensity": 0.2},
        {"shift": 1.2389, "intensity": 0.7},
        {"shift": 1.2524, "intensity": 1.0},
        {"shift": 1.2659, "intensity": 0.8},
    ]
    prepared, audit = prepare_query_peaks(peaks, nucleus="1H", solvent="CDCl3")
    assert len(prepared) == 1
    assert prepared[0]["line_count"] == 3
    assert 1.24 < prepared[0]["shift"] < 1.26
    assert {item["reason"] for item in audit["excluded"]} == {
        "reference_peak",
        "solvent_peak",
    }


def test_query_preparation_keeps_zero_region_when_reference_filter_disabled():
    prepared, audit = prepare_query_peaks(
        [{"shift": 0.01, "intensity": 1}],
        nucleus="1H",
        exclude_references=False,
    )
    assert prepared[0]["shift"] == pytest.approx(0.01)
    assert audit["excluded"] == []


def test_peak_matching_is_one_to_one():
    match = elucidation._match_score(
        [1.0, 1.01],
        [1.0],
        tolerance=0.18,
        nucleus_weight=1.0,
    )
    assert match["matched"] == 1
    assert len(match["assignments"]) == 1


def _insert_reference(
    conn,
    *,
    source_id: str,
    smiles: str,
    formula: str,
    peaks_1h: list[float],
):
    conn.execute(
        """
        INSERT INTO nmr_records
        (source, source_id, name, smiles, formula, mw, peaks_13c, peaks_1h, metadata)
        VALUES ('test', ?, '', ?, ?, 243.15, '[]', ?, '{}')
        """,
        (source_id, smiles, formula, __import__("json").dumps(peaks_1h)),
    )


def test_formula_search_is_canonical_strict_and_precedes_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR_INDEX", str(tmp_path / "index.sqlite"))
    monkeypatch.setenv("CHEMAPP_NMR_RANKER", str(tmp_path / "missing.joblib"))
    elucidation._load_records_cached.cache_clear()
    conn = elucidation._connect()
    _insert_reference(
        conn,
        source_id="other",
        smiles="CC",
        formula="C2H6",
        peaks_1h=[1.0],
    )
    _insert_reference(
        conn,
        source_id="target",
        smiles="CC(C)c1cnc(C(C)(C)Br)cn1",
        formula="C10H15BrN2",
        peaks_1h=[1.23, 1.55, 7.8, 8.0],
    )
    conn.commit()
    conn.close()

    result = elucidation.rank_candidates(
        peaks_1h=[{"shift": 1.23}, {"shift": 1.55}, {"shift": 7.8}, {"shift": 8.0}],
        formula="C10H15N2Br",
        top_k=5,
        max_records=1,
    )
    assert result["query"]["formula"] == "C10H15BrN2"
    assert result["candidate_pool_status"] == "formula_match"
    assert [candidate["source_id"] for candidate in result["candidates"]] == ["target"]
    assert "confidence" not in result["candidates"][0]
    assert result["candidates"][0]["ranking_score"] > 0


def test_formula_search_never_falls_back_to_unrelated_records(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR_INDEX", str(tmp_path / "index.sqlite"))
    monkeypatch.setenv("CHEMAPP_NMR_RANKER", str(tmp_path / "missing.joblib"))
    elucidation._load_records_cached.cache_clear()
    conn = elucidation._connect()
    _insert_reference(
        conn,
        source_id="other",
        smiles="CC",
        formula="C2H6",
        peaks_1h=[1.0],
    )
    conn.commit()
    conn.close()

    result = elucidation.rank_candidates(
        peaks_1h=[{"shift": 1.0}],
        formula="C10H15N2Br",
    )
    assert result["candidates"] == []
    assert result["candidate_pool_status"] == "no_formula_match"
    assert result["warnings"]


def test_generation_prompt_keeps_formula_precision_integral_and_multiplicity():
    prompt = build_generation_prompt(
        [{"shift": 123.456}],
        [{"shift": 1.23456, "integral": 3, "multiplicity": "d"}],
        formula="C10H15N2Br",
    )
    assert "formula: C10H15BrN2" in prompt
    assert "123.46" in prompt
    assert "1.235 integral=3.00 d" in prompt


def test_generated_smiles_are_validated_against_formula():
    accepted, audit = validate_generated_smiles(
        ["CC", "not-smiles", "CC", "CCO"],
        formula="C2H6",
    )
    assert [item["smiles"] for item in accepted] == ["CC"]
    assert accepted[0]["molecular_formula"] == "C2H6"
    assert audit["accepted"] == 1
    assert {item["reason"].split(":")[0] for item in audit["rejected"]} == {
        "invalid_smiles",
        "formula_mismatch",
    }
