from __future__ import annotations

import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from app.ml import external_nmr_benchmark as external
from app.ml.external_nmr_benchmark import (
    DERIVED_SCHEMA_VERSION,
    ExternalNMRBenchmarkError,
    build_external_nmr_bundle,
    derive_external_nmr_records,
    load_external_nmr_status,
    verify_external_nmr_bundle,
)


SMILES = ("C", "CC", "CCC", "CCCC", "CCO", "CCN")


def _jcamp(title: str) -> bytes:
    return (
        f"##TITLE={title}\n"
        "##JCAMPDX=6.0\n"
        "##DATA TYPE=NMR SPECTRUM\n"
        "##DATA CLASS=NTUPLES\n"
        "##.OBSERVE FREQUENCY=400.13\n"
        "##.OBSERVE NUCLEUS=^1H\n"
        "##.SOLVENT NAME=CDCl3\n"
        "##XUNITS=PPM\n"
        "##YUNITS=ARBITRARY UNITS\n"
        "##FIRSTX=10\n"
        "##LASTX=0\n"
        "##NPOINTS=3\n"
        "##XYDATA=(X++(Y..Y))\n"
        "10 1 2 3\n"
        "##END=\n"
    ).encode()


def _fixture_source(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    source = tmp_path / "source"
    source.mkdir()
    samples = []
    files = []
    for ordinal, smiles in enumerate(SMILES, start=1):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        molblock = Chem.MolToMolBlock(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        spectrum_name = f"spectra/nmr/sample-{ordinal}.jdx"
        sample = {
            "_id": f"uuid-{ordinal}",
            "$id": ["reference", f"sample-{ordinal}"],
            "$content": {
                "general": {
                    "name": [{"value": f"Sample {ordinal}"}],
                    "molfile": molblock,
                    "mf": formula,
                },
                "spectra": {
                    "nmr": [
                        {
                            "dimension": 1,
                            "nucleus": ["1H"],
                            "isFt": True,
                            "isComplex": False,
                            "title": f"Spectrum {ordinal}",
                            "solvent": "CDCl3",
                            "frequency": 400.13,
                            "temperature": 298.0,
                            "experiment": "1d",
                            "pulse": "zg30",
                            "range": [
                                {
                                    "from": 0.9,
                                    "to": 1.1,
                                    "integral": 3,
                                    "signal": [
                                        {
                                            "delta": 1.0,
                                            "multiplicity": "s",
                                            "diaID": [],
                                        }
                                    ],
                                }
                            ],
                            "jcamp": {"filename": spectrum_name},
                        }
                    ]
                },
            },
        }
        samples.append(sample)
        archive_path = source / f"{ordinal}.zip"
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(f"{ordinal}/structure.mol", molblock)
            archive.writestr(
                f"{ordinal}/index.json",
                json.dumps({"general": {"molfile": molblock}}),
            )
            archive.writestr(f"{ordinal}/{spectrum_name}", _jcamp(str(ordinal)))
        files.append(
            {
                "name": archive_path.name,
                "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            }
        )
    toc_path = source / "toc.json"
    toc_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    readme_path = source / "README.md"
    readme_path.write_text("fixture", encoding="utf-8")
    files.extend(
        [
            {
                "name": "toc.json",
                "sha256": hashlib.sha256(toc_path.read_bytes()).hexdigest(),
            },
            {
                "name": "README.md",
                "sha256": hashlib.sha256(readme_path.read_bytes()).hexdigest(),
            },
        ]
    )
    manifest = {
        "record_id": 16881130,
        "record_revision": 4,
        "doi": "10.5281/zenodo.16881130",
        "license": {"spdx_id": "CC0-1.0"},
        "inventory_sha256": "a" * 64,
        "scope": {
            "sample_count": 6,
            "role": "proof_of_concept_external_smoke_only",
            "headline_benchmark_allowed": False,
        },
        "files": files,
    }
    return source, manifest


def _index(path: Path, overlapping_key: str | None) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_metadata VALUES ('schema_version', '2');
        CREATE TABLE source_snapshots (
            id INTEGER PRIMARY KEY,
            source_name TEXT NOT NULL,
            source_version TEXT,
            sha256 TEXT NOT NULL
        );
        INSERT INTO source_snapshots
        VALUES (1, 'fixture-index', 'v1', 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb');
        CREATE TABLE molecules (
            id INTEGER PRIMARY KEY,
            source_record_ordinal INTEGER NOT NULL,
            inchi_key TEXT
        );
        """
    )
    if overlapping_key is not None:
        conn.execute(
            """
            INSERT INTO molecules(id, source_record_ordinal, inchi_key)
            VALUES (1, 1, ?)
            """,
            (f"{overlapping_key}-UHFFFAOYSA-N",),
        )
    conn.commit()
    conn.close()


def test_conversion_is_deterministic_non_extracting_and_grouped(
    tmp_path: Path,
) -> None:
    source, manifest = _fixture_source(tmp_path)
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.iterdir()
    }

    first = derive_external_nmr_records(manifest, source)
    second = derive_external_nmr_records(manifest, source)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    structures, spectra, groups, status = first
    assert len(structures) == 6
    assert len(spectra) == 6
    assert len(groups) == 6
    assert all(row["schema_version"] == DERIVED_SCHEMA_VERSION for row in spectra)
    assert all(row["jcamp"]["numeric_payload_decoded"] is True for row in spectra)
    assert all(
        row["jcamp"]["numeric_inspection"]["representation"]
        == {
            "spectrum_model_supported": True,
            "matrix_only": False,
            "verified_by": "NMRJCAMPParser.parse",
            "x_points": 3,
            "y_points": 3,
            "x_unit": "ppm",
            "x_axis_direction": "descending",
            "nucleus": "1H",
            "processing_qc_available": True,
        }
        for row in spectra
    )
    assert status["scope"]["numeric_payloads_decoded"] == 6
    assert status["scope"]["numeric_payloads_unsupported"] == 0
    assert status["scope"]["one_dimensional_spectrum_model_supported"] == 6
    assert status["scope"]["two_dimensional_matrix_only"] == 0
    assert status["conversion"]["archives_extracted"] is False
    assert status["accuracy_eligibility"]["headline_accuracy_allowed"] is False
    assert (
        status["partitioning"]["group_leakage_audit"]["molecule_key"]["leaked_groups"]
        == 0
    )
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.iterdir()
    } == before
    assert not list(source.rglob("*.mol"))
    assert not list(source.rglob("*.jdx"))


def test_exact_overlap_is_reported_and_blocks_independence(tmp_path: Path) -> None:
    source, manifest = _fixture_source(tmp_path)
    structures, _, _, _ = derive_external_nmr_records(manifest, source)
    index = tmp_path / "index.sqlite"
    _index(index, structures[0]["molecule_key"])

    _, _, _, status = derive_external_nmr_records(
        manifest,
        source,
        base_index_path=index,
    )

    overlap = status["base_index_overlap"]
    assert overlap["status"] == "checked"
    assert overlap["exact_molecule_overlaps"] == 1
    assert overlap["exact_overlap_rate"] == pytest.approx(1 / 6)
    assert status["accuracy_eligibility"]["independent_accuracy_test"] is False
    assert any("1/6" in reason for reason in status["accuracy_eligibility"]["reasons"])


def test_unsafe_toc_member_is_rejected_without_extraction(tmp_path: Path) -> None:
    source, manifest = _fixture_source(tmp_path)
    toc_path = source / "toc.json"
    toc = json.loads(toc_path.read_text(encoding="utf-8"))
    toc["samples"][0]["$content"]["spectra"]["nmr"][0]["jcamp"]["filename"] = (
        "../escape.jdx"
    )
    toc_path.write_text(json.dumps(toc), encoding="utf-8")
    next(item for item in manifest["files"] if item["name"] == "toc.json")["sha256"] = (
        hashlib.sha256(toc_path.read_bytes()).hexdigest()
    )

    with pytest.raises(ExternalNMRBenchmarkError, match="unsafe archive member"):
        derive_external_nmr_records(manifest, source)

    assert not (tmp_path / "escape.jdx").exists()


def test_build_and_verify_only_recompute_every_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, manifest = _fixture_source(tmp_path)
    output = tmp_path / "derived"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(external, "load_fixed_manifest", lambda _path: manifest)
    monkeypatch.setattr(external, "verify_dataset", lambda *_args, **_kwargs: [])

    built = build_external_nmr_bundle(
        manifest_path=manifest_path,
        source_directory=source,
        output_directory=output,
    )
    before = {
        path.name: (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in output.iterdir()
    }
    verified = verify_external_nmr_bundle(
        manifest_path=manifest_path,
        source_directory=source,
        output_directory=output,
    )

    assert built["status"] == "verified_smoke_only"
    assert verified["verification"] == {
        "status": "verified",
        "verify_only": True,
        "files_written": 0,
        "source_reverified": True,
        "derived_recomputed": True,
    }
    assert {
        path.name: (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in output.iterdir()
    } == before
    assert load_external_nmr_status(output / "summary.json")["scope"]["samples"] == 6

    with pytest.raises(ExternalNMRBenchmarkError, match="without --overwrite"):
        build_external_nmr_bundle(
            manifest_path=manifest_path,
            source_directory=source,
            output_directory=output,
        )


def test_verify_only_rejects_derived_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, manifest = _fixture_source(tmp_path)
    output = tmp_path / "derived"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(external, "load_fixed_manifest", lambda _path: manifest)
    monkeypatch.setattr(external, "verify_dataset", lambda *_args, **_kwargs: [])
    build_external_nmr_bundle(
        manifest_path=manifest_path,
        source_directory=source,
        output_directory=output,
    )
    with (output / "spectra.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(
        ExternalNMRBenchmarkError,
        match="recomputed source-derived records",
    ):
        verify_external_nmr_bundle(
            manifest_path=manifest_path,
            source_directory=source,
            output_directory=output,
        )


def test_output_cannot_be_inside_source_tree(tmp_path: Path) -> None:
    source, _ = _fixture_source(tmp_path)
    with pytest.raises(ExternalNMRBenchmarkError, match="descendants"):
        external._ensure_separate_output(source, source / "derived")
