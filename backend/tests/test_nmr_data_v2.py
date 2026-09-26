from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from app.ml.nmr_data_v2 import (
    SCHEMA_VERSION,
    SourceValidationError,
    build_nmr_index_v2,
    connect_readonly,
    iter_spectrum_records,
    schema_version,
    validate_sd_source,
)


MOLBLOCK = """example
  ChemApp

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
M  END
"""


def _tag(name: str, value: str) -> str:
    return f"> <{name}>\n{value}\n\n"


def _record(source_id: str, *, invalid: bool = False) -> str:
    measured = "not-a-peak" if invalid else "10.1;0.0S;0|20.2;1.0D;0|"
    return (
        MOLBLOCK
        + _tag("nmrshiftdb2 ID", source_id)
        + _tag("INChI key", f"KEY-{source_id}")
        + _tag("Solvent", "0:CDCl3 1:CDCl3 2:DMSO-D6")
        + _tag("Field Strength [MHz]", "0:100.0 1:125.0 2:400.0")
        + _tag("Temperature [K]", "0:298 1:300 2:310")
        + _tag("Measurement Type", "0:measured 1:calculated 2:measured")
        + _tag("Review Status", "0:reviewed 1:reviewed 2:pending")
        + _tag("Literature", "0:doi:10.1/example 2:doi:10.2/example")
        + _tag("Assignment Method", "0:manual 1:predicted 2:manual")
        + _tag(
            "Program",
            "1:ACD/Labs C+H NMR Predictor 2:JEOL Delta",
        )
        + _tag("Spectrum 13C 0", measured)
        + _tag("Spectrum 13C 1", "11.0;0.0S;0|21.0;0.0D;0|")
        + _tag("Spectrum 1H 2", "1.25;3.0t;1|")
        + _tag(
            "rawdata 1H 2",
            "https://example.test/60001234_1H.zip?spectrumid=60001234",
        )
        + "$$$$\n"
    )


def _write_sd(path: Path, *records: str) -> Path:
    path.write_text("".join(records), encoding="utf-8")
    return path


def _build(source: Path, output: Path, **kwargs: object):
    return build_nmr_index_v2(
        source,
        output,
        source_name="nmrshiftdb2",
        source_version="test-snapshot",
        source_uri="https://example.test/snapshot.sd",
        license_uri="https://example.test/license",
        min_bytes=1,
        **kwargs,
    )


def test_validate_sd_source_pins_hash_and_rejects_html(tmp_path: Path) -> None:
    source = _write_sd(tmp_path / "valid.sd", _record("42"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    validated = validate_sd_source(source, expected_sha256=digest, min_bytes=1)

    assert validated.sha256 == digest
    assert validated.byte_size == source.stat().st_size
    with pytest.raises(SourceValidationError, match="SHA-256 mismatch"):
        validate_sd_source(source, expected_sha256="0" * 64, min_bytes=1)

    html = tmp_path / "download.sd"
    html.write_text(
        "<!doctype html><html><body>SourceForge download page</body></html>",
        encoding="utf-8",
    )
    with pytest.raises(SourceValidationError, match="HTML"):
        validate_sd_source(html, min_bytes=1)


def test_failed_validation_does_not_replace_existing_index(tmp_path: Path) -> None:
    source = tmp_path / "download.sd"
    source.write_text("<html><body>not data</body></html>", encoding="utf-8")
    output = tmp_path / "index.sqlite"
    sentinel = b"existing validated index"
    output.write_bytes(sentinel)

    with pytest.raises(SourceValidationError, match="HTML"):
        _build(source, output)

    assert output.read_bytes() == sentinel
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_iter_spectrum_records_keeps_conditions_and_filters_calculated(
    tmp_path: Path,
) -> None:
    source = _write_sd(tmp_path / "spectra.sd", _record("42"))

    spectra = list(iter_spectrum_records(source))

    assert [item.spectrum_tag for item in spectra] == [
        "Spectrum 13C 0",
        "Spectrum 1H 2",
    ]
    carbon, proton = spectra
    assert carbon.measurement_kind == "measured"
    assert carbon.review_status == "reviewed"
    assert carbon.solvent == "CDCl3"
    assert carbon.field_mhz == 100.0
    assert carbon.temperature_k == 298.0
    assert carbon.literature == "doi:10.1/example"
    assert proton.source_spectrum_id == "60001234"
    assert proton.solvent == "DMSO-D6"
    assert proton.field_mhz == 400.0
    assert proton.peaks[0].multiplicity == "t"
    assert proton.peaks[0].atom_ref == 1

    all_spectra = list(iter_spectrum_records(source, measured_only=False))
    assert len(all_spectra) == 3
    calculated = next(item for item in all_spectra if item.spectrum_index == 1)
    assert calculated.measurement_kind == "calculated"
    assert "ACD/Labs" in (calculated.program or "")

    reviewed = list(iter_spectrum_records(source, require_reviewed=True))
    assert [item.spectrum_tag for item in reviewed] == ["Spectrum 13C 0"]


def test_atomic_build_preserves_duplicate_molecules_and_provenance(
    tmp_path: Path,
) -> None:
    source = _write_sd(tmp_path / "source.sd", _record("same"), _record("same"))
    output = tmp_path / "index.sqlite"
    output.write_bytes(b"old database sentinel")

    stats = _build(source, output)

    assert stats.scanned_molecules == 2
    assert stats.imported_molecules == 2
    assert stats.imported_spectra == 4
    assert stats.imported_peaks == 6
    assert stats.imported_inferred_measured == 0
    assert stats.filtered_calculated == 2
    assert stats.filtered_unknown == 0
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))

    conn = connect_readonly(output)
    assert schema_version(conn) == SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM spectra").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM peaks").fetchone()[0] == 6
    ordinals = [
        row[0]
        for row in conn.execute(
            "SELECT source_record_ordinal FROM molecules ORDER BY source_record_ordinal"
        )
    ]
    assert ordinals == [1, 2]
    identity = conn.execute(
        "SELECT smiles, formula, inchi_key FROM molecules ORDER BY id LIMIT 1"
    ).fetchone()
    assert identity["smiles"]
    assert identity["formula"]
    assert identity["inchi_key"]

    snapshot = conn.execute(
        "SELECT source_version, source_uri, sha256, license_uri, validation_json "
        "FROM source_snapshots"
    ).fetchone()
    assert snapshot["source_version"] == "test-snapshot"
    assert snapshot["source_uri"] == "https://example.test/snapshot.sd"
    assert snapshot["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert snapshot["license_uri"] == "https://example.test/license"
    assert json.loads(snapshot["validation_json"])["build_options"]["measured_only"] is True
    conn.close()

    ordinary = sqlite3.connect(output)
    raw_tags = ordinary.execute("SELECT raw_tags_json FROM molecules LIMIT 1").fetchone()[0]
    assignment = ordinary.execute(
        "SELECT assignment_json FROM peaks ORDER BY id LIMIT 1"
    ).fetchone()[0]
    ordinary.close()
    assert json.loads(raw_tags)["Review Status"].startswith("0:reviewed")
    assert json.loads(assignment) == {
        "atom_ref": 0,
        "raw_token": "10.1;0.0S;0",
    }


def test_build_quarantines_invalid_spectrum_without_partial_peaks(
    tmp_path: Path,
) -> None:
    source = _write_sd(
        tmp_path / "source.sd",
        _record("good"),
        _record("bad", invalid=True),
    )
    output = tmp_path / "index.sqlite"

    stats = _build(source, output)

    assert stats.rejected_spectra == 1
    conn = sqlite3.connect(output)
    rejection = conn.execute(
        "SELECT source_molecule_id, spectrum_tag, reason FROM import_rejections"
    ).fetchone()
    assert rejection[0:2] == ("bad", "Spectrum 13C 0")
    assert "expected three fields" in rejection[2]
    assert conn.execute(
        """
        SELECT COUNT(*)
        FROM peaks p
        JOIN spectra s ON s.id = p.spectrum_id
        JOIN molecules m ON m.id = s.molecule_id
        WHERE m.source_molecule_id = 'bad' AND s.spectrum_tag = 'Spectrum 13C 0'
        """
    ).fetchone()[0] == 0
    conn.close()


def test_require_reviewed_is_fail_closed_for_missing_review_metadata(
    tmp_path: Path,
) -> None:
    source = _write_sd(
        tmp_path / "unreviewed.sd",
        MOLBLOCK
        + _tag("nmrshiftdb2 ID", "7")
        + _tag("Spectrum 13C 0", "42.0;0.0S;0|")
        + "$$$$\n",
    )
    output = tmp_path / "index.sqlite"

    stats = _build(source, output, require_reviewed=True)

    assert stats.imported_spectra == 0
    assert stats.filtered_unknown == 1
    assert stats.filtered_review == 0


def test_allow_inferred_measured_is_explicit_and_audited(tmp_path: Path) -> None:
    source = _write_sd(
        tmp_path / "inferred.sd",
        MOLBLOCK
        + _tag("nmrshiftdb2 ID", "8")
        + _tag("Spectrum 13C 0", "42.0;0.0S;0|")
        + "$$$$\n",
    )
    output = tmp_path / "index.sqlite"

    stats = _build(source, output, allow_inferred_measured=True)

    assert stats.imported_spectra == 1
    assert stats.imported_inferred_measured == 1
    conn = sqlite3.connect(output)
    measurement_kind, metadata_json = conn.execute(
        "SELECT measurement_kind, metadata_json FROM spectra"
    ).fetchone()
    build_options = json.loads(
        conn.execute(
            "SELECT value FROM schema_metadata WHERE key = 'build_options'"
        ).fetchone()[0]
    )
    conn.close()
    assert measurement_kind == "inferred_measured"
    assert "operator allowed" in json.loads(metadata_json)["measurement_classification"]
    assert build_options["allow_inferred_measured"] is True
