"""Traceable, spectrum-level import pipeline for NMRShiftDB2 SD snapshots.

The legacy index stores one merged row per molecule.  This module deliberately
uses a different model: every ``Spectrum <nucleus> <index>`` SD tag becomes one
row and keeps its own acquisition/provenance metadata.  It never merges records
across SD entries, experimental conditions, calculations, or conformations.

The importer is intentionally independent from the production elucidation
module.  A v2 database can therefore be built and audited before it is allowed
to influence predictions.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import sqlite3
import tempfile
from collections.abc import Collection, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

SCHEMA_VERSION = "2"
DEFAULT_NUCLEI = ("1H", "13C")
DEFAULT_MIN_SD_BYTES = 1024

_SPECTRUM_TAG_RE = re.compile(r"^Spectrum\s+(.+?)\s+(\d+)$", re.IGNORECASE)
_SD_TAG_RE = re.compile(r"^>\s*<([^>]+)>")
_INDEX_MARKER_RE = re.compile(r"(?:^|\s)(\d+):")
_FLOAT_PREFIX_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_CALCULATED_WORDS = (
    "calculate",
    "calculated",
    "computation",
    "dft",
    "gaussian",
    "gamess",
    "orca",
    "predict",
    "predictor",
    "quantum espresso",
    "hose code",
    "simulat",
    "theoretical",
)
_MEASURED_WORDS = ("experimental", "measured", "observed")
_COMPUTATION_TAGS = (
    "NMRProgram",
    "NMRMethod",
    "NMRBasisSet",
    "NMRLocalis",
    "NMRStandard",
    "GeomMethod",
    "GeomBasisSet",
)
_REVIEWED_WORDS = ("accepted", "approved", "reviewed", "true", "verified", "yes")
_UNREVIEWED_WORDS = ("false", "no", "not reviewed", "pending", "unreviewed")
_REJECTED_WORDS = ("invalid", "rejected")


class SourceValidationError(ValueError):
    """Raised when an input is not a validated plain-text SD snapshot."""


class SpectrumParseError(ValueError):
    """Raised when a spectrum tag cannot be parsed without losing data."""


@dataclass(frozen=True)
class ValidatedSource:
    """Immutable identity and validation facts for one input snapshot."""

    path: Path
    byte_size: int
    sha256: str
    format: str = "MDL SD"


@dataclass(frozen=True)
class PeakRecord:
    """One source peak with its raw atom assignment preserved."""

    shift: float
    intensity: float
    multiplicity: str | None
    atom_ref: int
    raw_token: str


@dataclass(frozen=True)
class ParsedSpectrum:
    """One SD spectrum tag and the molecule record that contains it."""

    source_record_ordinal: int
    source_molecule_id: str
    source_spectrum_id: str
    nucleus: str
    spectrum_index: int
    spectrum_tag: str
    measurement_kind: str
    review_status: str
    solvent: str | None
    field_mhz: float | None
    temperature_k: float | None
    reference: str | None
    literature: str | None
    rawdata_uri: str | None
    assignment_method: str | None
    program: str | None
    method: str | None
    basis_set: str | None
    molblock: str
    name: str | None
    inchi: str | None
    inchi_key: str | None
    smiles: str | None
    formula: str | None
    record_sha256: str
    raw_tags: Mapping[str, str]
    raw_value: str
    peaks: tuple[PeakRecord, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class BuildStats:
    """Auditable counts returned after an atomic v2 rebuild."""

    output_path: str
    source_sha256: str
    scanned_molecules: int
    imported_molecules: int
    imported_spectra: int
    imported_peaks: int
    imported_inferred_measured: int
    filtered_calculated: int
    filtered_unknown: int
    filtered_review: int
    rejected_spectra: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_http_uri(value: str, label: str) -> str:
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an absolute HTTP(S) URI")
    return value


def validate_sd_source(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    min_bytes: int = DEFAULT_MIN_SD_BYTES,
) -> ValidatedSource:
    """Validate SD magic, size and optional pinned SHA-256 before parsing.

    nmrshiftdb2 download pages can be saved with a ``.sd`` suffix.  Those HTML
    responses are rejected explicitly instead of being interpreted as an empty
    dataset.
    """

    source = Path(path).expanduser().resolve()
    if min_bytes < 1:
        raise ValueError("min_bytes must be at least 1")
    if not source.is_file():
        raise SourceValidationError(f"SD source is not a regular file: {source}")

    byte_size = source.stat().st_size
    if byte_size < min_bytes:
        raise SourceValidationError(
            f"SD source is too small ({byte_size} bytes; minimum {min_bytes})"
        )

    with source.open("rb") as handle:
        head = handle.read(min(byte_size, 1024 * 1024))
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if head.startswith((b"\x1f\x8b", b"PK\x03\x04")):
        raise SourceValidationError("Compressed input must be unpacked before SD import")
    if (
        stripped.startswith((b"<!doctype html", b"<html"))
        or b"<html" in stripped[:65536]
        or (b"sourceforge.net" in stripped[:65536] and b"<body" in stripped[:65536])
    ):
        raise SourceValidationError("Input is HTML, not an SD snapshot")
    if b"\x00" in head:
        raise SourceValidationError("Input contains binary NUL bytes and is not plain-text SD")
    if b"M  END" not in head or b"$$$$" not in head:
        raise SourceValidationError("Input does not contain MDL SD record markers")

    actual_sha256 = _sha256_file(source)
    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()
        if not _SHA256_RE.fullmatch(expected):
            raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
        if not hmac.compare_digest(actual_sha256, expected):
            raise SourceValidationError(
                f"SHA-256 mismatch: expected {expected}, got {actual_sha256}"
            )

    return ValidatedSource(path=source, byte_size=byte_size, sha256=actual_sha256)


def _indexed_values(value: str) -> dict[int, str] | None:
    matches = list(_INDEX_MARKER_RE.finditer(value))
    if not matches:
        return None
    result: dict[int, str] = {}
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(value)
        result[int(match.group(1))] = value[match.end() : end].strip()
    return result


def _value_for_index(value: str | None, index: int) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().rstrip("\\").strip()
    if not cleaned:
        return None
    indexed = _indexed_values(cleaned)
    if indexed is None:
        return cleaned
    selected = indexed.get(index, "").strip()
    return selected or None


def _first_tag(tags: Mapping[str, str], *names: str) -> str | None:
    folded = {key.casefold(): value for key, value in tags.items()}
    for name in names:
        value = folded.get(name.casefold())
        if value is not None and value.strip():
            return value.strip().rstrip("\\").strip() or None
    return None


def _first_indexed_tag(tags: Mapping[str, str], index: int, *names: str) -> str | None:
    for name in names:
        value = _first_tag(tags, name)
        selected = _value_for_index(value, index)
        if selected is not None:
            return selected
    return None


def _optional_float(value: str | None) -> float | None:
    if not value:
        return None
    match = _FLOAT_PREFIX_RE.search(value)
    if match is None:
        return None
    parsed = float(match.group(0))
    return parsed if math.isfinite(parsed) else None


def _parse_peak_token(token: str) -> PeakRecord:
    raw = token.strip()
    parts = raw.split(";")
    if len(parts) != 3:
        raise SpectrumParseError(f"invalid signal token {raw!r}: expected three fields")
    try:
        shift = float(parts[0].strip())
        atom_ref = int(parts[2].strip())
    except ValueError as exc:
        raise SpectrumParseError(f"invalid shift or atom reference in {raw!r}") from exc
    if not math.isfinite(shift) or atom_ref < 0:
        raise SpectrumParseError(f"non-finite shift or negative atom reference in {raw!r}")

    intensity_and_multiplicity = parts[1].strip()
    match = _FLOAT_PREFIX_RE.match(intensity_and_multiplicity)
    if match is None:
        raise SpectrumParseError(f"invalid intensity in signal token {raw!r}")
    intensity = float(match.group(0))
    if not math.isfinite(intensity):
        raise SpectrumParseError(f"non-finite intensity in signal token {raw!r}")
    multiplicity = intensity_and_multiplicity[match.end() :].strip() or None
    return PeakRecord(
        shift=shift,
        intensity=intensity,
        multiplicity=multiplicity,
        atom_ref=atom_ref,
        raw_token=raw,
    )


def _parse_peaks(value: str) -> tuple[PeakRecord, ...]:
    tokens = [token for token in value.split("|") if token.strip()]
    if not tokens:
        raise SpectrumParseError("spectrum contains no peak tokens")
    return tuple(_parse_peak_token(token) for token in tokens)


def _meaningful(value: str | None) -> bool:
    return bool(value and value.strip().casefold() not in {"n/a", "none", "unknown", "unreported"})


def _classify_measurement(
    tags: Mapping[str, str],
    index: int,
    *,
    program: str | None,
    rawdata_uri: str | None,
    allow_inferred_measured: bool,
) -> tuple[str, str]:
    explicit = _first_indexed_tag(
        tags,
        index,
        "Measurement Type",
        "Spectrum Type",
        "Data Type",
        "Origin",
    )
    explicit_folded = (explicit or "").casefold()
    if any(word in explicit_folded for word in _CALCULATED_WORDS):
        return "calculated", "explicit source metadata"
    if any(word in explicit_folded for word in _MEASURED_WORDS):
        return "measured", "explicit source metadata"

    computation_values = [
        _first_indexed_tag(tags, index, tag_name) for tag_name in _COMPUTATION_TAGS
    ]
    program_folded = (program or "").casefold()
    if any(word in program_folded for word in _CALCULATED_WORDS) or any(
        _meaningful(value) for value in computation_values
    ):
        return "calculated", "calculation program/method metadata"
    if rawdata_uri:
        return "measured", "source raw-data link"
    if allow_inferred_measured:
        return (
            "inferred_measured",
            "operator allowed NMRShiftDB2 inference from absence of calculation metadata",
        )
    return "unknown", "source has no explicit measured-or-calculated evidence"


def _classify_review(tags: Mapping[str, str], index: int) -> tuple[str, str | None]:
    for key, raw_value in tags.items():
        folded_key = key.casefold()
        if "review" not in folded_key and "validat" not in folded_key:
            continue
        value = _value_for_index(raw_value, index)
        if not value:
            continue
        folded_value = value.casefold()
        if any(word in folded_value for word in _REJECTED_WORDS):
            return "rejected", value
        if any(word in folded_value for word in _UNREVIEWED_WORDS):
            return "unreviewed", value
        if any(word in folded_value for word in _REVIEWED_WORDS):
            return "reviewed", value
        return "unknown", value
    return "unknown", None


def _extract_rawdata_spectrum_id(rawdata_uri: str | None) -> str | None:
    if not rawdata_uri:
        return None
    parsed = urlparse(rawdata_uri)
    query_value = parse_qs(parsed.query).get("spectrumid")
    if query_value and query_value[0].strip():
        return query_value[0].strip()
    match = re.search(r"/(\d+)(?:_[^/]*)?(?:\.zip)?$", parsed.path)
    return match.group(1) if match else None


def _nmredata_db_id(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"(?:^|[\r\n\\])DB_ID=([^\\\r\n]+)", value)
    return match.group(1).strip() if match else None


def _record_hash(molblock: str, tags: Mapping[str, str]) -> str:
    canonical = json.dumps(
        {"molblock": molblock, "tags": dict(tags)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _derive_structure(
    molblock: str,
    tags: Mapping[str, str],
) -> tuple[str | None, str | None, str | None, str | None, dict[str, Any]]:
    """Return InChI, InChIKey, canonical SMILES and formula from the molblock."""

    raw_inchi = _first_tag(tags, "INChI", "NMREDATA_INCHI")
    raw_inchi_key = _first_tag(tags, "INChI key", "InChIKey", "INCHI_KEY")
    raw_smiles = _first_tag(tags, "SMILES", "NMREDATA_SMILES")
    raw_formula = _first_tag(tags, "Formula", "Molecular Formula")
    structure_metadata: dict[str, Any] = {
        "structure_source": "molblock",
        "source_had_inchi": raw_inchi is not None,
        "source_had_inchi_key": raw_inchi_key is not None,
        "source_had_smiles": raw_smiles is not None,
        "source_had_formula": raw_formula is not None,
    }
    molecule = Chem.MolFromMolBlock(
        molblock,
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    if molecule is None:
        structure_metadata["structure_parse_status"] = "invalid_molblock"
        return raw_inchi, raw_inchi_key, raw_smiles, raw_formula, structure_metadata

    heavy_molecule = Chem.RemoveHs(molecule)
    canonical_smiles = Chem.MolToSmiles(
        heavy_molecule,
        canonical=True,
        isomericSmiles=True,
    )
    derived_formula = rdMolDescriptors.CalcMolFormula(heavy_molecule)
    derived_inchi = Chem.MolToInchi(heavy_molecule)
    derived_inchi_key = Chem.MolToInchiKey(heavy_molecule)
    structure_metadata.update(
        {
            "structure_parse_status": "ok",
            "source_inchi_key_mismatch": bool(
                raw_inchi_key
                and derived_inchi_key
                and raw_inchi_key.strip() != derived_inchi_key
            ),
            "source_smiles_mismatch": bool(
                raw_smiles
                and canonical_smiles
                and raw_smiles.strip() != canonical_smiles
            ),
            "source_formula_mismatch": bool(
                raw_formula
                and derived_formula
                and raw_formula.replace(" ", "") != derived_formula.replace(" ", "")
            ),
        }
    )
    return (
        derived_inchi or raw_inchi,
        derived_inchi_key or raw_inchi_key,
        canonical_smiles or raw_smiles,
        derived_formula or raw_formula,
        structure_metadata,
    )


def _build_sd_entry(lines: list[str]) -> dict[str, Any]:
    """Parse molblock and arbitrary SD tags without interpreting peak fields."""

    mol_end = next(
        (position for position, line in enumerate(lines) if line.strip() == "M  END"),
        None,
    )
    if mol_end is None:
        return {"molblock": "\n".join(lines), "tags": {}}

    molblock = "\n".join(lines[: mol_end + 1])
    tag_lines = lines[mol_end + 1 :]
    tags: dict[str, str] = {}
    position = 0
    while position < len(tag_lines):
        match = _SD_TAG_RE.match(tag_lines[position].strip())
        if match is None:
            position += 1
            continue
        tag_name = match.group(1).strip()
        position += 1
        value_lines: list[str] = []
        while position < len(tag_lines):
            if _SD_TAG_RE.match(tag_lines[position].strip()) is not None:
                break
            value_lines.append(tag_lines[position])
            position += 1
        tags[tag_name] = "\n".join(value_lines).strip()
    return {"molblock": molblock, "tags": tags}


def _stream_sd(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield raw SD entries without eagerly parsing any spectrum tag."""

    buffer: list[str] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.strip() == "$$$$":
                if buffer:
                    yield _build_sd_entry(buffer)
                    buffer = []
                continue
            buffer.append(line)
    if buffer:
        yield _build_sd_entry(buffer)


def _join_metadata(tags: Mapping[str, str], index: int, names: tuple[str, ...]) -> str | None:
    values: list[str] = []
    for name in names:
        value = _first_indexed_tag(tags, index, name)
        if _meaningful(value) and value not in values:
            values.append(value)
    return "; ".join(values) or None


def _entry_identity(tags: Mapping[str, str], ordinal: int) -> str:
    return (
        _first_tag(tags, "nmrshiftdb2 ID", "DB_ID", "ID")
        or _nmredata_db_id(_first_tag(tags, "NMREDATA_ID"))
        or f"record-{ordinal}"
    )


def _entry_spectra(
    entry: Mapping[str, Any],
    ordinal: int,
    *,
    nuclei: Collection[str] | None,
    allow_inferred_measured: bool = False,
) -> tuple[list[ParsedSpectrum], list[tuple[str, str, str]]]:
    tags = entry.get("tags")
    if not isinstance(tags, Mapping):
        return [], [("", "", "entry has no tag mapping")]
    tags = {str(key): str(value) for key, value in tags.items()}
    molblock = str(entry.get("molblock", ""))
    source_molecule_id = _entry_identity(tags, ordinal)
    record_sha256 = _record_hash(molblock, tags)
    inchi, inchi_key, smiles, formula, structure_metadata = _derive_structure(
        molblock, tags
    )
    allowed_nuclei = {item.casefold() for item in nuclei} if nuclei is not None else None
    parsed_records: list[ParsedSpectrum] = []
    rejections: list[tuple[str, str, str]] = []

    descriptors: list[tuple[int, str, str, str]] = []
    for tag_name, raw_value in tags.items():
        match = _SPECTRUM_TAG_RE.fullmatch(tag_name.strip())
        if match is None:
            continue
        nucleus = match.group(1).strip()
        if allowed_nuclei is not None and nucleus.casefold() not in allowed_nuclei:
            continue
        descriptors.append((int(match.group(2)), nucleus, tag_name, raw_value))

    for spectrum_index, nucleus, spectrum_tag, raw_value in sorted(
        descriptors, key=lambda row: (row[0], row[1].casefold(), row[2])
    ):
        rawdata_uri = _first_tag(tags, f"rawdata {nucleus} {spectrum_index}")
        program = _first_indexed_tag(tags, spectrum_index, "Program", "NMRProgram")
        measurement_kind, measurement_evidence = _classify_measurement(
            tags,
            spectrum_index,
            program=program,
            rawdata_uri=rawdata_uri,
            allow_inferred_measured=allow_inferred_measured,
        )
        review_status, review_raw = _classify_review(tags, spectrum_index)
        try:
            peaks = _parse_peaks(raw_value)
        except SpectrumParseError as exc:
            rejections.append((spectrum_tag, raw_value, str(exc)))
            continue

        source_spectrum_id = (
            _extract_rawdata_spectrum_id(rawdata_uri)
            or _first_indexed_tag(tags, spectrum_index, "Spectrum ID", "NMR Spectrum ID")
            or f"{source_molecule_id}:{nucleus}:{spectrum_index}"
        )
        method = _join_metadata(
            tags, spectrum_index, ("NMRMethod", "GeomMethod", "Assignment Method")
        )
        basis_set = _join_metadata(
            tags, spectrum_index, ("NMRBasisSet", "GeomBasisSet")
        )
        metadata = {
            "measurement_classification": measurement_evidence,
            "review_source_value": review_raw,
            "source_record_ordinal": ordinal,
            "atom_assignment_count": sum(peak.atom_ref >= 0 for peak in peaks),
            **structure_metadata,
        }
        parsed_records.append(
            ParsedSpectrum(
                source_record_ordinal=ordinal,
                source_molecule_id=source_molecule_id,
                source_spectrum_id=source_spectrum_id,
                nucleus=nucleus,
                spectrum_index=spectrum_index,
                spectrum_tag=spectrum_tag,
                measurement_kind=measurement_kind,
                review_status=review_status,
                solvent=_first_indexed_tag(tags, spectrum_index, "Solvent", "NMREDATA_SOLVENT"),
                field_mhz=_optional_float(
                    _first_indexed_tag(
                        tags,
                        spectrum_index,
                        "Field Strength [MHz]",
                        "Larmor",
                    )
                ),
                temperature_k=_optional_float(
                    _first_indexed_tag(
                        tags,
                        spectrum_index,
                        "Temperature [K]",
                        "NMREDATA_TEMPERATURE",
                    )
                ),
                reference=_first_indexed_tag(
                    tags,
                    spectrum_index,
                    "Reference",
                    "NMRReference",
                    "NMRStandard",
                    "Standard",
                ),
                literature=_first_indexed_tag(
                    tags,
                    spectrum_index,
                    "Literature",
                    "AUTHOR_LITERATURE",
                    "Citation",
                    "DOI",
                ),
                rawdata_uri=rawdata_uri,
                assignment_method=_first_indexed_tag(
                    tags, spectrum_index, "Assignment Method"
                ),
                program=program,
                method=method,
                basis_set=basis_set,
                molblock=molblock,
                name=_first_tag(tags, "Name", "CHEMNAME", "NAME"),
                inchi=inchi,
                inchi_key=inchi_key,
                smiles=smiles,
                formula=formula,
                record_sha256=record_sha256,
                raw_tags=tags,
                raw_value=raw_value,
                peaks=peaks,
                metadata=metadata,
            )
        )
    return parsed_records, rejections


def iter_spectrum_records(
    path: str | Path,
    *,
    measured_only: bool = True,
    require_reviewed: bool = False,
    nuclei: Collection[str] | None = DEFAULT_NUCLEI,
    strict: bool = True,
    allow_inferred_measured: bool = False,
) -> Iterator[ParsedSpectrum]:
    """Yield independent, filtered spectrum records from a validated SD file.

    Call :func:`validate_sd_source` before this low-level iterator when accepting
    user-selected files.  The atomic builder always performs that validation.
    """

    for ordinal, entry in enumerate(_stream_sd(path), start=1):
        spectra, rejections = _entry_spectra(
            entry,
            ordinal,
            nuclei=nuclei,
            allow_inferred_measured=allow_inferred_measured,
        )
        if strict and rejections:
            spectrum_tag, _raw_value, reason = rejections[0]
            raise SpectrumParseError(f"record {ordinal}, {spectrum_tag}: {reason}")
        for spectrum in spectra:
            accepted_measurement = spectrum.measurement_kind == "measured" or (
                allow_inferred_measured
                and spectrum.measurement_kind == "inferred_measured"
            )
            if measured_only and not accepted_measurement:
                continue
            if require_reviewed and spectrum.review_status != "reviewed":
                continue
            yield spectrum


_SCHEMA_SQL = """
CREATE TABLE schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE source_snapshots (
    id INTEGER PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_version TEXT,
    source_uri TEXT NOT NULL,
    local_filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    byte_size INTEGER NOT NULL CHECK (byte_size > 0),
    license_uri TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    validation_json TEXT NOT NULL
);

CREATE TABLE molecules (
    id INTEGER PRIMARY KEY,
    snapshot_id INTEGER NOT NULL REFERENCES source_snapshots(id),
    source_record_ordinal INTEGER NOT NULL,
    source_molecule_id TEXT NOT NULL,
    record_sha256 TEXT NOT NULL,
    name TEXT,
    molblock TEXT NOT NULL,
    inchi TEXT,
    inchi_key TEXT,
    smiles TEXT,
    formula TEXT,
    raw_tags_json TEXT NOT NULL,
    UNIQUE (snapshot_id, source_record_ordinal)
);

CREATE TABLE spectra (
    id INTEGER PRIMARY KEY,
    molecule_id INTEGER NOT NULL REFERENCES molecules(id) ON DELETE CASCADE,
    source_spectrum_id TEXT NOT NULL,
    nucleus TEXT NOT NULL,
    spectrum_index INTEGER NOT NULL,
    spectrum_tag TEXT NOT NULL,
    measurement_kind TEXT NOT NULL
        CHECK (
            measurement_kind IN (
                'measured', 'inferred_measured', 'calculated', 'unknown'
            )
        ),
    review_status TEXT NOT NULL
        CHECK (review_status IN ('reviewed', 'unreviewed', 'rejected', 'unknown')),
    solvent TEXT,
    field_mhz REAL,
    temperature_k REAL,
    reference TEXT,
    literature TEXT,
    rawdata_uri TEXT,
    assignment_method TEXT,
    program TEXT,
    method TEXT,
    basis_set TEXT,
    metadata_json TEXT NOT NULL,
    raw_value TEXT NOT NULL,
    UNIQUE (molecule_id, spectrum_tag)
);

CREATE TABLE peaks (
    id INTEGER PRIMARY KEY,
    spectrum_id INTEGER NOT NULL REFERENCES spectra(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    shift REAL NOT NULL,
    intensity REAL NOT NULL,
    multiplicity TEXT,
    atom_ref INTEGER NOT NULL CHECK (atom_ref >= 0),
    assignment_json TEXT NOT NULL,
    UNIQUE (spectrum_id, ordinal)
);

CREATE TABLE import_rejections (
    id INTEGER PRIMARY KEY,
    source_record_ordinal INTEGER NOT NULL,
    source_molecule_id TEXT NOT NULL,
    spectrum_tag TEXT,
    reason TEXT NOT NULL,
    raw_value TEXT NOT NULL
);

CREATE INDEX idx_molecules_source_id ON molecules(source_molecule_id);
CREATE INDEX idx_molecules_inchi_key ON molecules(inchi_key);
CREATE INDEX idx_spectra_source_id ON spectra(source_spectrum_id);
CREATE INDEX idx_spectra_nucleus_kind ON spectra(nucleus, measurement_kind, review_status);
CREATE INDEX idx_peaks_spectrum_shift ON peaks(spectrum_id, shift);
"""


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    conn.execute(
        "INSERT INTO schema_metadata(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )


def _insert_molecule(
    conn: sqlite3.Connection,
    snapshot_id: int,
    spectrum: ParsedSpectrum,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO molecules (
            snapshot_id, source_record_ordinal, source_molecule_id, record_sha256,
            name, molblock, inchi, inchi_key, smiles, formula, raw_tags_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot_id,
            spectrum.source_record_ordinal,
            spectrum.source_molecule_id,
            spectrum.record_sha256,
            spectrum.name,
            spectrum.molblock,
            spectrum.inchi,
            spectrum.inchi_key,
            spectrum.smiles,
            spectrum.formula,
            json.dumps(dict(spectrum.raw_tags), sort_keys=True, ensure_ascii=False),
        ),
    )
    return int(cursor.lastrowid)


def _insert_spectrum(
    conn: sqlite3.Connection,
    molecule_id: int,
    spectrum: ParsedSpectrum,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO spectra (
            molecule_id, source_spectrum_id, nucleus, spectrum_index, spectrum_tag,
            measurement_kind, review_status, solvent, field_mhz, temperature_k,
            reference, literature, rawdata_uri, assignment_method, program, method,
            basis_set, metadata_json, raw_value
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            molecule_id,
            spectrum.source_spectrum_id,
            spectrum.nucleus,
            spectrum.spectrum_index,
            spectrum.spectrum_tag,
            spectrum.measurement_kind,
            spectrum.review_status,
            spectrum.solvent,
            spectrum.field_mhz,
            spectrum.temperature_k,
            spectrum.reference,
            spectrum.literature,
            spectrum.rawdata_uri,
            spectrum.assignment_method,
            spectrum.program,
            spectrum.method,
            spectrum.basis_set,
            json.dumps(dict(spectrum.metadata), sort_keys=True, ensure_ascii=False),
            spectrum.raw_value,
        ),
    )
    return int(cursor.lastrowid)


def _insert_peaks(
    conn: sqlite3.Connection,
    spectrum_id: int,
    peaks: tuple[PeakRecord, ...],
) -> None:
    conn.executemany(
        """
        INSERT INTO peaks (
            spectrum_id, ordinal, shift, intensity, multiplicity, atom_ref,
            assignment_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                spectrum_id,
                ordinal,
                peak.shift,
                peak.intensity,
                peak.multiplicity,
                peak.atom_ref,
                json.dumps(
                    {"atom_ref": peak.atom_ref, "raw_token": peak.raw_token},
                    sort_keys=True,
                ),
            )
            for ordinal, peak in enumerate(peaks)
        ),
    )


def build_nmr_index_v2(
    source_path: str | Path,
    output_path: str | Path,
    *,
    source_name: str,
    source_uri: str,
    license_uri: str,
    source_version: str | None = None,
    expected_sha256: str | None = None,
    measured_only: bool = True,
    require_reviewed: bool = False,
    nuclei: Collection[str] | None = DEFAULT_NUCLEI,
    min_bytes: int = DEFAULT_MIN_SD_BYTES,
    limit_molecules: int | None = None,
    allow_inferred_measured: bool = False,
) -> BuildStats:
    """Validate a snapshot and atomically replace ``output_path`` with a v2 DB."""

    if not source_name.strip():
        raise ValueError("source_name must not be empty")
    source_uri = _require_http_uri(source_uri, "source_uri")
    license_uri = _require_http_uri(license_uri, "license_uri")
    if limit_molecules is not None and limit_molecules < 1:
        raise ValueError("limit_molecules must be at least 1")

    validated = validate_sd_source(
        source_path,
        expected_sha256=expected_sha256,
        min_bytes=min_bytes,
    )
    output = Path(output_path).expanduser().resolve()
    if output == validated.path:
        raise ValueError("output_path must not overwrite the source SD file")
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()

    scanned_molecules = 0
    imported_molecules = 0
    imported_spectra = 0
    imported_peaks = 0
    imported_inferred_measured = 0
    filtered_calculated = 0
    filtered_unknown = 0
    filtered_review = 0
    rejected_spectra = 0
    conn: sqlite3.Connection | None = None

    try:
        conn = sqlite3.connect(temporary_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("PRAGMA synchronous = FULL")
        _create_schema(conn)
        imported_at = datetime.now(timezone.utc).isoformat()
        build_options = {
            "measured_only": measured_only,
            "require_reviewed": require_reviewed,
            "allow_inferred_measured": allow_inferred_measured,
            "nuclei": sorted(nuclei) if nuclei is not None else None,
            "limit_molecules": limit_molecules,
        }
        validation = {
            "format": validated.format,
            "html_rejected": True,
            "magic_checked": ["M  END", "$$$$"],
            "expected_sha256": expected_sha256.lower() if expected_sha256 else None,
            "rehash_before_publish": True,
            "build_options": build_options,
        }
        cursor = conn.execute(
            """
            INSERT INTO source_snapshots (
                source_name, source_version, source_uri, local_filename, sha256,
                byte_size, license_uri, imported_at, validation_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_name.strip(),
                source_version.strip() if source_version else None,
                source_uri,
                validated.path.name,
                validated.sha256,
                validated.byte_size,
                license_uri,
                imported_at,
                json.dumps(validation, sort_keys=True),
            ),
        )
        snapshot_id = int(cursor.lastrowid)
        conn.execute(
            "INSERT INTO schema_metadata(key, value) VALUES ('build_options', ?)",
            (json.dumps(build_options, sort_keys=True),),
        )

        for ordinal, entry in enumerate(_stream_sd(validated.path), start=1):
            if limit_molecules is not None and ordinal > limit_molecules:
                break
            scanned_molecules += 1
            spectra, rejections = _entry_spectra(
                entry,
                ordinal,
                nuclei=nuclei,
                allow_inferred_measured=allow_inferred_measured,
            )
            tags = entry.get("tags", {})
            source_molecule_id = (
                _entry_identity(tags, ordinal)
                if isinstance(tags, Mapping)
                else f"record-{ordinal}"
            )
            for spectrum_tag, raw_value, reason in rejections:
                conn.execute(
                    """
                    INSERT INTO import_rejections (
                        source_record_ordinal, source_molecule_id, spectrum_tag,
                        reason, raw_value
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (ordinal, source_molecule_id, spectrum_tag, reason, raw_value),
                )
                rejected_spectra += 1

            accepted: list[ParsedSpectrum] = []
            for spectrum in spectra:
                accepted_measurement = spectrum.measurement_kind == "measured" or (
                    allow_inferred_measured
                    and spectrum.measurement_kind == "inferred_measured"
                )
                if measured_only and not accepted_measurement:
                    if spectrum.measurement_kind == "calculated":
                        filtered_calculated += 1
                    else:
                        filtered_unknown += 1
                    continue
                if require_reviewed and spectrum.review_status != "reviewed":
                    filtered_review += 1
                    continue
                accepted.append(spectrum)
            if not accepted:
                continue

            molecule_id = _insert_molecule(conn, snapshot_id, accepted[0])
            imported_molecules += 1
            for spectrum in accepted:
                spectrum_id = _insert_spectrum(conn, molecule_id, spectrum)
                _insert_peaks(conn, spectrum_id, spectrum.peaks)
                imported_spectra += 1
                imported_peaks += len(spectrum.peaks)
                imported_inferred_measured += int(
                    spectrum.measurement_kind == "inferred_measured"
                )

        conn.commit()
        foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise RuntimeError(f"v2 index has foreign-key violations: {foreign_key_errors[:3]}")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise RuntimeError(f"v2 index integrity check failed: {integrity}")
        conn.close()
        conn = None

        final_size = validated.path.stat().st_size
        final_sha256 = _sha256_file(validated.path)
        if final_size != validated.byte_size or not hmac.compare_digest(
            final_sha256, validated.sha256
        ):
            raise SourceValidationError("SD source changed while the v2 index was being built")

        with temporary_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    except Exception:
        if conn is not None:
            conn.close()
        temporary_path.unlink(missing_ok=True)
        raise

    return BuildStats(
        output_path=str(output),
        source_sha256=validated.sha256,
        scanned_molecules=scanned_molecules,
        imported_molecules=imported_molecules,
        imported_spectra=imported_spectra,
        imported_peaks=imported_peaks,
        imported_inferred_measured=imported_inferred_measured,
        filtered_calculated=filtered_calculated,
        filtered_unknown=filtered_unknown,
        filtered_review=filtered_review,
        rejected_spectra=rejected_spectra,
    )


def connect_readonly(path: str | Path) -> sqlite3.Connection:
    """Open a v2 SQLite index in read-only mode with named columns."""

    resolved = Path(path).expanduser().resolve()
    uri_path = quote(resolved.as_posix(), safe="/:")
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def schema_version(conn: sqlite3.Connection) -> str:
    """Return and validate the database schema version."""

    row = conn.execute(
        "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        raise ValueError("database does not declare a schema version")
    value = str(row[0])
    if value != SCHEMA_VERSION:
        raise ValueError(f"unsupported NMR index schema version: {value}")
    return value
