from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.DataStructs import TanimotoSimilarity
except Exception:  # pragma: no cover - handled at runtime
    Chem = None
    Descriptors = None
    rdFingerprintGenerator = None
    rdMolDescriptors = None
    MurckoScaffold = None
    TanimotoSimilarity = None

from app.ml.graphdiff.sd_parser import stream_sd
from app.ml.nmr_evidence import FormulaError, canonical_formula, parse_formula


@dataclass
class QueryPeak:
    shift: float
    intensity: float = 1.0
    multiplicity: str = ""
    integral: float | None = None


def _default_external_dir() -> Path | None:
    configured = os.environ.get("CHEMAPP_NMR_DATA_DIR")
    if configured:
        p = Path(configured)
        if p.exists():
            return p
    repo_data = Path(__file__).resolve().parents[2] / "data"
    return repo_data if repo_data.exists() else None


def _index_path() -> Path:
    return Path(os.environ.get("CHEMAPP_NMR_INDEX", str(Path("data") / "nmr_spectral_index.sqlite")))


def _model_path() -> Path:
    return Path(os.environ.get("CHEMAPP_NMR_RANKER", str(Path("data") / "nmr_joint_ranker.joblib")))


def _connect() -> sqlite3.Connection:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nmr_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            source_id TEXT DEFAULT '',
            name TEXT DEFAULT '',
            smiles TEXT NOT NULL,
            inchikey TEXT DEFAULT '',
            formula TEXT DEFAULT '',
            mw REAL,
            peaks_13c TEXT DEFAULT '[]',
            peaks_1h TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            UNIQUE(source, source_id, smiles)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nmr_records_source ON nmr_records(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nmr_records_inchikey ON nmr_records(inchikey)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nmr_records_formula ON nmr_records(formula)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nmr_import_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            source_path TEXT NOT NULL,
            scanned INTEGER DEFAULT 0,
            imported INTEGER DEFAULT 0,
            merged INTEGER DEFAULT 0,
            skipped INTEGER DEFAULT 0,
            status TEXT DEFAULT 'running',
            created_at TEXT DEFAULT (datetime('now')),
            finished_at TEXT
        )
    """)
    conn.commit()
    return conn


# Proton multiplets arrive as several line-level peaks, whereas decoupled 13C
# signals are already individual resonances.  For 13C, collapse only numerical
# duplicates rather than chemically distinct close signals.
_RESONANCE_GROUP_TOLERANCES = {"1H": 0.035, "13C": 0.0001}
_MATCH_TOLERANCES = {"1H": 0.18, "13C": 3.0}
# Heuristic score composition (recall / reference coverage / count balance /
# exponential error term).  Module-level so calibration experiments can sweep
# them; production values were validated by the perturbation harness in
# backend/reports/ml_eval/ (clean top-1 must stay 1.0 on the paired baseline).
_SCORE_WEIGHTS = {
    "recall": 0.40,
    "reference_coverage": 0.20,
    "count_balance": 0.15,
    "error_score": 0.25,
}


def _peak_values(peaks: list[Any]) -> list[float]:
    values = []
    for peak in peaks:
        if isinstance(peak, QueryPeak):
            value = float(peak.shift)
            if np.isfinite(value):
                values.append(value)
        elif isinstance(peak, (int, float)) and np.isfinite(float(peak)):
            values.append(float(peak))
        elif isinstance(peak, dict):
            raw = peak.get("shift", peak.get("position", peak.get("center_ppm")))
            if raw is not None:
                try:
                    value = float(raw)
                    if np.isfinite(value):
                        values.append(value)
                except (TypeError, ValueError):
                    pass
    return sorted(values)


def _normalise_resonance_shifts(peaks: list[Any], nucleus: str) -> list[float]:
    """Collapse line-level or duplicated peaks into comparable resonances.

    Both query and reference spectra pass through this function.  The 1H
    tolerance intentionally matches query multiplet preprocessing, while the
    much tighter 13C tolerance only removes effectively duplicated signals.
    """

    if nucleus not in _RESONANCE_GROUP_TOLERANCES:
        raise ValueError(f"Unsupported NMR nucleus: {nucleus}")
    weighted_values: list[tuple[float, float]] = []
    for peak in peaks:
        if isinstance(peak, QueryPeak):
            raw_shift = peak.shift
            raw_intensity = peak.intensity
        elif isinstance(peak, (int, float)):
            raw_shift = peak
            raw_intensity = 1.0
        elif isinstance(peak, dict):
            raw_shift = peak.get("shift", peak.get("position", peak.get("center_ppm")))
            raw_intensity = peak.get("intensity", peak.get("height", 1.0))
        else:
            continue
        try:
            shift = float(raw_shift)
            intensity = float(raw_intensity or 0.0)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(shift):
            continue
        if not np.isfinite(intensity):
            intensity = 1.0
        weighted_values.append((shift, max(abs(intensity), 1e-12)))
    if not weighted_values:
        return []

    tolerance = _RESONANCE_GROUP_TOLERANCES[nucleus]
    groups: list[list[tuple[float, float]]] = []
    for shift, weight in sorted(weighted_values):
        if not groups:
            groups.append([(shift, weight)])
            continue
        current = groups[-1]
        current_weight = sum(item[1] for item in current)
        current_center = sum(item[0] * item[1] for item in current) / current_weight
        if abs(shift - current_center) <= tolerance:
            current.append((shift, weight))
        else:
            groups.append([(shift, weight)])

    return [
        round(
            sum(shift * weight for shift, weight in group)
            / sum(weight for _, weight in group),
            6,
        )
        for group in groups
    ]


def _query_peaks(raw: list[dict[str, Any]] | None) -> list[QueryPeak]:
    peaks = []
    for item in raw or []:
        try:
            peaks.append(QueryPeak(
                shift=float(item.get("shift", item.get("position", item.get("center_ppm")))),
                intensity=float(item.get("intensity", item.get("height", 1)) or 1),
                multiplicity=str(item.get("multiplicity", item.get("type", "")) or ""),
                integral=float(item["integral"]) if item.get("integral") is not None else (
                    float(item["relative_area"]) if item.get("relative_area") is not None else None
                ),
            ))
        except (TypeError, ValueError):
            continue
    return peaks


def _mol_info_from_smiles(smiles: str) -> dict[str, Any]:
    if Chem is None:
        return {"canonical_smiles": smiles, "formula": "", "mw": None, "inchikey": ""}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"canonical_smiles": smiles, "formula": "", "mw": None, "inchikey": ""}
    return {
        "canonical_smiles": Chem.MolToSmiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": float(Descriptors.MolWt(mol)),
        "inchikey": Chem.MolToInchiKey(mol) if hasattr(Chem, "MolToInchiKey") else "",
    }


def _formula_key(formula: str | None) -> str:
    try:
        return canonical_formula(formula)
    except FormulaError:
        # Imported third-party records may contain adduct notation.  They remain
        # searchable as metadata but never satisfy a strict user constraint.
        return ""


def _first_block_key(inchikey: str | None) -> str:
    value = inchikey or ""
    return value.split("-")[0] if value else ""


def _merge_peaks(existing_json: str, new_peaks: list[Any], tolerance: float) -> list[Any]:
    try:
        existing = json.loads(existing_json or "[]")
    except json.JSONDecodeError:
        existing = []
    if not existing:
        return new_peaks
    merged = list(existing)
    for peak in new_peaks:
        value = peak if isinstance(peak, (int, float)) else peak.get("shift")
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        found = False
        for old in merged:
            old_value = old if isinstance(old, (int, float)) else old.get("shift")
            try:
                if abs(float(old_value) - value_f) <= tolerance:
                    found = True
                    break
            except (TypeError, ValueError):
                continue
        if not found:
            merged.append(peak)
    return merged


def _upsert_record(
    conn: sqlite3.Connection,
    *,
    source: str,
    source_id: str,
    name: str,
    smiles: str,
    formula: str,
    mw: float | None,
    inchikey: str,
    peaks_13c: list[Any],
    peaks_1h: list[Any],
    metadata: dict[str, Any],
) -> str:
    formula = _formula_key(formula)
    block = _first_block_key(inchikey)
    if block:
        row = conn.execute(
            "SELECT id, peaks_13c, peaks_1h, metadata, source FROM nmr_records WHERE substr(inchikey,1,14)=? LIMIT 1",
            (block,),
        ).fetchone()
    else:
        row = None
    if row is None:
        row = conn.execute(
            "SELECT id, peaks_13c, peaks_1h, metadata, source FROM nmr_records WHERE smiles=? LIMIT 1",
            (smiles,),
        ).fetchone()
    if row is not None:
        old_meta = json.loads(row[3] or "{}")
        sources = sorted(set(old_meta.get("merged_sources", []) + [row[4], source]))
        old_meta.update(metadata)
        old_meta["merged_sources"] = sources
        old_meta["merged_source_ids"] = sorted(set(old_meta.get("merged_source_ids", []) + [source_id]))
        conn.execute(
            """
            UPDATE nmr_records
            SET name=COALESCE(NULLIF(name,''), ?),
                formula=COALESCE(NULLIF(formula,''), ?),
                mw=COALESCE(mw, ?),
                peaks_13c=?,
                peaks_1h=?,
                metadata=?
            WHERE id=?
            """,
            (
                name,
                formula,
                mw,
                json.dumps(_merge_peaks(row[1], peaks_13c, 0.08)),
                json.dumps(_merge_peaks(row[2], peaks_1h, 0.01)),
                json.dumps(old_meta),
                row[0],
            ),
        )
        return "merged"
    conn.execute(
        """
        INSERT OR IGNORE INTO nmr_records
        (source, source_id, name, smiles, inchikey, formula, mw, peaks_13c, peaks_1h, metadata)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            source,
            source_id,
            name,
            smiles,
            inchikey,
            formula,
            mw,
            json.dumps(peaks_13c),
            json.dumps(peaks_1h),
            json.dumps(metadata),
        ),
    )
    return "inserted" if conn.total_changes else "skipped"


def _start_import_run(conn: sqlite3.Connection, source: str, source_path: Path) -> int:
    cursor = conn.execute(
        "INSERT INTO nmr_import_runs (source, source_path) VALUES (?,?)",
        (source, str(source_path)),
    )
    conn.commit()
    return int(cursor.lastrowid)


def _finish_import_run(conn: sqlite3.Connection, run_id: int, result: dict[str, Any]) -> None:
    conn.execute(
        """
        UPDATE nmr_import_runs
        SET scanned=?, imported=?, merged=?, skipped=?, status=?, finished_at=datetime('now')
        WHERE id=?
        """,
        (
            int(result.get("scanned", 0) or 0),
            int(result.get("imported", 0) or 0),
            int(result.get("merged", 0) or 0),
            int(result.get("skipped", 0) or 0),
            str(result.get("status", "ok")),
            run_id,
        ),
    )
    conn.commit()


def import_retrieval_db(path: str | Path | None = None, limit: int | None = None) -> dict[str, Any]:
    import torch

    base = _default_external_dir()
    target = Path(path) if path else None
    if target is None and base is not None:
        candidate = base / "data" / "graphdiff_cache" / "retrieval_db_v2.pt"
        target = candidate if candidate.exists() else None
    if target is None:
        candidate = (
            Path(__file__).resolve().parents[2] / "data" / "retrieval_db_v2.pt"
        )
        target = candidate if candidate.exists() else None
    if target is None or not target.exists():
        return {"imported": 0, "source": str(target) if target else "", "status": "missing"}

    # The retrieval cache is data-only. Never allow pickle object construction,
    # even though API callers cannot select an arbitrary path.
    obj = torch.load(str(target), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError("Retrieval cache must contain a dictionary")
    smiles_list = obj.get("smiles_list", [])
    shift_lists = obj.get("shift_lists", [])
    if not isinstance(smiles_list, (list, tuple)) or not isinstance(shift_lists, (list, tuple)):
        raise ValueError("Retrieval cache has an invalid schema")
    conn = _connect()
    imported = 0
    for i, smiles in enumerate(smiles_list[:limit]):
        shifts = shift_lists[i] if i < len(shift_lists) else []
        if not smiles or not shifts:
            continue
        info = _mol_info_from_smiles(str(smiles))
        status = _upsert_record(
            conn,
            source="retrieval_db_v2",
            source_id=str(i),
            name="",
            smiles=info["canonical_smiles"],
            inchikey=info["inchikey"],
            formula=info["formula"],
            mw=info["mw"],
            peaks_13c=[float(s) for s in shifts],
            peaks_1h=[],
            metadata={"source_file": str(target)},
        )
        imported += int(status == "inserted")
    conn.commit()
    conn.close()
    return {"imported": imported, "source": str(target), "status": "ok"}


def import_nmrshiftdb_sd(
    path: str | Path | None = None,
    limit: int | None = 25000,
    time_budget_s: float = 30.0,
    progress_every: int = 0,
    commit_every: int = 1000,
    quiet_rdkit: bool = False,
) -> dict[str, Any]:
    base = _default_external_dir()
    target = Path(path) if path else (base / "nmrshiftdb2withsignals.sd" if base else None)
    if target is None or not target.exists():
        return {"imported": 0, "source": str(target) if target else "", "status": "missing"}
    if Chem is None:
        return {"imported": 0, "source": str(target), "status": "rdkit_unavailable"}
    if quiet_rdkit:
        try:
            from rdkit import RDLogger
            RDLogger.DisableLog("rdApp.*")
        except Exception:
            pass

    conn = _connect()
    run_id = _start_import_run(conn, "nmrshiftdb2", target)
    imported = 0
    merged = 0
    skipped = 0
    scanned = 0
    started = time.time()
    for entry in stream_sd(target):
        scanned += 1
        if limit is not None and scanned > limit:
            break
        if time.time() - started > time_budget_s:
            break
        if not entry.get("signals_13c") and not entry.get("signals_1h"):
            continue
        mol = Chem.MolFromMolBlock(entry.get("molblock", ""), sanitize=True, removeHs=False)
        if mol is None:
            continue
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        info = _mol_info_from_smiles(smiles)
        tags = entry.get("tags", {})
        source_id = str(tags.get("nmrshiftdb2 ID", tags.get("ID", scanned)))
        p13 = [float(p["shift"]) for p in entry.get("signals_13c", []) if p.get("shift") is not None]
        p1h = [
            {
                "shift": float(p["shift"]),
                "multiplicity": p.get("multiplicity") or "",
                "intensity": p.get("intensity", 0),
            }
            for p in entry.get("signals_1h", [])
            if p.get("shift") is not None
        ]
        status = _upsert_record(
            conn,
            source="nmrshiftdb2",
            source_id=source_id,
            name=str(tags.get("Name", tags.get("NAME", ""))),
            smiles=info["canonical_smiles"],
            inchikey=info["inchikey"],
            formula=info["formula"],
            mw=info["mw"],
            peaks_13c=p13,
            peaks_1h=p1h,
            metadata={
                "solvent": tags.get("Solvent", ""),
                "field_mhz": tags.get("Field Strength [MHz]", ""),
                "source_file": str(target),
            },
        )
        imported += int(status == "inserted")
        merged += int(status == "merged")
        skipped += int(status == "skipped")
        if commit_every > 0 and scanned % commit_every == 0:
            conn.commit()
        if progress_every > 0 and scanned % progress_every == 0:
            elapsed = time.time() - started
            rate = scanned / elapsed if elapsed > 0 else 0
            print(
                f"[nmr-import] scanned={scanned} imported={imported} merged={merged} "
                f"skipped={skipped} elapsed={elapsed:.0f}s rate={rate:.1f}/s",
                flush=True,
            )
    conn.commit()
    result = {"imported": imported, "merged": merged, "skipped": skipped, "scanned": scanned, "source": str(target), "status": "ok"}
    _finish_import_run(conn, run_id, result)
    conn.close()
    return result


def ensure_quick_index() -> dict[str, Any]:
    conn = _connect()
    row = conn.execute("SELECT COUNT(*) FROM nmr_records").fetchone()
    count = int(row[0] or 0)
    conn.close()
    if count:
        return {"status": "ready", "records": count, "index": str(_index_path())}
    result = import_retrieval_db(limit=5000)
    conn = _connect()
    count = int(conn.execute("SELECT COUNT(*) FROM nmr_records").fetchone()[0] or 0)
    conn.close()
    return {"status": result.get("status"), "records": count, "index": str(_index_path()), "import": result}


def index_status() -> dict[str, Any]:
    conn = _connect()
    rows = conn.execute("SELECT source, COUNT(*) FROM nmr_records GROUP BY source").fetchall()
    total = int(conn.execute("SELECT COUNT(*) FROM nmr_records").fetchone()[0] or 0)
    with_1h = int(conn.execute("SELECT COUNT(*) FROM nmr_records WHERE peaks_1h != '[]'").fetchone()[0] or 0)
    with_13c = int(conn.execute("SELECT COUNT(*) FROM nmr_records WHERE peaks_13c != '[]'").fetchone()[0] or 0)
    formulas = int(conn.execute("SELECT COUNT(DISTINCT formula) FROM nmr_records WHERE formula != ''").fetchone()[0] or 0)
    runs = conn.execute(
        """
        SELECT source, source_path, scanned, imported, merged, skipped, status, created_at, finished_at
        FROM nmr_import_runs
        ORDER BY id DESC
        LIMIT 5
        """
    ).fetchall()
    conn.close()
    return {
        "index": str(_index_path()),
        "total_records": total,
        "with_1h": with_1h,
        "with_13c": with_13c,
        "distinct_formulas": formulas,
        "by_source": {source: count for source, count in rows},
        "recent_imports": [
            {
                "source": row[0],
                "source_path": row[1],
                "scanned": row[2],
                "imported": row[3],
                "merged": row[4],
                "skipped": row[5],
                "status": row[6],
                "created_at": row[7],
                "finished_at": row[8],
            }
            for row in runs
        ],
        "external_data_dir": str(_default_external_dir() or ""),
    }


@lru_cache(maxsize=8)
def _load_records_cached(index_path: str, mtime_ns: int, size: int, limit: int) -> tuple[dict[str, Any], ...]:
    conn = sqlite3.connect(index_path)
    query = """
        SELECT id, source, source_id, name, smiles, inchikey, formula, mw,
               peaks_13c, peaks_1h, metadata
        FROM nmr_records
    """
    if limit > 0:
        rows = conn.execute(query + " LIMIT ?", (limit,)).fetchall()
    else:
        rows = conn.execute(query).fetchall()
    conn.close()
    return _records_from_rows(rows)


def _records_from_rows(rows: list[tuple[Any, ...]]) -> tuple[dict[str, Any], ...]:
    records = []
    for row in rows:
        records.append({
            "id": row[0],
            "source": row[1],
            "source_id": row[2],
            "name": row[3],
            "smiles": row[4],
            "inchikey": row[5],
            "formula": row[6],
            "mw": row[7],
            "peaks_13c": json.loads(row[8] or "[]"),
            "peaks_1h": json.loads(row[9] or "[]"),
            "metadata": json.loads(row[10] or "{}"),
        })
    return tuple(records)


@lru_cache(maxsize=8)
def _normalized_records_cached(
    index_path: str, mtime_ns: int, size: int, limit: int
) -> tuple[tuple[list[float], list[float]], ...]:
    """Precompute per-record normalised peak lists, cached alongside the rows.

    Reference-side normalisation is deterministic, so computing it once per
    index (instead of once per query per record) preserves scoring exactly
    while removing the dominant per-record cost of rank_candidates.
    """

    records = _load_records_cached(index_path, mtime_ns, size, limit)
    return tuple(
        (
            _normalise_resonance_shifts(rec["peaks_13c"], "13C"),
            _normalise_resonance_shifts(rec["peaks_1h"], "1H"),
        )
        for rec in records
    )


def _normalized_records(
    index_path: str, mtime_ns: int, size: int, limit: int
) -> tuple[tuple[list[float], list[float]], ...]:
    return _normalized_records_cached(index_path, mtime_ns, size, limit)


def _load_records(limit: int = 0) -> list[dict[str, Any]]:
    ensure_quick_index()
    path = _index_path()
    stat = path.stat()
    return list(_load_records_cached(str(path), stat.st_mtime_ns, stat.st_size, limit))


def _load_records_by_formula(formula: str) -> list[dict[str, Any]]:
    """Load every exact-formula isomer before applying any result limit."""

    formula_key = canonical_formula(formula)
    ensure_quick_index()
    conn = _connect()
    rows = conn.execute(
        """
        SELECT id, source, source_id, name, smiles, inchikey, formula, mw,
               peaks_13c, peaks_1h, metadata
        FROM nmr_records
        WHERE formula = ?
        """,
        (formula_key,),
    ).fetchall()
    conn.close()
    return list(_records_from_rows(rows))


def _match_score(query: list[float], reference: list[float], tolerance: float, nucleus_weight: float) -> dict[str, Any]:
    empty_result = {
        "score": 0.0,
        "matched": 0,
        "mae": None,
        "query_recall": 0.0 if query else None,
        "reference_coverage": 0.0 if reference else None,
        "count_balance": 0.0,
        "error_score": 0.0,
        "query_count": len(query),
        "reference_count": len(reference),
        "explained": [],
        "assignments": [],
    }
    if not query or not reference:
        return empty_result

    query_values = np.asarray(query, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    costs = np.abs(query_values[:, None] - reference_values[None, :])
    query_indices, reference_indices = linear_sum_assignment(costs)
    assignments = []
    errors = []
    explained = []
    for query_index, reference_index in zip(query_indices, reference_indices, strict=True):
        error = float(costs[query_index, reference_index])
        if error <= tolerance:
            errors.append(error)
            explained.append(float(query_values[query_index]))
            assignments.append(
                {
                    "query_shift": round(float(query_values[query_index]), 6),
                    "reference_shift": round(float(reference_values[reference_index]), 6),
                    "error": round(error, 6),
                }
            )
    matched = len(assignments)
    if matched == 0:
        return empty_result

    recall = matched / len(query)
    reference_coverage = matched / len(reference)
    balance = 1.0 - min(abs(len(query) - len(reference)) / max(len(query), len(reference), 1), 1.0)
    err_score = float(np.exp(-np.mean(errors) / max(tolerance, 1e-6)))
    score = nucleus_weight * (
        _SCORE_WEIGHTS["recall"] * recall
        + _SCORE_WEIGHTS["reference_coverage"] * reference_coverage
        + _SCORE_WEIGHTS["count_balance"] * balance
        + _SCORE_WEIGHTS["error_score"] * err_score
    )
    return {
        "score": score,
        "matched": matched,
        "mae": round(float(np.mean(errors)), 4),
        "query_recall": recall,
        "reference_coverage": reference_coverage,
        "count_balance": balance,
        "error_score": err_score,
        "query_count": len(query),
        "reference_count": len(reference),
        "explained": explained,
        "assignments": assignments,
    }


def _feature_vector(
    q13: list[float],
    q1h: list[float],
    rec: dict[str, Any],
    formula_constraint: str | None = None,
    *,
    _query_norm: tuple[list[float], list[float]] | None = None,
    _ref_norm: tuple[list[float], list[float]] | None = None,
) -> tuple[list[float], dict[str, Any]]:
    # _query_norm/_ref_norm let hot paths (rank_candidates scanning tens of
    # thousands of records) hoist the deterministic normalisation out of the
    # per-record loop.  The values must be produced by
    # _normalise_resonance_shifts on the same inputs, so scoring stays
    # bit-identical to computing them here.
    if _query_norm is not None:
        q13, q1h = _query_norm
    else:
        q13 = _normalise_resonance_shifts(q13, "13C")
        q1h = _normalise_resonance_shifts(q1h, "1H")
    if _ref_norm is not None:
        ref13, ref1h = _ref_norm
    else:
        ref13 = _normalise_resonance_shifts(rec["peaks_13c"], "13C")
        ref1h = _normalise_resonance_shifts(rec["peaks_1h"], "1H")
    s13 = _match_score(q13, ref13, tolerance=_MATCH_TOLERANCES["13C"], nucleus_weight=1.0)
    s1h = _match_score(q1h, ref1h, tolerance=_MATCH_TOLERANCES["1H"], nucleus_weight=1.0)
    formula_match = 1.0 if formula_constraint and _formula_key(rec.get("formula")) == _formula_key(formula_constraint) else 0.0
    formula_missing = 1.0 if not formula_constraint else 0.0
    mw = float(rec.get("mw") or 0)
    features = [
        float(s13["score"]),
        float(s1h["score"]),
        float(s13["matched"]) / max(len(q13), 1),
        float(s1h["matched"]) / max(len(q1h), 1),
        0.0 if s13["mae"] is None else float(s13["mae"]),
        0.0 if s1h["mae"] is None else float(s1h["mae"]),
        float(s13["mae"] is None),
        float(s1h["mae"] is None),
        len(q13),
        len(q1h),
        len(ref13),
        len(ref1h),
        abs(len(q13) - len(ref13)),
        abs(len(q1h) - len(ref1h)),
        formula_match,
        formula_missing,
        mw / 1000.0,
    ]
    return features, {"c13": s13, "h1": s1h, "formula_match": bool(formula_match)}


def _load_ranker():
    path = _model_path()
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _formula_candidates(records: list[dict[str, Any]], formula: str | None, include_isomers: bool = True) -> list[dict[str, Any]]:
    formula_norm = _formula_key(formula)
    if not formula_norm:
        return records
    matched = [rec for rec in records if _formula_key(rec.get("formula")) == formula_norm]
    return matched


def _eligible_training_records(limit_records: int) -> list[dict[str, Any]]:
    return [
        rec
        for rec in _load_records(limit=limit_records)
        if rec.get("smiles")
        and (
            len(rec.get("peaks_13c") or []) >= 3
            or len(rec.get("peaks_1h") or []) >= 2
        )
    ]


def _record_group(rec: dict[str, Any]) -> str:
    """Return a molecule/scaffold group used to prevent split leakage."""

    smiles = str(rec.get("smiles") or "")
    if Chem is not None and MurckoScaffold is not None and smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if scaffold:
                return f"scaffold:{scaffold}"
    identity = _first_block_key(str(rec.get("inchikey") or ""))
    return f"molecule:{identity or smiles}"


def _build_pairs_for_records(
    records: list[dict[str, Any]],
    *,
    negatives_per_positive: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(records) < 20:
        raise ValueError("Not enough NMR records to train a ranker")
    formula_groups: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        formula_groups.setdefault(_formula_key(rec.get("formula")), []).append(i)
    X: list[list[float]] = []
    y: list[int] = []
    for i, rec in enumerate(records):
        q13 = [float(v) for v in rec["peaks_13c"] if isinstance(v, (int, float))]
        q1h = _peak_values(rec["peaks_1h"])
        feat, _ = _feature_vector(q13, q1h, rec, formula_constraint=rec.get("formula"))
        X.append(feat)
        y.append(1)
        candidate_negatives = list(range(len(records)))
        same_formula = [idx for idx in formula_groups.get(_formula_key(rec.get("formula")), []) if idx != i]
        hard_pool = same_formula if same_formula else candidate_negatives
        for j in range(negatives_per_positive):
            if j < len(same_formula):
                neg_idx = int(rng.choice(same_formula))
            else:
                neg_idx = int(rng.choice(hard_pool))
                tries = 0
                while neg_idx == i and tries < 10:
                    neg_idx = int(rng.choice(candidate_negatives))
                    tries += 1
            neg = records[neg_idx]
            feat, _ = _feature_vector(q13, q1h, neg, formula_constraint=rec.get("formula"))
            X.append(feat)
            y.append(0)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def build_training_pairs(
    limit_records: int = 30000,
    negatives_per_positive: int = 8,
    seed: int = 20260711,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build pairs for diagnostics.

    Model training uses a group split *before* this operation so augmented
    pairs and negative candidates cannot cross the validation boundary.
    """

    records = _eligible_training_records(limit_records)
    X, y = _build_pairs_for_records(
        records,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
    )
    meta = {
        "records": len(records),
        "pairs": len(y),
        "negatives_per_positive": negatives_per_positive,
        "evaluation_protocol": "diagnostic_unsplit",
    }
    return X, y, meta


def train_joint_ranker(
    limit_records: int = 30000,
    negatives_per_positive: int = 8,
    seed: int = 20260711,
) -> dict[str, Any]:
    records = _eligible_training_records(limit_records)
    groups = np.asarray([_record_group(rec) for rec in records], dtype=object)
    if len(set(groups.tolist())) < 5:
        raise ValueError("Not enough distinct molecular scaffolds for grouped validation")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, validation_indices = next(
        splitter.split(np.arange(len(records)), groups=groups)
    )
    train_records = [records[int(index)] for index in train_indices]
    validation_records = [records[int(index)] for index in validation_indices]
    X_train, y_train = _build_pairs_for_records(
        train_records,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
    )
    X_val, y_val = _build_pairs_for_records(
        validation_records,
        negatives_per_positive=negatives_per_positive,
        seed=seed + 1,
    )
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, max_leaf_nodes=31, random_state=seed)),
    ])
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    auc = float(roc_auc_score(y_val, proba)) if len(set(y_val.tolist())) > 1 else 0.0
    ap = float(average_precision_score(y_val, proba))
    supports_modalities = sorted(
        {
            ("1h+13c" if rec.get("peaks_1h") and rec.get("peaks_13c") else "1h" if rec.get("peaks_1h") else "13c")
            for rec in train_records
        }
    )
    meta = {
        "records": len(records),
        "training_records": len(train_records),
        "validation_records": len(validation_records),
        "training_pairs": len(y_train),
        "validation_pairs": len(y_val),
        "negatives_per_positive": negatives_per_positive,
        "evaluation_protocol": "legacy_pairwise_scaffold_split_diagnostic_v1",
        "production_eligible": False,
        "limitation": (
            "Positive pairs reuse the same stored spectrum as query and "
            "reference; this diagnostic cannot establish structure-level "
            "generalisation or calibrated correctness probabilities."
        ),
        "supports_modalities": supports_modalities,
        "seed": seed,
    }
    configured_path = _model_path()
    path = configured_path.with_name(
        f"{configured_path.stem}.legacy-diagnostic{configured_path.suffix}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(
            {"model": model, "meta": meta, "features": _FEATURE_NAMES},
            temporary_path,
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "status": "diagnostic_only",
        "model_path": str(path),
        "production_model_unchanged": str(configured_path),
        "validation_roc_auc": round(auc, 4),
        "validation_average_precision": round(ap, 4),
        **meta,
    }


_FEATURE_NAMES = [
    "c13_score",
    "h1_score",
    "c13_recall",
    "h1_recall",
    "c13_mae",
    "h1_mae",
    "c13_mae_missing",
    "h1_mae_missing",
    "query_13c_count",
    "query_1h_count",
    "ref_13c_count",
    "ref_1h_count",
    "delta_13c_count",
    "delta_1h_count",
    "formula_match",
    "formula_missing",
    "mw_scaled",
]


def _fingerprint_similarity(smiles_a: str, smiles_b: str) -> float:
    if Chem is None or rdFingerprintGenerator is None or TanimotoSimilarity is None:
        return 0.0
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return float(TanimotoSimilarity(gen.GetFingerprint(mol_a), gen.GetFingerprint(mol_b)))


def rank_candidates(
    peaks_13c: list[dict[str, Any]] | None = None,
    peaks_1h: list[dict[str, Any]] | None = None,
    generated_smiles: list[str] | None = None,
    formula: str | None = None,
    top_k: int = 10,
    max_records: int = 30000,
) -> dict[str, Any]:
    q13 = _normalise_resonance_shifts(peaks_13c or [], "13C")
    q1h = _normalise_resonance_shifts(peaks_1h or [], "1H")
    if not q13 and not q1h:
        raise ValueError("No NMR peaks provided")

    # Hoist the query-side second normalisation out of the per-record loop.
    # _feature_vector historically re-normalised the (already normalised)
    # query for every record; computing it once here is bit-identical.
    query_norm = (
        _normalise_resonance_shifts(q13, "13C"),
        _normalise_resonance_shifts(q1h, "1H"),
    )
    formula_info = parse_formula(formula)
    if formula_info is not None:
        # Formula filtering happens in SQL and before any limit.  It is a hard
        # chemical constraint; zero matches must never fall back to all records.
        records = _load_records_by_formula(formula_info.canonical)
        ref_norms = [
            (
                _normalise_resonance_shifts(rec["peaks_13c"], "13C"),
                _normalise_resonance_shifts(rec["peaks_1h"], "1H"),
            )
            for rec in records
        ]
    else:
        records = _load_records(limit=max_records)
        index_path = _index_path()
        stat = index_path.stat()
        ref_norms = _normalized_records(
            str(index_path), stat.st_mtime_ns, stat.st_size, max_records
        )

    modality = "1h+13c" if q1h and q13 else "1h" if q1h else "13c"
    ranker_bundle = _load_ranker()
    ranker_meta = ranker_bundle.get("meta", {}) if isinstance(ranker_bundle, dict) else {}
    grouped_protocol = (
        ranker_meta.get("evaluation_protocol")
        == "independent_spectrum_scaffold_split_v2"
        and ranker_meta.get("production_eligible") is True
    )
    supported_modalities = set(ranker_meta.get("supports_modalities") or [])
    stored_features = ranker_bundle.get("features") if isinstance(ranker_bundle, dict) else None
    feature_schema_matches = (
        isinstance(stored_features, (list, tuple))
        and list(stored_features) == _FEATURE_NAMES
    )
    ranker = (
        ranker_bundle.get("model")
        if isinstance(ranker_bundle, dict)
        and grouped_protocol
        and modality in supported_modalities
        and feature_schema_matches
        else None
    )
    if ranker is not None:
        ranker_reason = "independent_spectrum_grouped_validation_model"
    elif ranker_bundle is None:
        ranker_reason = "model_unavailable"
    elif not grouped_protocol:
        ranker_reason = "unvalidated_training_protocol_disabled"
    elif not feature_schema_matches:
        ranker_reason = "feature_schema_mismatch"
    else:
        ranker_reason = f"unsupported_modality:{modality}"

    candidate_rows = []
    candidate_features = []
    for rec, ref_norm in zip(records, ref_norms, strict=True):
        features, breakdown = _feature_vector(
            q13,
            q1h,
            rec,
            formula_constraint=formula,
            _query_norm=query_norm,
            _ref_norm=ref_norm,
        )
        s13 = breakdown["c13"]
        s1h = breakdown["h1"]
        score = None
        if ranker is None:
            if q13 and q1h:
                score = 0.5 * s13["score"] + 0.5 * s1h["score"]
            elif q13:
                score = s13["score"]
            else:
                score = s1h["score"]
        row = {
            "smiles": rec["smiles"],
            "compound_name": rec["name"] or rec["source_id"] or rec["smiles"],
            "molecular_formula": rec["formula"],
            "molecular_weight": round(float(rec["mw"]), 3) if rec["mw"] else None,
            "source": rec["source"],
            "source_id": rec["source_id"],
            "score_breakdown": {
                "c13": s13,
                "h1": s1h,
                "formula_match": breakdown["formula_match"],
                "ranker_model": ranker is not None,
                "generated_similarity_bonus": False,
            },
            "matched_13c": s13["matched"],
            "matched_1h": s1h["matched"],
            "_score": score,
        }
        candidate_rows.append(row)
        candidate_features.append(features)
    if ranker is not None and candidate_features:
        scores = ranker.predict_proba(np.array(candidate_features, dtype=float))[:, 1]
        for row, score in zip(candidate_rows, scores, strict=False):
            # A learned prior must never manufacture spectral evidence when
            # neither observed nucleus has a valid assignment.
            row["_score"] = (
                float(score)
                if int(row["matched_13c"]) + int(row["matched_1h"]) > 0
                else 0.0
            )

    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        score = float(row.pop("_score", 0.0) or 0.0)
        c13 = row["score_breakdown"]["c13"]
        h1 = row["score_breakdown"]["h1"]
        observed = [
            (match, _MATCH_TOLERANCES[nucleus])
            for nucleus, query, match in (("13C", q13, c13), ("1H", q1h, h1))
            if query
        ]
        query_recalls = [float(match["query_recall"] or 0.0) for match, _ in observed]
        reference_coverages = [
            float(match["reference_coverage"] or 0.0)
            for match, _ in observed
        ]
        reference_counts = [
            int(match["reference_count"])
            for match, _ in observed
        ]
        matched_counts = [
            int(match["matched"])
            for match, _ in observed
        ]
        normalised_maes = [
            float(match["mae"]) / tolerance
            for match, tolerance in observed
            if match["mae"] is not None
        ]
        all_observed_matched = bool(observed) and all(
            int(match["matched"]) > 0 and int(match["reference_count"]) > 0
            for match, _ in observed
        )
        mean_recall = float(np.mean(query_recalls)) if query_recalls else 0.0
        mean_reference_coverage = (
            float(np.mean(reference_coverages)) if reference_coverages else 0.0
        )
        worst_normalised_mae = max(normalised_maes, default=float("inf"))
        if observed and not any(reference_counts):
            evidence_level = "no_reference"
        elif observed and not any(matched_counts):
            evidence_level = "no_match"
        elif (
            q13
            and q1h
            and all_observed_matched
            and min(query_recalls) >= 0.7
            and min(reference_coverages) >= 0.7
            and worst_normalised_mae <= 0.35
        ):
            evidence_level = "strong"
        elif (
            all_observed_matched
            and mean_recall >= 0.5
            and mean_reference_coverage >= 0.5
            and worst_normalised_mae <= 0.65
        ):
            evidence_level = "moderate"
        else:
            evidence_level = "weak"
        row["_ranking_score"] = score
        row["ranking_score"] = round(max(min(score, 1.0), 0.0), 4)
        row["evidence_level"] = evidence_level
        row["evidence"] = {
            "query_modality": modality,
            "c13_recall": round(float(c13["query_recall"]), 4) if q13 else None,
            "h1_recall": round(float(h1["query_recall"]), 4) if q1h else None,
            "c13_reference_coverage": (
                round(float(c13["reference_coverage"] or 0.0), 4) if q13 else None
            ),
            "h1_reference_coverage": (
                round(float(h1["reference_coverage"] or 0.0), 4) if q1h else None
            ),
            "c13_normalized_mae": (
                round(float(c13["mae"]) / _MATCH_TOLERANCES["13C"], 4)
                if q13 and c13["mae"] is not None
                else None
            ),
            "h1_normalized_mae": (
                round(float(h1["mae"]) / _MATCH_TOLERANCES["1H"], 4)
                if q1h and h1["mae"] is not None
                else None
            ),
            "calibrated_probability": False,
        }
        rows.append(row)

    rows.sort(
        key=lambda item: (
            -float(item["_ranking_score"]),
            -(int(item["matched_13c"]) + int(item["matched_1h"])),
            str(item["smiles"]),
        )
    )
    unique = []
    seen = set()
    for row in rows:
        if row["smiles"] in seen:
            continue
        seen.add(row["smiles"])
        row.pop("_ranking_score", None)
        row["rank"] = len(unique) + 1
        unique.append(row)
        if len(unique) >= top_k:
            break

    warnings = []
    if formula_info is not None and not records:
        warnings.append(
            "No reference structures match the canonical molecular formula; "
            "the search was not broadened automatically."
        )
    if modality != "1h+13c":
        warnings.append(
            "Only one NMR nucleus was supplied; candidates are hypotheses and "
            "normally cannot establish a unique structure."
        )
    if ranker is None:
        warnings.append(
            "The learned ranker was not used because no valid grouped-validation "
            f"model supports {modality}."
        )
    if generated_smiles:
        warnings.append(
            "Generated SMILES are reported separately and do not influence "
            "spectral-library ranking until a forward-spectrum check is available."
        )

    return {
        "candidates": unique,
        "candidate_pool_status": (
            "formula_match"
            if formula_info is not None and records
            else "no_formula_match"
            if formula_info is not None
            else "unconstrained_limited_search"
        ),
        "query": {
            "n_13c": len(q13),
            "n_1h": len(q1h),
            "peaks_13c": q13,
            "peaks_1h": q1h,
            "formula": formula_info.canonical if formula_info else None,
            "formula_input": formula,
            "dbe": formula_info.dbe if formula_info else None,
            "modality": modality,
        },
        "warnings": warnings,
        "index": index_status(),
        "ranker": {
            "loaded": ranker is not None,
            "path": str(_model_path()),
            "reason": ranker_reason,
            "calibrated_probability": False,
        },
    }


def analyze_mixture(
    peaks_13c: list[dict[str, Any]] | None = None,
    peaks_1h: list[dict[str, Any]] | None = None,
    formula: str | None = None,
    top_k: int = 5,
    max_components: int = 3,
) -> dict[str, Any]:
    q13 = _normalise_resonance_shifts(peaks_13c or [], "13C")
    q1h_peaks = _query_peaks(peaks_1h)
    q1h = _normalise_resonance_shifts(q1h_peaks, "1H")
    first_pass = rank_candidates(peaks_13c, peaks_1h, formula=formula, top_k=30)
    if not first_pass["candidates"]:
        return {
            "status": "no_candidates",
            "is_mixture": None,
            "components": [],
            "unexplained": {"peaks_13c": q13, "peaks_1h": q1h},
            "coverage": {"c13": 0.0 if q13 else None, "h1": 0.0 if q1h else None},
            "warnings": first_pass.get("warnings", []),
        }
    components = []
    unexplained_13c = set(round(v, 3) for v in q13)
    unexplained_1h = set(round(v, 4) for v in q1h)
    for cand in first_pass["candidates"]:
        bd = cand["score_breakdown"]
        explained13 = {round(v, 3) for v in bd["c13"].get("explained", [])}
        explained1h = {round(v, 4) for v in bd["h1"].get("explained", [])}
        new13 = explained13 & unexplained_13c
        new1h = explained1h & unexplained_1h
        if not new13 and not new1h:
            continue
        components.append({
            **cand,
            "explained_13c": sorted(new13),
            "explained_1h": sorted(new1h),
        })
        unexplained_13c -= new13
        unexplained_1h -= new1h
        if len(components) >= max_components:
            break
    total_integral = sum((p.integral or 0) for p in q1h_peaks)
    for comp in components:
        comp_integral = sum((p.integral or 0) for p in q1h_peaks if round(p.shift, 4) in set(comp["explained_1h"]))
        comp["relative_amount_percent"] = round(comp_integral / total_integral * 100, 2) if total_integral > 0 else None
    evidence_sufficient = bool(q13 and q1h) or total_integral > 0
    mixture_signal = len(components) > 1 or (
        (len(unexplained_13c) + len(unexplained_1h))
        > max(3, 0.25 * (len(q13) + len(q1h)))
    )
    warnings = list(first_pass.get("warnings", []))
    if not evidence_sufficient:
        warnings.append(
            "Mixture classification was withheld: a single unintegrated nucleus "
            "does not provide enough evidence for component decomposition."
        )
    return {
        "status": "exploratory" if evidence_sufficient else "insufficient_evidence",
        "is_mixture": mixture_signal if evidence_sufficient else None,
        "components": components[:top_k],
        "unexplained": {
            "peaks_13c": sorted(unexplained_13c),
            "peaks_1h": sorted(unexplained_1h),
        },
        "coverage": {
            "c13": round(1 - len(unexplained_13c) / max(len(q13), 1), 3) if q13 else None,
            "h1": round(1 - len(unexplained_1h) / max(len(q1h), 1), 3) if q1h else None,
        },
        "warnings": warnings,
    }
