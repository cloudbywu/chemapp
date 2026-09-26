"""NMR → SMILES structure elucidation using generation + spectral retrieval."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Annotated, Any, Literal, Optional
import uuid

import torch
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from app.api.deps import require_admin
from app.paths import nmr_index_v2_path
from app.core.models import Technique
from app.ml.nmr_data_v2 import connect_readonly, schema_version
from app.ml.nmr_evidence import (
    FormulaError,
    build_generation_prompt,
    parse_formula,
    prepare_query_peaks,
    validate_generated_smiles,
)
from app.ml.external_nmr_benchmark import (
    ExternalNMRBenchmarkError,
    load_external_nmr_status,
)
from app.ml.nmr_forward import (
    NMRForwardConfigurationError,
    NMRForwardError,
    NMRForwardInputError,
    NMRForwardProtocolError,
    NMRForwardTimeoutError,
    NMRForwardUnavailableError,
    dp5q_sidecar_is_configured,
    get_nmr_forward_adapter,
)
from app.ml.nmr_quantile_forward import (
    DP5Q_UNASSIGNED_SHADOW_SEMANTICS,
    dp5q_quantile_sidecar_is_configured,
    get_nmr_quantile_forward_adapter,
)
from app.ml.nmr_hybrid_predictor import (
    CandidateConstraints,
    HybridPredictorInputError,
    PIPELINE_VERSION as HYBRID_PIPELINE_VERSION,
    SCHEMA_VERSION as HYBRID_SCHEMA_VERSION,
    hybrid_rank_candidates,
    hybrid_to_legacy_rank_result,
)
from app.ml.nmr_structure_elucidation import (
    analyze_mixture,
    ensure_quick_index,
    import_nmrshiftdb_sd,
    import_retrieval_db,
    index_status,
    rank_candidates,
    train_joint_ranker,
)

router = APIRouter(prefix="/api/ml/elucidate", tags=["elucidation"])
logger = logging.getLogger(__name__)

NMRSHIFTDB2_LICENSE_URI = (
    "https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/"
    "nmrshiftdb2datalicense.txt"
)
FORWARD_SCHEMA_VERSION = "dp5q-shadow-v1"
FORWARD_PROVIDER = "dp5q_mean_13c"
FORWARD_EVIDENCE_SCHEMA_VERSION = "dp5q-13c-fit-v1"
QUANTILE_SHADOW_SCHEMA_VERSION = "dp5q-quantile-shadow-v1"
QUANTILE_SHADOW_PROVIDER = "dp5q_99quantiles_13c"
_FORWARD_SEMAPHORE = threading.BoundedSemaphore(1)
_FORWARD_SUPPORTED_ATOMIC_NUMBERS = frozenset(
    {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35}
)
_FORWARD_MAX_HEAVY_ATOMS = 80
_FORWARD_MAX_TOTAL_ATOMS = 200
_FORWARD_MAX_ROTATABLE_BONDS = 20
_FORWARD_MAX_SMILES_LENGTH = 1024

MODEL_DIR = Path(
    os.environ.get(
        "CHEMAPP_T5_MODEL_DIR",
        str(Path(__file__).parent.parent.parent / "ml" / "pretrained" / "t5_nmr"),
    )
)
EXTERNAL_SMOKE_STATUS_PATH = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "external-nmr-smoke-v1-summary.json"
)
_model = None
_tokenizer = None
_device = None
_model_load_lock = threading.Lock()
_forward_scorer_cache: dict[str, Any] = {}
_forward_scorer_lock = threading.Lock()
_candidate_providers_cache: dict[str, tuple[Any, ...]] = {}
_candidate_providers_lock = threading.Lock()
_ranker_jobs: dict[str, dict[str, Any]] = {}
_ranker_jobs_lock = threading.Lock()
_RANKER_JOB_MAX_AGE_SECONDS = 24 * 3600
_RANKER_JOB_MAX_ENTRIES = 100


def _prune_ranker_jobs() -> None:
    """Drop only terminal jobs and cap the in-memory job table.

    Running jobs are never evicted, even when the table is over capacity.
    """

    now_ts = time.time()
    with _ranker_jobs_lock:
        stale = [
            job_id
            for job_id, job in _ranker_jobs.items()
            if job.get("status") in {"done", "error"}
            and (
                (
                    "finished_at" in job
                    and now_ts - float(job["finished_at"])
                    > _RANKER_JOB_MAX_AGE_SECONDS
                )
                or (
                    "finished_at" not in job
                    and "started_at" in job
                    and now_ts - float(job["started_at"])
                    > _RANKER_JOB_MAX_AGE_SECONDS
                )
            )
        ]
        for job_id in stale:
            _ranker_jobs.pop(job_id, None)
        terminal = [
            job_id
            for job_id, job in _ranker_jobs.items()
            if job.get("status") in {"done", "error"}
        ]
        terminal.sort(
            key=lambda job_id: _ranker_jobs[job_id].get(
                "finished_at",
                _ranker_jobs[job_id].get("started_at", 0.0),
            )
        )
        while len(_ranker_jobs) > _RANKER_JOB_MAX_ENTRIES and terminal:
            oldest = terminal.pop(0)
            _ranker_jobs.pop(oldest, None)


def _run_ranker_job(
    job_id: str,
    payload: "TrainRankerRequest",
    lock_path: Path,
) -> None:
    from app.api.routes.ml import _release_training_lock

    try:
        result = train_joint_ranker(
            limit_records=payload.limit_records,
            negatives_per_positive=payload.negatives_per_positive,
            seed=payload.seed,
        )
        status: dict[str, Any] = {"status": "done", "result": result}
    except Exception:
        # Never store raw exception text in the job table; the traceback
        # (with the job id) stays in the server log.
        logger.exception("Ranker training job %s failed", job_id)
        status = {"status": "error", "error": "Ranker training failed"}
    finally:
        _release_training_lock(lock_path, job_id)
        status["finished_at"] = time.time()
    with _ranker_jobs_lock:
        _ranker_jobs[job_id].update(status)


def _cached_forward_scorer() -> Any | None:
    """Process-wide cached CSP5 scorer honoring on/auto/off mode semantics."""

    from app.ml.deployment_check import csp5_mode

    mode = csp5_mode()
    if mode == "off":
        return None
    with _forward_scorer_lock:
        cached = _forward_scorer_cache.get("csp5")
        if cached is not None:
            return cached
    from app.ml.forward_v1.factory import get_forward_scorer

    scorer = get_forward_scorer("csp5")
    if scorer is None:
        return None
    if mode in {"1", "on", "true", "yes"}:
        from app.ml.forward_v1.csp5_scorer import Csp5ForwardScorer

        if not isinstance(scorer, Csp5ForwardScorer):
            logger.warning(
                "CHEMAPP_CSP5_MODE=on but CSP5 scorer is unavailable; "
                "refusing forward scoring"
            )
            return None
    if scorer is not None:
        with _forward_scorer_lock:
            _forward_scorer_cache["csp5"] = scorer
    return scorer


def _cached_candidate_providers() -> tuple[Any, ...]:
    """Process-wide cached hybrid generation providers (per enabled mode)."""

    mode = os.getenv("CHEMAPP_HYBRID_GENERATION", "off").strip().casefold()
    if mode not in {"1", "on", "true", "yes"}:
        return ()
    with _candidate_providers_lock:
        cached = _candidate_providers_cache.get(mode)
        if cached is not None:
            return cached
    providers: tuple[Any, ...] = ()
    try:
        from pathlib import Path as _Path

        from app.ml.nmr_candidate_generation_v2 import (
            HybridOpenWorldProvider,
            PubChemFormulaSource,
        )

        backend_root = _Path(__file__).resolve().parents[3]
        pubchem = PubChemFormulaSource(
            backend_root
            / "data"
            / "derived"
            / "nmr-candidate-generation-v2"
            / "pubchem-cache",
            offline=True,
        )
        provider = HybridOpenWorldProvider(
            backend_root / "data" / "nmr_spectral_index_v2.sqlite",
            pubchem,
        )
        providers = (provider,)
    except Exception:
        logger.exception("hybrid candidate provider unavailable")
        providers = ()
    with _candidate_providers_lock:
        _candidate_providers_cache[mode] = providers
    return providers


def _external_smoke_status() -> dict[str, Any]:
    """Expose only a validated, accuracy-gated external smoke status."""

    try:
        return load_external_nmr_status(EXTERNAL_SMOKE_STATUS_PATH)
    except ExternalNMRBenchmarkError:
        logger.warning("External NMR smoke status artifact is unavailable or invalid")
        return {
            "schema_version": "external-nmr-smoke-derived-v1",
            "status": "unavailable",
            "accuracy_eligibility": {
                "headline_accuracy_allowed": False,
                "independent_accuracy_test": False,
                "open_world_claim_allowed": False,
                "reasons": ["validated status artifact is unavailable"],
            },
        }


def _bounded_environment_number(
    name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> tuple[float | None, str | None]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default, None
    try:
        value = float(raw)
    except ValueError:
        return None, f"{name}_invalid"
    if not minimum <= value <= maximum:
        return None, f"{name}_out_of_range"
    return value, None


def _forward_settings() -> dict[str, Any]:
    raw_mode = os.getenv("CHEMAPP_DP5Q_MODE", "off").strip().casefold()
    if raw_mode not in {"off", "shadow"}:
        return {
            "mode": "off",
            "valid": False,
            "reason_code": "invalid_server_configuration",
        }
    maximum, max_error = _bounded_environment_number(
        "CHEMAPP_DP5Q_MAX_CANDIDATES",
        default=5,
        minimum=1,
        maximum=20,
    )
    busy_timeout, busy_error = _bounded_environment_number(
        "CHEMAPP_DP5Q_BUSY_TIMEOUT_SECONDS",
        default=0.1,
        minimum=0.0,
        maximum=2.0,
    )
    _runtime_timeout, runtime_error = _bounded_environment_number(
        "CHEMAPP_DP5Q_TIMEOUT_SECONDS",
        default=20.0,
        minimum=0.05,
        maximum=60.0,
    )
    _startup_timeout, startup_error = _bounded_environment_number(
        "CHEMAPP_DP5Q_STARTUP_TIMEOUT_SECONDS",
        default=45.0,
        minimum=0.05,
        maximum=90.0,
    )
    if maximum is not None and not float(maximum).is_integer():
        max_error = "CHEMAPP_DP5Q_MAX_CANDIDATES_invalid"
    if max_error or busy_error or runtime_error or startup_error:
        return {
            "mode": raw_mode,
            "valid": False,
            "reason_code": "invalid_server_configuration",
        }
    return {
        "mode": raw_mode,
        "valid": True,
        "reason_code": None,
        "max_candidates": int(maximum),
        "busy_timeout_seconds": float(busy_timeout),
    }


def _quantile_shadow_settings() -> dict[str, Any]:
    raw_mode = os.getenv(
        "CHEMAPP_DP5Q_QUANTILE_MODE", "off"
    ).strip().casefold()
    if raw_mode not in {"off", "shadow"}:
        return {
            "mode": "off",
            "valid": False,
            "reason_code": "invalid_server_configuration",
        }
    maximum, max_error = _bounded_environment_number(
        "CHEMAPP_DP5Q_QUANTILE_MAX_CANDIDATES",
        default=3,
        minimum=1,
        maximum=5,
    )
    busy_timeout, busy_error = _bounded_environment_number(
        "CHEMAPP_DP5Q_QUANTILE_BUSY_TIMEOUT_SECONDS",
        default=0.1,
        minimum=0.0,
        maximum=2.0,
    )
    _runtime_timeout, runtime_error = _bounded_environment_number(
        "CHEMAPP_DP5Q_TIMEOUT_SECONDS",
        default=20.0,
        minimum=0.05,
        maximum=60.0,
    )
    _startup_timeout, startup_error = _bounded_environment_number(
        "CHEMAPP_DP5Q_STARTUP_TIMEOUT_SECONDS",
        default=45.0,
        minimum=0.05,
        maximum=90.0,
    )
    if maximum is not None and not float(maximum).is_integer():
        max_error = "CHEMAPP_DP5Q_QUANTILE_MAX_CANDIDATES_invalid"
    if max_error or busy_error or runtime_error or startup_error:
        return {
            "mode": raw_mode,
            "valid": False,
            "reason_code": "invalid_server_configuration",
        }
    return {
        "mode": raw_mode,
        "valid": True,
        "reason_code": None,
        "max_candidates": int(maximum),
        "busy_timeout_seconds": float(busy_timeout),
    }


def _forward_summary(
    *,
    status: str,
    reason_code: str | None,
    model_called: bool,
    elapsed_ms: float = 0.0,
    evaluated_candidate_count: int = 0,
    skipped_candidate_count: int = 0,
    model: dict[str, Any] | None = None,
    mode: str = "shadow",
) -> dict[str, Any]:
    return {
        "schema_version": FORWARD_SCHEMA_VERSION,
        "provider": FORWARD_PROVIDER,
        "mode": mode,
        "status": status,
        "reason_code": reason_code,
        "nucleus": "13C",
        "model_called": model_called,
        "used_for_ranking": False,
        "diagnostic_only": True,
        "evidence_kind": "relative_13c_forward_evidence",
        "calibrated_probability": False,
        "quantile_enabled": False,
        "evaluated_candidate_count": evaluated_candidate_count,
        "skipped_candidate_count": skipped_candidate_count,
        "elapsed_ms": round(max(elapsed_ms, 0.0), 1),
        "model": model,
    }


def _forward_candidate_is_safe(smiles: str) -> tuple[bool, str | None]:
    if not smiles or len(smiles) > _FORWARD_MAX_SMILES_LENGTH:
        return False, "candidate_input_not_supported"
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            return False, "candidate_input_not_supported"
        if molecule.GetNumHeavyAtoms() > _FORWARD_MAX_HEAVY_ATOMS:
            return False, "candidate_too_large"
        if Chem.AddHs(molecule).GetNumAtoms() > _FORWARD_MAX_TOTAL_ATOMS:
            return False, "candidate_too_large"
        if (
            rdMolDescriptors.CalcNumRotatableBonds(molecule)
            > _FORWARD_MAX_ROTATABLE_BONDS
        ):
            return False, "candidate_too_flexible"
        if {
            atom.GetAtomicNum() for atom in molecule.GetAtoms()
        } - _FORWARD_SUPPORTED_ATOMIC_NUMBERS:
            return False, "candidate_elements_not_supported"
    except (RuntimeError, ValueError):
        return False, "candidate_input_not_supported"
    return True, None


def _run_forward_shadow(
    observed_13c: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    formula: str | None,
) -> dict[str, Any]:
    """Attach fail-open 13C forward diagnostics without changing ranking."""

    if not observed_13c:
        return _forward_summary(
            status="unsupported_modality",
            reason_code="observed_13c_required",
            model_called=False,
        )
    if not candidates:
        return _forward_summary(
            status="no_candidates",
            reason_code="no_ranked_candidates",
            model_called=False,
        )

    settings = _forward_settings()
    if not settings["valid"]:
        return _forward_summary(
            status="disabled",
            reason_code=settings["reason_code"],
            model_called=False,
            mode=settings["mode"],
        )
    if settings["mode"] != "shadow":
        return _forward_summary(
            status="disabled",
            reason_code="disabled_by_server",
            model_called=False,
            mode=settings["mode"],
        )
    if not dp5q_sidecar_is_configured():
        return _forward_summary(
            status="not_configured",
            reason_code="sidecar_not_configured",
            model_called=False,
        )

    maximum = settings["max_candidates"]
    selected = candidates[:maximum]
    skipped = candidates[maximum:]
    sidecar_candidates: list[dict[str, str]] = []
    by_candidate_id: dict[str, dict[str, Any]] = {}
    for position, candidate in enumerate(selected, start=1):
        candidate_id = f"ranked-{position}"
        smiles = str(candidate.get("smiles") or "")
        allowed, reason_code = _forward_candidate_is_safe(smiles)
        if not allowed:
            candidate["forward_evidence"] = {
                "schema_version": FORWARD_EVIDENCE_SCHEMA_VERSION,
                "status": "skipped",
                "reason_code": reason_code,
                "used_for_ranking": False,
                "diagnostic_only": True,
                "calibrated_probability": False,
                "quantile_enabled": False,
            }
            skipped.append(candidate)
            continue
        sidecar_candidates.append(
            {"candidate_id": candidate_id, "smiles": smiles}
        )
        by_candidate_id[candidate_id] = candidate
    for candidate in candidates[maximum:]:
        candidate["forward_evidence"] = {
            "schema_version": FORWARD_EVIDENCE_SCHEMA_VERSION,
            "status": "skipped",
            "reason_code": "candidate_limit",
            "used_for_ranking": False,
            "diagnostic_only": True,
            "calibrated_probability": False,
            "quantile_enabled": False,
        }

    skipped_count = len(skipped)
    if not sidecar_candidates:
        return _forward_summary(
            status="unavailable",
            reason_code="no_eligible_candidates",
            model_called=False,
            skipped_candidate_count=skipped_count,
        )
    acquired = _FORWARD_SEMAPHORE.acquire(
        timeout=settings["busy_timeout_seconds"]
    )
    if not acquired:
        return _forward_summary(
            status="busy",
            reason_code="concurrency_limit",
            model_called=False,
            skipped_candidate_count=skipped_count,
        )

    started = time.perf_counter()
    try:
        adapter = get_nmr_forward_adapter()
        result = adapter.score_candidates(
            observed_13c,
            sidecar_candidates,
            formula=formula,
        )
        for evidence in result["candidates"]:
            candidate = by_candidate_id[evidence["candidate_id"]]
            prediction = evidence["prediction"]
            candidate["forward_evidence"] = {
                "schema_version": FORWARD_EVIDENCE_SCHEMA_VERSION,
                "status": "evaluated",
                "reason_code": None,
                "used_for_ranking": False,
                "diagnostic_only": True,
                "nucleus": "13C",
                "evidence_kind": "relative_13c_forward_evidence",
                "relative_fit_rank": evidence["relative_rank"],
                "assignment_method": evidence["assignment_mode"],
                "matched_count": evidence["matched_count"],
                "observed_count": evidence["observed_count"],
                "predicted_carbon_count": evidence["predicted_atom_count"],
                "unmatched_observed_count": evidence[
                    "unmatched_observed_count"
                ],
                "unmatched_predicted_carbon_count": evidence[
                    "unmatched_predicted_atom_count"
                ],
                "observed_coverage": evidence["observed_coverage"],
                "predicted_coverage": evidence["predicted_atom_coverage"],
                "bidirectional_coverage": evidence[
                    "bidirectional_coverage"
                ],
                "assignment_complete": evidence["assignment_complete"],
                "mae_ppm": evidence["mae_ppm"],
                "rmse_ppm": evidence["rmse_ppm"],
                "max_abs_error_ppm": evidence["max_abs_error_ppm"],
                "calibrated_probability": False,
                "quantile_enabled": False,
                "prediction": {
                    "output": "boltzmann_weighted_mean_shift_ppm",
                    "canonical_smiles": prediction["canonical_smiles"],
                    "conformer_count": prediction["conformer_count"],
                    "atom_predictions": prediction["atom_predictions"],
                    "warnings": prediction["warnings"],
                },
            }
        summary = _forward_summary(
            status="partial" if skipped_count else "completed",
            reason_code=(
                "some_candidates_not_evaluated" if skipped_count else None
            ),
            model_called=True,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            evaluated_candidate_count=len(result["candidates"]),
            skipped_candidate_count=skipped_count,
            model=result["model"],
        )
        summary["rank_basis"] = result["rank_basis"]
        summary["assignment_limitation"] = result["assignment_limitation"]
        return summary
    except NMRForwardTimeoutError:
        return _forward_summary(
            status="timed_out",
            reason_code="sidecar_timeout",
            model_called=True,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            skipped_candidate_count=skipped_count,
        )
    except NMRForwardConfigurationError:
        return _forward_summary(
            status="unavailable",
            reason_code="sidecar_unavailable",
            model_called=False,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            skipped_candidate_count=skipped_count,
        )
    except (
        NMRForwardInputError,
        NMRForwardProtocolError,
        NMRForwardUnavailableError,
        NMRForwardError,
    ):
        return _forward_summary(
            status="unavailable",
            reason_code="sidecar_unavailable",
            model_called=True,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            skipped_candidate_count=skipped_count,
        )
    except Exception:
        logger.exception("Unexpected DP5q shadow-evaluation failure")
        return _forward_summary(
            status="unavailable",
            reason_code="sidecar_unavailable",
            model_called=True,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            skipped_candidate_count=skipped_count,
        )
    finally:
        _FORWARD_SEMAPHORE.release()


def _v2_index_path() -> Path:
    return nmr_index_v2_path()


def _v2_index_status() -> dict:
    path = _v2_index_path()
    if not path.is_file():
        return {
            "available": False,
            "path": str(path),
            "role": "audit_only_not_production",
        }
    conn = connect_readonly(path)
    try:
        version = schema_version(conn)
        total = int(conn.execute("SELECT COUNT(*) FROM spectra").fetchone()[0])
        peaks = int(conn.execute("SELECT COUNT(*) FROM peaks").fetchone()[0])
        molecules = int(conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0])
        by_kind = {
            str(row[0]): int(row[1])
            for row in conn.execute(
                "SELECT measurement_kind, COUNT(*) FROM spectra GROUP BY measurement_kind"
            )
        }
        by_nucleus = {
            str(row[0]): int(row[1])
            for row in conn.execute(
                "SELECT nucleus, COUNT(*) FROM spectra GROUP BY nucleus"
            )
        }
        snapshot = conn.execute(
            """
            SELECT source_name, source_version, sha256, license_uri, validation_json
            FROM source_snapshots
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        rejections = int(
            conn.execute("SELECT COUNT(*) FROM import_rejections").fetchone()[0]
        )
        build_options = json.loads(snapshot["validation_json"]).get(
            "build_options", {}
        )
        return {
            "available": True,
            "path": str(path),
            "schema_version": version,
            "molecules": molecules,
            "spectra": total,
            "peaks": peaks,
            "by_measurement_kind": by_kind,
            "by_nucleus": by_nucleus,
            "rejected_spectra": rejections,
            "snapshot": {
                "source_name": snapshot["source_name"],
                "source_version": snapshot["source_version"],
                "sha256": snapshot["sha256"],
                "license_uri": snapshot["license_uri"],
                "build_options": build_options,
            },
            "role": "audit_only_not_production",
            "production_eligible": False,
        }
    except Exception as exc:
        logger.exception("Unable to inspect NMR index v2")
        return {
            "available": False,
            "path": str(path),
            "role": "audit_only_not_production",
            "error": type(exc).__name__,
        }
    finally:
        # The read-only connection must be released even when an inspection
        # query fails halfway.
        conn.close()


def _load():
    global _model, _tokenizer, _device
    if _model is not None:
        return
    with _model_load_lock:
        if _model is not None:
            return
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = T5Tokenizer.from_pretrained(str(MODEL_DIR))
        _model = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR)).to(
            _device
        )
        _model.eval()


class PeakInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shift: float = Field(allow_inf_nan=False, ge=-20.0, le=1000.0)
    intensity: Optional[float] = Field(default=1.0, allow_inf_nan=False, ge=-1e15, le=1e15)
    integral: Optional[float] = Field(default=None, allow_inf_nan=False, ge=0, le=1e9)
    multiplicity: Annotated[str, StringConstraints(max_length=32)] | None = ""
    assignment: Annotated[str, StringConstraints(max_length=128)] | None = ""


ConstraintPattern = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
CandidateSmiles = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
]
CarbonShift = Annotated[
    float,
    Field(allow_inf_nan=False, ge=-20.0, le=300.0),
]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    peaks_13c: list[PeakInput] = Field(default_factory=list, max_length=1000)
    peaks_1h: list[PeakInput] = Field(default_factory=list, max_length=1000)
    formula: Annotated[str, StringConstraints(max_length=256)] | None = None
    solvent: Annotated[str, StringConstraints(max_length=64)] | None = None
    exclude_reference_peaks: bool = True
    cluster_1h_lines: bool = True
    analyze_mixture: bool = True
    generate_experimental: bool = False
    candidate_smiles: list[CandidateSmiles] = Field(
        default_factory=list,
        max_length=100,
    )
    required_smarts: list[ConstraintPattern] = Field(
        default_factory=list,
        max_length=20,
    )
    forbidden_smarts: list[ConstraintPattern] = Field(
        default_factory=list,
        max_length=20,
    )
    top_k: int = Field(default=5, ge=1, le=20)
    num_beams: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def require_peaks(self):
        if not self.peaks_13c and not self.peaks_1h:
            raise ValueError("At least one NMR peak is required")
        return self


class QuantileShadowRequest(BaseModel):
    """Bounded input for the opt-in, diagnostic-only quantile sidecar."""

    model_config = ConfigDict(extra="forbid")
    observed_shifts_ppm: list[CarbonShift] = Field(
        min_length=1,
        max_length=256,
    )
    candidate_smiles: list[CandidateSmiles] = Field(
        min_length=1,
        max_length=5,
    )
    formula: Annotated[str, StringConstraints(max_length=256)] | None = None


class ImportIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["quick", "retrieval_db_v2", "nmrshiftdb2", "nmrshiftdb2_full"] = "quick"
    limit: int | None = Field(default=None, ge=1, le=2_000_000)
    time_budget_s: float = Field(default=3600.0, allow_inf_nan=False, ge=1.0, le=7200.0)
    progress_every: int = Field(default=0, ge=0, le=1_000_000)
    quiet_rdkit: bool = True


class TrainRankerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit_records: int = Field(default=30000, ge=100, le=2_000_000)
    negatives_per_positive: int = Field(default=8, ge=1, le=64)
    seed: int = Field(default=20260711, ge=0, le=2**32 - 1)


class CombinedPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id1: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    id2: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    formula: Annotated[str, StringConstraints(max_length=256)] | None = None
    generate_experimental: bool = False
    candidate_smiles: list[CandidateSmiles] = Field(
        default_factory=list,
        max_length=100,
    )
    required_smarts: list[ConstraintPattern] = Field(
        default_factory=list,
        max_length=20,
    )
    forbidden_smarts: list[ConstraintPattern] = Field(
        default_factory=list,
        max_length=20,
    )
    top_k: int = Field(default=5, ge=1, le=20)
    num_beams: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def require_distinct_spectra(self):
        if self.id1 == self.id2:
            raise ValueError("Choose two different NMR spectra")
        return self


def _predict_from_peaks(
    p13: list[dict],
    p1h: list[dict],
    formula: str | None,
    top_k: int,
    num_beams: int,
) -> dict:
    import time
    if not p13 and not p1h:
        raise HTTPException(status_code=400, detail="No peaks provided")

    inp = build_generation_prompt(p13, p1h, formula=formula)

    t0 = time.time()
    enc = _tokenizer(inp, return_tensors="pt", truncation=True, max_length=128, padding="max_length").to(_device)
    generation_count = min(top_k, num_beams)
    with torch.no_grad():
        gen = _model.generate(
            **enc,
            max_length=128,
            num_beams=num_beams,
            num_return_sequences=generation_count,
            early_stopping=True,
        )

    decoded = [_tokenizer.decode(item, skip_special_tokens=True).strip() for item in gen]
    candidates, validation = validate_generated_smiles(decoded, formula=formula)
    return {
        "status": "completed",
        "inference_time_ms": round((time.time() - t0) * 1000, 1),
        "candidates": candidates,
        "validation": validation,
        "prompt_schema": "formula+nucleus+shift+integral+multiplicity-v1",
    }


def _safe_generate(
    p13: list[dict],
    p1h: list[dict],
    formula: str | None,
    top_k: int,
    num_beams: int,
) -> dict:
    try:
        _load()
        return _predict_from_peaks(p13, p1h, formula, top_k, num_beams)
    except Exception:
        logger.exception("NMR structure generation failed")
        return {
            "status": "generation_unavailable",
            "inference_time_ms": 0,
            "candidates": [],
            "error_code": "generation_failed",
        }


@router.get("/status")
def get_status():
    model_files_available = MODEL_DIR.is_dir() and (
        (MODEL_DIR / "model.safetensors").is_file()
        or (MODEL_DIR / "pytorch_model.bin").is_file()
        or (MODEL_DIR / "final" / "model.safetensors").is_file()
    )
    forward_settings = _forward_settings()
    quantile_settings = _quantile_shadow_settings()
    return {
        "model_loaded": _model is not None,
        "model_files_available": model_files_available,
        "device": str(_device or "not_loaded"),
        "model": HYBRID_SCHEMA_VERSION,
        "model_role": (
            "Constraint-aware candidate ranking with mandatory abstention; "
            "no structure-correctness probability is emitted."
        ),
        "hybrid_predictor": {
            "schema_version": HYBRID_SCHEMA_VERSION,
            "pipeline_version": HYBRID_PIPELINE_VERSION,
            "automatic_selection": False,
            "decision_policy": "always_abstain_until_independently_calibrated",
            "candidate_sources": [
                "formula-constrained_local_reference_index",
                "explicit_user_candidates",
                "opt_in_experimental_generator",
            ],
        },
        "experimental_generation": {
            "available": model_files_available,
            "enabled_by_default": False,
            "role": "opt-in hypothesis generation only; excluded from formal ranking",
        },
        "ranker_training": {
            "legacy_endpoint_mode": "diagnostic_only",
            "production_requirement": "independent_spectrum_scaffold_split_v2",
        },
        "spectral_index": index_status(),
        "spectral_index_v2": _v2_index_status(),
        "external_smoke_benchmark": _external_smoke_status(),
        "forward_model": {
            "schema_version": FORWARD_SCHEMA_VERSION,
            "provider": FORWARD_PROVIDER,
            "mode": forward_settings["mode"],
            "enabled": (
                forward_settings["valid"]
                and forward_settings["mode"] == "shadow"
            ),
            "configured": dp5q_sidecar_is_configured(),
            "configuration_valid": forward_settings["valid"],
            "reason_code": forward_settings["reason_code"],
            "nucleus": "13C",
            "used_for_ranking": False,
            "calibrated_probability": False,
            "quantile_enabled": False,
            "max_candidates": forward_settings.get("max_candidates"),
            "quantile_shadow": {
                "schema_version": QUANTILE_SHADOW_SCHEMA_VERSION,
                "provider": QUANTILE_SHADOW_PROVIDER,
                "mode": quantile_settings["mode"],
                "enabled": (
                    quantile_settings["valid"]
                    and quantile_settings["mode"] == "shadow"
                ),
                "configured": dp5q_quantile_sidecar_is_configured(),
                "configuration_valid": quantile_settings["valid"],
                "reason_code": quantile_settings["reason_code"],
                "asset_verification": "deferred_until_sidecar_start",
                "runtime_health": "not_probed_by_status_endpoint",
                "nucleus": "13C",
                "used_for_ranking": False,
                "diagnostic_only": True,
                "official_assignment_semantics": False,
                "official_workflow_parity": False,
                "calibrated_probability": False,
                "quantile_enabled": True,
                "max_candidates": quantile_settings.get("max_candidates"),
            },
        },
    }


def _compact_quantile_shadow_candidate(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if (
        candidate.get("score_semantics")
        != DP5Q_UNASSIGNED_SHADOW_SEMANTICS
        or candidate.get("assignment_mode")
        != "unassigned_hungarian_q50_atom_level"
        or candidate.get("official_assignment_semantics") is not False
        or candidate.get("official_equation_parity") is not True
        or candidate.get("official_workflow_parity") is not False
        or candidate.get("conformer_protocol")
        != "chemapp.dp5q-conformer.v2"
        or not isinstance(candidate.get("equation_details"), dict)
        or candidate["equation_details"].get("official_equation_parity")
        is not True
        or candidate["equation_details"].get("official_workflow_parity")
        is not False
        or candidate["equation_details"].get(
            "official_assignment_semantics"
        )
        is not False
        or candidate["equation_details"].get("conformer_protocol")
        != candidate.get("conformer_protocol")
        or candidate.get("diagnostic_only") is not True
        or candidate.get("used_for_ranking") is not False
        or candidate.get("calibrated_probability") is not False
        or candidate.get("quantile_enabled") is not True
    ):
        raise NMRForwardProtocolError(
            "Quantile candidate changed the shadow evidence contract."
        )
    prediction = candidate["prediction"]
    atoms = prediction["atom_predictions"]
    crossing_counts = [
        count
        for atom in atoms
        for count in atom["conformer_quantile_crossing_counts"]
    ]
    maximum_crossings = [
        crossing
        for atom in atoms
        for crossing in atom["conformer_max_quantile_crossing_ppm"]
    ]
    return {
        "candidate_id": candidate["candidate_id"],
        "canonical_smiles": prediction["canonical_smiles"],
        "relative_rank": candidate["relative_rank"],
        "score_semantics": candidate["score_semantics"],
        "assignment_mode": candidate["assignment_mode"],
        "official_assignment_semantics": False,
        "official_equation_parity": True,
        "official_workflow_parity": False,
        "conformer_protocol": candidate["conformer_protocol"],
        "diagnostic_only": True,
        "used_for_ranking": False,
        "matched_count": candidate["matched_count"],
        "observed_count": candidate["observed_count"],
        "predicted_atom_count": candidate["predicted_atom_count"],
        "unmatched_observed_count": candidate["unmatched_observed_count"],
        "unmatched_predicted_atom_count": candidate[
            "unmatched_predicted_atom_count"
        ],
        "bidirectional_coverage": candidate["bidirectional_coverage"],
        "assignment_complete": candidate["assignment_complete"],
        "q50_assignment_mae_ppm": candidate[
            "q50_assignment_mae_ppm"
        ],
        "dp5q_shadow_score": candidate["dp5q_shadow_score"],
        "calibrated_probability": False,
        "quantile_enabled": True,
        "prediction_summary": {
            "conformer_count": prediction["conformer_count"],
            "predicted_carbon_count": len(atoms),
            "quantile_levels": 99,
            "quantile_crossing_count": sum(crossing_counts),
            "max_quantile_crossing_ppm": max(
                maximum_crossings,
                default=0.0,
            ),
            "warnings": prediction["warnings"],
        },
    }


@router.post(
    "/dp5q/quantile-shadow",
    dependencies=[Depends(require_admin)],
)
def evaluate_dp5q_quantile_shadow(request: QuantileShadowRequest):
    """Run bounded, non-ranking DP5q quantile diagnostics on explicit candidates."""

    settings = _quantile_shadow_settings()
    if not settings["valid"]:
        raise HTTPException(
            status_code=503,
            detail="DP5q quantile shadow has invalid server configuration",
        )
    if settings["mode"] != "shadow":
        raise HTTPException(
            status_code=503,
            detail="DP5q quantile shadow is disabled",
        )
    if not dp5q_quantile_sidecar_is_configured():
        raise HTTPException(
            status_code=503,
            detail="DP5q quantile sidecar is not configured",
        )
    if len(request.candidate_smiles) > settings["max_candidates"]:
        raise HTTPException(
            status_code=422,
            detail="Candidate count exceeds the configured quantile limit",
        )
    try:
        formula_info = parse_formula(request.formula)
    except FormulaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for smiles in request.candidate_smiles:
        allowed, _reason = _forward_candidate_is_safe(smiles)
        if not allowed:
            raise HTTPException(
                status_code=422,
                detail="A candidate is outside the supported model contract",
            )
    acquired = _FORWARD_SEMAPHORE.acquire(
        timeout=settings["busy_timeout_seconds"]
    )
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="DP5q model capacity is currently busy",
        )

    started = time.perf_counter()
    try:
        adapter = get_nmr_quantile_forward_adapter()
        result = adapter.score_candidates(
            request.observed_shifts_ppm,
            [
                {
                    "candidate_id": f"quantile-{position}",
                    "smiles": smiles,
                }
                for position, smiles in enumerate(
                    request.candidate_smiles,
                    start=1,
                )
            ],
            formula=formula_info.canonical if formula_info else None,
        )
        if (
            result.get("status") != "ok"
            or result.get("evidence_kind")
            != DP5Q_UNASSIGNED_SHADOW_SEMANTICS
            or result.get("official_assignment_semantics") is not False
            or result.get("official_equation_parity") is not True
            or result.get("official_workflow_parity") is not False
            or result.get("conformer_protocol")
            != "chemapp.dp5q-conformer.v2"
            or result.get("used_for_ranking") is not False
            or result.get("calibrated_probability") is not False
            or result.get("quantile_enabled") is not True
        ):
            raise NMRForwardProtocolError(
                "Quantile adapter changed the shadow evidence contract."
            )
        return {
            "schema_version": QUANTILE_SHADOW_SCHEMA_VERSION,
            "provider": QUANTILE_SHADOW_PROVIDER,
            "status": "completed",
            "mode": "shadow",
            "nucleus": "13C",
            "model_called": True,
            "used_for_ranking": False,
            "diagnostic_only": True,
            "official_assignment_semantics": False,
            "official_equation_parity": True,
            "official_workflow_parity": False,
            "conformer_protocol": result["conformer_protocol"],
            "calibrated_probability": False,
            "quantile_enabled": True,
            "elapsed_ms": round(
                (time.perf_counter() - started) * 1000,
                1,
            ),
            "rank_basis": result["rank_basis"],
            "assignment_limitation": result["assignment_limitation"],
            "model": result["model"],
            "runtime": result["runtime"],
            "candidates": [
                _compact_quantile_shadow_candidate(candidate)
                for candidate in result["candidates"]
            ],
        }
    except NMRForwardInputError as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid quantile shadow candidate or formula input",
        ) from exc
    except NMRForwardTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="DP5q quantile sidecar timed out",
        ) from exc
    except NMRForwardProtocolError as exc:
        raise HTTPException(
            status_code=502,
            detail="DP5q quantile sidecar returned an invalid response",
        ) from exc
    except (
        NMRForwardConfigurationError,
        NMRForwardUnavailableError,
        NMRForwardError,
    ) as exc:
        raise HTTPException(
            status_code=503,
            detail="DP5q quantile sidecar is unavailable",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected DP5q quantile-shadow failure")
        raise HTTPException(
            status_code=503,
            detail="DP5q quantile sidecar is unavailable",
        ) from exc
    finally:
        _FORWARD_SEMAPHORE.release()


@router.post("/index/import", dependencies=[Depends(require_admin)])
def import_index(payload: ImportIndexRequest | None = None):
    payload = payload or ImportIndexRequest()
    source = payload.source
    if source == "nmrshiftdb2":
        return import_nmrshiftdb_sd(
            path=None,
            limit=payload.limit,
            time_budget_s=payload.time_budget_s,
            progress_every=payload.progress_every,
            quiet_rdkit=payload.quiet_rdkit,
        )
    if source == "nmrshiftdb2_full":
        return import_nmrshiftdb_sd(
            path=None,
            limit=None,
            time_budget_s=payload.time_budget_s,
            progress_every=payload.progress_every,
            quiet_rdkit=payload.quiet_rdkit,
        )
    if source in {"quick", "retrieval_db_v2"}:
        if source == "quick":
            return ensure_quick_index()
        return import_retrieval_db(path=None, limit=payload.limit)
    raise HTTPException(400, "source must be quick, retrieval_db_v2, or nmrshiftdb2")


@router.post("/ranker/train", dependencies=[Depends(require_admin)])
def train_ranker(payload: TrainRankerRequest | None = None):
    payload = payload or TrainRankerRequest()
    from app.api.routes.ml import _acquire_training_lock

    # Bound the in-memory job table before registering a new job.
    _prune_ranker_jobs()
    job_id = f"ranker-{uuid.uuid4().hex}"
    lock_path = _acquire_training_lock(job_id)
    if lock_path is None:
        raise HTTPException(409, "Another model training job is already running")
    with _ranker_jobs_lock:
        _ranker_jobs[job_id] = {
            "status": "started",
            "job_id": job_id,
            "started_at": time.time(),
        }
    thread = threading.Thread(
        target=_run_ranker_job,
        args=(job_id, payload, lock_path),
        daemon=True,
    )
    thread.start()
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "started"},
    )


@router.get(
    "/ranker/train/status/{job_id}",
    dependencies=[Depends(require_admin)],
)
def ranker_train_status(job_id: str):
    _prune_ranker_jobs()
    with _ranker_jobs_lock:
        job = _ranker_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown ranker training job")
    return job


@router.post("/predict")
def predict_structure(request: PredictRequest):
    import time
    started = time.time()
    try:
        formula_info = parse_formula(request.formula)
    except FormulaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    raw_p13 = [peak.model_dump() for peak in request.peaks_13c]
    raw_p1h = [peak.model_dump() for peak in request.peaks_1h]
    p13, p13_audit = prepare_query_peaks(
        raw_p13,
        nucleus="13C",
        solvent=request.solvent,
        exclude_references=request.exclude_reference_peaks,
        cluster_1h_lines=False,
    )
    p1h, p1h_audit = prepare_query_peaks(
        raw_p1h,
        nucleus="1H",
        solvent=request.solvent,
        exclude_references=request.exclude_reference_peaks,
        cluster_1h_lines=request.cluster_1h_lines,
    )
    if not p13 and not p1h:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No analyte peaks remain after validation and reference filtering",
                "preprocessing": {"13c": p13_audit, "1h": p1h_audit},
            },
        )

    formula = formula_info.canonical if formula_info else None
    if request.generate_experimental:
        generation_count = min(max(request.top_k, 5), request.num_beams)
        generated = _safe_generate(
            p13,
            p1h,
            formula,
            generation_count,
            request.num_beams,
        )
    else:
        generated = {
            "status": "not_requested",
            "inference_time_ms": 0,
            "candidates": [],
            "reason": "Experimental T5 generation is disabled by default.",
        }
    generated_smiles = [c["smiles"] for c in generated.get("candidates", []) if c.get("smiles")]
    forward_scorer = _cached_forward_scorer()
    candidate_providers = _cached_candidate_providers()
    try:
        hybrid_result = hybrid_rank_candidates(
            peaks_13c=p13,
            peaks_1h=p1h,
            formula=formula,
            solvent=request.solvent,
            constraints=CandidateConstraints(
                required_smarts=tuple(request.required_smarts),
                forbidden_smarts=tuple(request.forbidden_smarts),
            ),
            generated_smiles=generated_smiles,
            candidate_smiles=request.candidate_smiles,
            top_k=request.top_k,
            reference_ranker=rank_candidates,
            forward_scorer=forward_scorer,
            candidate_providers=candidate_providers,
        )
        ranked = hybrid_to_legacy_rank_result(hybrid_result)
    except HybridPredictorInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    forward_model = _run_forward_shadow(
        p13,
        ranked["candidates"],
        formula,
    )
    mixture = (
        analyze_mixture(p13, p1h, formula=formula, top_k=min(request.top_k, 5))
        if request.analyze_mixture
        else {"status": "not_requested", "is_mixture": None, "components": []}
    )
    preprocessing_warnings = []
    excluded_count = len(p13_audit["excluded"]) + len(p1h_audit["excluded"])
    if excluded_count:
        preprocessing_warnings.append(
            f"Excluded {excluded_count} reference/solvent peak(s) before ranking."
        )
    if p1h_audit["clustered"] and p1h_audit["received"] != p1h_audit["retained"]:
        preprocessing_warnings.append(
            f"Collapsed {p1h_audit['received']} 1H lines to "
            f"{p1h_audit['retained']} resonance groups."
        )
    index_details = ranked.get("index") or {}
    by_source = index_details.get("by_source", {})
    data_attribution = []
    if any(str(source).lower().startswith("nmrshiftdb2") for source in by_source):
        data_attribution.append(
            {
                "source": "nmrshiftdb2",
                "notice": (
                    "Contains information from nmrshiftdb2 "
                    "(www.nmrshiftdb.org), made available under the "
                    "nmrshiftdb2 Database License."
                ),
                "license_uri": NMRSHIFTDB2_LICENSE_URI,
            }
        )
    return {
        "status": "candidate_ranking" if ranked["candidates"] else "insufficient_reference_coverage",
        "inference_time_ms": round((time.time() - started) * 1000, 1),
        "method": (
            "formula-constrained spectral retrieval within hybrid-v1 candidate "
            "generation, one-to-one assignment, conservative ranking, and "
            "mandatory abstention"
        ),
        "result_type": "candidate_ranking_not_identification",
        "schema_version": hybrid_result["schema_version"],
        "pipeline_version": hybrid_result["pipeline_version"],
        "decision": hybrid_result["decision"],
        "evidence_level": hybrid_result["evidence_state"],
        "uncertainty_kind": hybrid_result["uncertainty_kind"],
        "calibrated_probability": hybrid_result["calibrated_probability"],
        "top1_calibrated_probability": hybrid_result.get(
            "top1_calibrated_probability"
        ),
        "pipeline": {
            "stages": hybrid_result["stages"],
            "reference_context": hybrid_result["reference_context"],
            "forward_context": hybrid_result["forward_context"],
        },
        "literature_basis": [
            "Candidate generation and database retrieval must be verified by forward spectrum prediction and independent NMR evidence.",
            "Molecular formula is enforced as a hard constraint when supplied.",
            "Optional T5 outputs are experimental hypotheses only and do not change retrieval ranking.",
        ],
        "candidates": ranked["candidates"],
        "generated_candidates": generated.get("candidates", []),
        "generation": {
            key: value
            for key, value in generated.items()
            if key != "candidates"
        },
        "mixture_analysis": mixture,
        "query": {
            **ranked["query"],
            "preprocessing": {"13c": p13_audit, "1h": p1h_audit},
        },
        "warnings": [*preprocessing_warnings, *ranked.get("warnings", [])],
        "candidate_pool_status": ranked["candidate_pool_status"],
        "ranker": ranked["ranker"],
        "forward_model": forward_model,
        "index": index_details,
        "data_attribution": data_attribution,
    }


@router.post("/predict/combined")
def predict_structure_combined(request: CombinedPredictRequest):
    """Combined prediction — user selects 2 spectrum IDs manually."""
    from app.api.deps import get_store

    store = get_store()
    p13: list[PeakInput] = []
    p1h: list[PeakInput] = []
    solvent = ""
    seen_nuclei: set[str] = set()

    for sid in (request.id1, request.id2):
        s = store.get(sid)
        if s is None:
            raise HTTPException(status_code=404, detail=f"Spectrum not found: {sid}")
        if s.spectrum.technique != Technique.NMR:
            raise HTTPException(status_code=422, detail=f"Spectrum is not NMR: {sid}")
        if s.result is None:
            raise HTTPException(
                status_code=409,
                detail=f"Analyze spectrum before combined prediction: {sid}",
            )
        nucleus = s.spectrum.parameters.get("nucleus", "")
        nucleus = "13C" if nucleus == "13C" else "1H"
        if nucleus in seen_nuclei:
            raise HTTPException(
                status_code=422,
                detail="Combined prediction requires one 1H and one 13C spectrum",
            )
        seen_nuclei.add(nucleus)
        solvent = solvent or s.spectrum.metadata.solvent
        if nucleus == "1H" and getattr(s.result, "multiplets", None):
            p1h = [
                PeakInput(
                    shift=float(item["center_ppm"]),
                    intensity=float(item.get("intensity_max") or 1.0),
                    integral=item.get("relative_area"),
                    multiplicity=item.get("multiplicity", ""),
                    assignment=item.get("assignment", ""),
                )
                for item in s.result.multiplets
                if item.get("center_ppm") is not None
            ]
        else:
            peaks = [
                PeakInput(
                    shift=float(peak.position),
                    intensity=float(peak.intensity),
                    integral=peak.area,
                    multiplicity=peak.multiplicity,
                    assignment=peak.assignment,
                )
                for peak in s.result.peaks
            ]
            if nucleus == "13C":
                p13 = peaks
            else:
                p1h = peaks

    request2 = PredictRequest(
        peaks_13c=p13,
        peaks_1h=p1h,
        formula=request.formula,
        generate_experimental=request.generate_experimental,
        candidate_smiles=request.candidate_smiles,
        required_smarts=request.required_smarts,
        forbidden_smarts=request.forbidden_smarts,
        solvent=solvent or None,
        cluster_1h_lines=False,
        top_k=request.top_k,
        num_beams=request.num_beams,
    )
    return predict_structure(request2)
