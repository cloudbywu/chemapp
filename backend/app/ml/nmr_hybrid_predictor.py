"""Conservative hybrid NMR candidate ranking pipeline.

The v1 pipeline composes the existing exact-formula spectral-library search
with optional candidate providers and an optional 13C forward predictor.  It
deliberately separates relative ranking evidence from a structure-correctness
probability: v1 has no frozen calibration model and therefore always abstains
from automatic structure selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except Exception:  # pragma: no cover - backend dependency checked at runtime
    Chem = None
    Descriptors = None
    rdMolDescriptors = None

from app.ml.nmr_evidence import FormulaInfo, canonical_formula, parse_formula
from app.ml.nmr_structure_elucidation import rank_candidates


SCHEMA_VERSION = "chemapp.nmr.hybrid-prediction.v1"
PIPELINE_VERSION = "1.0.0"
EVIDENCE_STATES = frozenset({"no_reference", "no_match", "weak"})
UNCERTAINTY_KINDS = frozenset(
    {
        "unavailable",
        "fixed_tolerance_only",
        "point_prediction_only",
        "predictive_interval_uncalibrated",
    }
)


class HybridPredictorError(RuntimeError):
    """Base error raised by the hybrid predictor."""


class HybridPredictorInputError(HybridPredictorError, ValueError):
    """Raised when a query or structural constraint is invalid."""


class CandidateProvider(Protocol):
    """Pluggable source of formula-constrained candidate structures."""

    provider_id: str

    def generate(
        self,
        *,
        formula: str | None,
        constraints: Mapping[str, Any],
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Return candidate mappings containing at least a ``smiles`` value."""


class ForwardScorer(Protocol):
    """Minimal interface implemented by :class:`NMRForwardAdapter`."""

    def score_candidates(
        self,
        observed_13c: Sequence[float | Mapping[str, Any]] | None,
        candidates: Sequence[Mapping[str, Any]],
        *,
        formula: str | None = None,
    ) -> Mapping[str, Any]:
        """Return relative 13C forward-fit evidence."""


ReferenceRanker = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class CandidateConstraints:
    """Hard structure filters applied after candidate generation."""

    required_smarts: tuple[str, ...] = ()
    forbidden_smarts: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_smarts": list(self.required_smarts),
            "forbidden_smarts": list(self.forbidden_smarts),
        }


@dataclass(frozen=True)
class HybridPredictorConfig:
    """Resource and result limits for the v1 pipeline."""

    candidate_pool_limit: int = 100
    result_limit: int = 10
    reference_record_limit: int = 30000
    forward_candidate_limit: int = 20
    supplied_candidate_limit: int = 100

    def __post_init__(self) -> None:
        for name in (
            "candidate_pool_limit",
            "result_limit",
            "reference_record_limit",
            "forward_candidate_limit",
            "supplied_candidate_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise HybridPredictorInputError(f"{name} must be a positive integer")
        if self.result_limit > self.candidate_pool_limit:
            raise HybridPredictorInputError(
                "result_limit cannot exceed candidate_pool_limit"
            )


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    numeric = _finite_float(value)
    if numeric is None or numeric < 0 or not numeric.is_integer():
        return None
    return int(numeric)


def _candidate_id(smiles: str) -> str:
    digest = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
    return f"candidate-{digest[:24]}"


def _compile_constraints(
    constraints: CandidateConstraints,
) -> tuple[list[Any], list[Any]]:
    if Chem is None:
        if constraints.required_smarts or constraints.forbidden_smarts:
            raise HybridPredictorInputError(
                "RDKit is required to apply substructure constraints"
            )
        return [], []

    required = []
    forbidden = []
    for kind, values, destination in (
        ("required", constraints.required_smarts, required),
        ("forbidden", constraints.forbidden_smarts, forbidden),
    ):
        for position, smarts in enumerate(values):
            if not isinstance(smarts, str) or not smarts.strip():
                raise HybridPredictorInputError(
                    f"{kind}_smarts[{position}] must be a non-empty string"
                )
            query = Chem.MolFromSmarts(smarts)
            if query is None:
                raise HybridPredictorInputError(
                    f"{kind}_smarts[{position}] is invalid"
                )
            destination.append(query)
    return required, forbidden


def _normalise_candidate(
    value: Mapping[str, Any],
    *,
    formula_info: FormulaInfo | None,
    required_queries: Sequence[Any],
    forbidden_queries: Sequence[Any],
) -> tuple[dict[str, Any] | None, str | None]:
    raw_smiles = str(value.get("smiles") or "").strip()
    if not raw_smiles:
        return None, "missing_smiles"
    if len(raw_smiles) > 4096:
        return None, "smiles_too_long"
    if Chem is None or Descriptors is None or rdMolDescriptors is None:
        raise HybridPredictorInputError(
            "RDKit is required for candidate identity and formula validation"
        )
    molecule = Chem.MolFromSmiles(raw_smiles)
    if molecule is None:
        return None, "invalid_smiles"
    if len(Chem.GetMolFrags(molecule)) != 1:
        return None, "disconnected_structure"

    candidate_smiles = Chem.MolToSmiles(molecule)
    try:
        candidate_formula = canonical_formula(
            rdMolDescriptors.CalcMolFormula(molecule)
        )
    except ValueError:
        return None, "unsupported_candidate_formula"
    if formula_info is not None and candidate_formula != formula_info.canonical:
        return None, "formula_mismatch"
    if any(not molecule.HasSubstructMatch(query) for query in required_queries):
        return None, "required_substructure_missing"
    if any(molecule.HasSubstructMatch(query) for query in forbidden_queries):
        return None, "forbidden_substructure_present"

    return (
        {
            "candidate_id": _candidate_id(candidate_smiles),
            "smiles": candidate_smiles,
            "molecular_formula": candidate_formula,
            "molecular_weight": round(float(Descriptors.MolWt(molecule)), 3),
            "compound_name": str(value.get("compound_name") or ""),
            "provenance": [],
            "_reference": value.get("_reference"),
        },
        None,
    )


def _provenance_key(provenance: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(provenance.get("kind") or ""),
        str(provenance.get("provider") or ""),
        str(provenance.get("source") or ""),
        str(provenance.get("source_id") or ""),
    )


def _add_candidate(
    destination: dict[str, dict[str, Any]],
    value: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    formula_info: FormulaInfo | None,
    required_queries: Sequence[Any],
    forbidden_queries: Sequence[Any],
) -> str | None:
    candidate, rejection = _normalise_candidate(
        value,
        formula_info=formula_info,
        required_queries=required_queries,
        forbidden_queries=forbidden_queries,
    )
    if candidate is None:
        return rejection

    smiles = candidate["smiles"]
    existing = destination.get(smiles)
    if existing is None:
        destination[smiles] = candidate
        existing = candidate
    elif existing.get("_reference") is None and value.get("_reference") is not None:
        existing["_reference"] = value["_reference"]
    if not existing.get("compound_name") and candidate.get("compound_name"):
        existing["compound_name"] = candidate["compound_name"]

    safe_provenance = {
        "kind": str(provenance.get("kind") or "unknown"),
        "provider": str(provenance.get("provider") or "unknown"),
        "source": str(provenance.get("source") or ""),
        "source_id": str(provenance.get("source_id") or ""),
    }
    keys = {_provenance_key(item) for item in existing["provenance"]}
    if _provenance_key(safe_provenance) not in keys:
        existing["provenance"].append(safe_provenance)
    return None


def _reference_metrics(
    candidate: Mapping[str, Any],
    *,
    has_13c: bool,
    has_1h: bool,
) -> dict[str, Any]:
    reference = candidate.get("_reference")
    if not isinstance(reference, Mapping):
        return {
            "status": "no_reference",
            "source_evidence_level": "no_reference",
            "relative_score": None,
            "score_breakdown": {},
            "matched_13c": 0,
            "matched_1h": 0,
            "calibrated_probability": False,
        }
    breakdown = reference.get("score_breakdown")
    if not isinstance(breakdown, Mapping):
        breakdown = {}
    c13 = breakdown.get("c13")
    h1 = breakdown.get("h1")
    c13 = c13 if isinstance(c13, Mapping) else {}
    h1 = h1 if isinstance(h1, Mapping) else {}
    matched_13c = int(
        _finite_float(
            c13.get("matched", reference.get("matched_13c")),
            0.0,
        )
        or 0
    )
    matched_1h = int(
        _finite_float(
            h1.get("matched", reference.get("matched_1h")),
            0.0,
        )
        or 0
    )
    observed_components = [
        component
        for enabled, component in ((has_13c, c13), (has_1h, h1))
        if enabled
    ]
    reference_counts = [
        int(_finite_float(item.get("reference_count"), 0.0) or 0)
        for item in observed_components
    ]
    observed_matched = (
        (matched_13c if has_13c else 0)
        + (matched_1h if has_1h else 0)
    )
    if observed_matched > 0:
        status = "weak"
    elif any(reference_counts):
        status = "no_match"
    else:
        status = "no_reference"
    return {
        "status": status,
        "source_evidence_level": str(
            reference.get("evidence_level") or status
        ),
        "relative_score": _finite_float(reference.get("ranking_score")),
        "score_breakdown": dict(breakdown),
        "matched_13c": matched_13c,
        "matched_1h": matched_1h,
        "calibrated_probability": False,
    }


def _normalise_forward_candidate(
    value: Mapping[str, Any],
    *,
    quantile_enabled: bool,
) -> dict[str, Any] | None:
    if value.get("prediction_failed") is True:
        return None
    relative_rank = _nonnegative_int(value.get("relative_rank"))
    matched = _nonnegative_int(value.get("matched_count"))
    observed_count = _nonnegative_int(value.get("observed_count"))
    predicted_count = _nonnegative_int(value.get("predicted_atom_count"))
    unmatched_observed = _nonnegative_int(
        value.get("unmatched_observed_count")
    )
    unmatched_predicted = _nonnegative_int(
        value.get("unmatched_predicted_atom_count")
    )
    coverage = _finite_float(value.get("bidirectional_coverage"))
    mae = _finite_float(value.get("mae_ppm"))
    rmse = _finite_float(value.get("rmse_ppm"))
    maximum_error = _finite_float(value.get("max_abs_error_ppm"))
    if (
        relative_rank is None
        or relative_rank < 1
        or matched is None
        or observed_count is None
        or predicted_count is None
        or unmatched_observed is None
        or unmatched_predicted is None
        or coverage is None
        or not 0.0 <= coverage <= 1.0
        or matched > min(observed_count, predicted_count)
        or unmatched_observed != observed_count - matched
        or unmatched_predicted != predicted_count - matched
        or (matched > 0 and mae is None)
        or any(
            metric is not None and metric < 0.0
            for metric in (mae, rmse, maximum_error)
        )
    ):
        return None
    status = "weak" if matched > 0 else "no_match"
    return {
        "status": status,
        "relative_rank": relative_rank,
        "prediction_failed": False,
        "assignment_mode": str(value.get("assignment_mode") or ""),
        "matched_count": matched,
        "observed_count": observed_count,
        "predicted_atom_count": predicted_count,
        "unmatched_observed_count": unmatched_observed,
        "unmatched_predicted_atom_count": unmatched_predicted,
        "bidirectional_coverage": round(coverage, 6),
        "mae_ppm": mae,
        "rmse_ppm": rmse,
        "max_abs_error_ppm": maximum_error,
        "diagnostic_only": True,
        "quantile_enabled": quantile_enabled,
        "calibrated_probability": False,
    }


def _candidate_evidence_state(
    reference: Mapping[str, Any],
    forward: Mapping[str, Any] | None,
) -> str:
    states = {str(reference.get("status") or "no_reference")}
    if forward is not None:
        states.add(str(forward.get("status") or "no_reference"))
    if "weak" in states:
        return "weak"
    if "no_match" in states:
        return "no_match"
    return "no_reference"


def _uncertainty_kind(
    reference: Mapping[str, Any],
    forward: Mapping[str, Any] | None,
) -> str:
    if forward is not None:
        if forward.get("quantile_enabled") is True:
            return "predictive_interval_uncalibrated"
        return "point_prediction_only"
    if reference.get("status") != "no_reference":
        return "fixed_tolerance_only"
    return "unavailable"


def _reference_component(
    reference: Mapping[str, Any],
    nucleus_key: str,
) -> float:
    breakdown = reference.get("score_breakdown")
    if not isinstance(breakdown, Mapping):
        return 0.0
    component = breakdown.get(nucleus_key)
    if not isinstance(component, Mapping):
        return 0.0
    return _finite_float(component.get("score"), 0.0) or 0.0


def _rank_key(
    candidate: Mapping[str, Any],
    *,
    has_13c: bool,
    use_forward: bool,
) -> tuple[Any, ...]:
    reference = candidate["_reference_evidence"]
    forward = candidate.get("_forward_evidence")
    reference_13c = _reference_component(reference, "c13")
    reference_1h = _reference_component(reference, "h1")
    legacy_reference_rank = int(candidate.get("_legacy_reference_rank") or 10**9)
    if has_13c and use_forward and isinstance(forward, Mapping):
        forward_rank = int(forward.get("relative_rank") or 10**9)
        coverage = _finite_float(forward.get("bidirectional_coverage"), 0.0) or 0.0
        unmatched = int(forward.get("unmatched_observed_count") or 0) + int(
            forward.get("unmatched_predicted_atom_count") or 0
        )
        mae = _finite_float(forward.get("mae_ppm"), float("inf"))
        return (
            0,
            forward_rank,
            -coverage,
            unmatched,
            mae,
            -reference_13c,
            -reference_1h,
            legacy_reference_rank,
            str(candidate["smiles"]),
        )
    if has_13c:
        return (
            1,
            -reference_13c,
            -reference_1h,
            legacy_reference_rank,
            str(candidate["smiles"]),
        )
    return (
        1,
        -reference_1h,
        legacy_reference_rank,
        str(candidate["smiles"]),
    )


class HybridPredictorV1:
    """Run candidate generation, evidence scoring, ranking, and abstention."""

    def __init__(
        self,
        *,
        reference_ranker: ReferenceRanker = rank_candidates,
        candidate_providers: Sequence[CandidateProvider] = (),
        forward_scorer: ForwardScorer | None = None,
        config: HybridPredictorConfig | None = None,
    ) -> None:
        self.reference_ranker = reference_ranker
        self.candidate_providers = tuple(candidate_providers)
        self.forward_scorer = forward_scorer
        self.config = config or HybridPredictorConfig()

    def predict(
        self,
        *,
        peaks_13c: Sequence[Mapping[str, Any]] | None = None,
        peaks_1h: Sequence[Mapping[str, Any]] | None = None,
        formula: str | None = None,
        solvent: str | None = None,
        constraints: CandidateConstraints | None = None,
        generated_smiles: Sequence[str] | None = None,
        candidate_smiles: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        constraints = constraints or CandidateConstraints()
        result_limit = self.config.result_limit if top_k is None else top_k
        if (
            isinstance(result_limit, bool)
            or not isinstance(result_limit, int)
            or not 1 <= result_limit <= self.config.candidate_pool_limit
        ):
            raise HybridPredictorInputError(
                "top_k must be between 1 and candidate_pool_limit"
            )
        formula_info = parse_formula(formula)
        required_queries, forbidden_queries = _compile_constraints(constraints)
        q13 = _peak_shifts(peaks_13c, nucleus="13C")
        q1h = _peak_shifts(peaks_1h, nucleus="1H")
        if not q13 and not q1h:
            raise HybridPredictorInputError("No finite analyte NMR peaks provided")
        if len(candidate_smiles or ()) > self.config.supplied_candidate_limit:
            raise HybridPredictorInputError("Too many supplied candidate SMILES")
        if len(generated_smiles or ()) > self.config.supplied_candidate_limit:
            raise HybridPredictorInputError("Too many generated candidate SMILES")

        modality = "1h+13c" if q13 and q1h else "13c" if q13 else "1h"
        input_stage = {
            "stage": "input_validation",
            "status": "completed",
            "formula_constraint": (
                formula_info.canonical if formula_info is not None else None
            ),
            "modality": modality,
            "n_13c": len(q13),
            "n_1h": len(q1h),
            "hard_constraints_applied": bool(
                formula_info
                or constraints.required_smarts
                or constraints.forbidden_smarts
            ),
        }

        legacy_result: Mapping[str, Any] = {}
        reference_error = False
        try:
            legacy_result = self.reference_ranker(
                peaks_13c=[{"shift": shift} for shift in q13],
                peaks_1h=[{"shift": shift} for shift in q1h],
                generated_smiles=None,
                formula=formula_info.canonical if formula_info else None,
                top_k=self.config.candidate_pool_limit,
                max_records=self.config.reference_record_limit,
            )
        except Exception:
            reference_error = True
            legacy_result = {}

        reference_candidates = legacy_result.get("candidates", [])
        if not isinstance(reference_candidates, Sequence) or isinstance(
            reference_candidates, (str, bytes)
        ):
            reference_candidates = []

        candidates: dict[str, dict[str, Any]] = {}
        rejection_counts: dict[str, int] = {}

        def add(
            value: Mapping[str, Any],
            provenance: Mapping[str, Any],
        ) -> None:
            rejection = _add_candidate(
                candidates,
                value,
                provenance=provenance,
                formula_info=formula_info,
                required_queries=required_queries,
                forbidden_queries=forbidden_queries,
            )
            if rejection:
                rejection_counts[rejection] = rejection_counts.get(rejection, 0) + 1

        for position, smiles in enumerate(candidate_smiles or ()):
            add(
                {"smiles": smiles},
                {
                    "kind": "provided",
                    "provider": "request",
                    "source": "user_provided_candidate",
                    "source_id": str(position),
                },
            )
        for position, smiles in enumerate(generated_smiles or ()):
            add(
                {"smiles": smiles},
                {
                    "kind": "generated",
                    "provider": "upstream_generator",
                    "source": "unverified_generated_candidate",
                    "source_id": str(position),
                },
            )

        provider_failures: list[str] = []
        for provider in self.candidate_providers:
            provider_id = str(
                getattr(provider, "provider_id", provider.__class__.__name__)
            )
            try:
                provided = provider.generate(
                    formula=formula_info.canonical if formula_info else None,
                    constraints=constraints.as_dict(),
                    limit=self.config.candidate_pool_limit,
                )
            except Exception:
                provider_failures.append(provider_id)
                continue
            if not isinstance(provided, Sequence) or isinstance(
                provided, (str, bytes)
            ):
                provider_failures.append(provider_id)
                continue
            for position, value in enumerate(provided):
                if position >= self.config.candidate_pool_limit:
                    break
                if not isinstance(value, Mapping):
                    rejection_counts["invalid_provider_candidate"] = (
                        rejection_counts.get("invalid_provider_candidate", 0) + 1
                    )
                    continue
                add(
                    value,
                    {
                        "kind": "provider",
                        "provider": provider_id,
                        "source": str(value.get("source") or provider_id),
                        "source_id": str(value.get("source_id") or position),
                    },
                )

        for position, value in enumerate(reference_candidates):
            if position >= self.config.candidate_pool_limit:
                break
            if not isinstance(value, Mapping):
                rejection_counts["invalid_reference_candidate"] = (
                    rejection_counts.get("invalid_reference_candidate", 0) + 1
                )
                continue
            add(
                {**value, "_reference": value},
                {
                    "kind": "spectral_library",
                    "provider": "legacy_reference_ranker",
                    "source": str(value.get("source") or ""),
                    "source_id": str(value.get("source_id") or position),
                },
            )

        pool = list(candidates.values())
        truncated_count = max(
            len(pool) - self.config.candidate_pool_limit,
            0,
        )
        pool = pool[: self.config.candidate_pool_limit]
        generation_stage = {
            "stage": "candidate_generation",
            "status": (
                "partial"
                if pool and (provider_failures or truncated_count)
                else "completed"
                if pool
                else "no_candidates"
            ),
            "candidate_count": len(pool),
            "reference_candidate_count": len(reference_candidates),
            "provider_count": len(self.candidate_providers),
            "provider_failures": provider_failures,
            "rejection_counts": rejection_counts,
            "truncated_count": truncated_count,
            "formula_fallback_used": False,
        }
        reference_stage = {
            "stage": "reference_scoring",
            "status": (
                "failed"
                if reference_error
                else "completed"
                if reference_candidates
                else "no_reference"
            ),
            "reason_code": (
                "reference_ranker_error" if reference_error else None
            ),
            "scored_candidate_count": len(reference_candidates),
            "calibrated_probability": False,
        }

        for candidate in pool:
            reference = _reference_metrics(
                candidate,
                has_13c=bool(q13),
                has_1h=bool(q1h),
            )
            candidate["_reference_evidence"] = reference
            source_reference = candidate.get("_reference")
            candidate["_legacy_reference_rank"] = (
                int(source_reference.get("rank") or 0)
                if isinstance(source_reference, Mapping)
                else 0
            )

        forward_by_id: dict[str, dict[str, Any]] = {}
        forward_model: Mapping[str, Any] | None = None
        forward_used_for_ranking = False
        forward_stage: dict[str, Any]
        if not q13:
            forward_stage = {
                "stage": "forward_scoring",
                "status": "skipped",
                "reason_code": "observed_13c_required",
                "used_for_ranking": False,
                "calibrated_probability": False,
            }
        elif not pool:
            forward_stage = {
                "stage": "forward_scoring",
                "status": "skipped",
                "reason_code": "no_candidates",
                "used_for_ranking": False,
                "calibrated_probability": False,
            }
        elif self.forward_scorer is None:
            forward_stage = {
                "stage": "forward_scoring",
                "status": "skipped",
                "reason_code": "forward_scorer_not_configured",
                "used_for_ranking": False,
                "calibrated_probability": False,
            }
        else:
            selected = pool[: self.config.forward_candidate_limit]
            sidecar_candidates = [
                {
                    "candidate_id": candidate["candidate_id"],
                    "smiles": candidate["smiles"],
                }
                for candidate in selected
            ]
            try:
                raw_forward = self.forward_scorer.score_candidates(
                    q13,
                    sidecar_candidates,
                    formula=formula_info.canonical if formula_info else None,
                )
            except Exception:
                raw_forward = {
                    "status": "unavailable",
                    "reason": "forward_scorer_error",
                    "calibrated_probability": False,
                    "candidates": [],
                }

            raw_status = str(raw_forward.get("status") or "invalid_output")
            probability_claim = raw_forward.get("calibrated_probability")
            raw_evidence = raw_forward.get("candidates")
            if (
                raw_status == "ok"
                and probability_claim is False
                and isinstance(raw_evidence, Sequence)
                and not isinstance(raw_evidence, (str, bytes))
            ):
                quantile_enabled = raw_forward.get("quantile_enabled") is True
                selected_ids = {
                    candidate["candidate_id"] for candidate in selected
                }
                duplicate_ids: set[str] = set()
                invalid_candidate_output = False
                for value in raw_evidence:
                    if not isinstance(value, Mapping):
                        invalid_candidate_output = True
                        continue
                    candidate_id = str(value.get("candidate_id") or "")
                    if candidate_id not in selected_ids:
                        invalid_candidate_output = True
                        continue
                    if candidate_id in forward_by_id:
                        duplicate_ids.add(candidate_id)
                        continue
                    normalised = _normalise_forward_candidate(
                        value,
                        quantile_enabled=quantile_enabled,
                    )
                    if normalised is None:
                        invalid_candidate_output = True
                        continue
                    forward_by_id[candidate_id] = normalised
                ranks = [
                    int(item["relative_rank"])
                    for item in forward_by_id.values()
                ]
                ranks_are_valid = (
                    len(set(ranks)) == len(ranks)
                    and set(ranks) == set(range(1, len(ranks) + 1))
                )
                if duplicate_ids or invalid_candidate_output or not ranks_are_valid:
                    forward_by_id = {}
                    raw_status = "invalid_output"
                full_coverage = (
                    len(pool) <= self.config.forward_candidate_limit
                    and len(forward_by_id) == len(pool)
                )
                forward_used_for_ranking = full_coverage
                forward_model = (
                    raw_forward.get("model")
                    if isinstance(raw_forward.get("model"), Mapping)
                    else None
                )
                forward_stage = {
                    "stage": "forward_scoring",
                    "status": (
                        "completed"
                        if full_coverage
                        else "partial"
                        if forward_by_id
                        else "failed"
                    ),
                    "reason_code": (
                        None
                        if full_coverage
                        else "candidate_coverage_incomplete"
                        if forward_by_id
                        else "invalid_forward_candidate_results"
                    ),
                    "evaluated_candidate_count": len(forward_by_id),
                    "candidate_count": len(pool),
                    "used_for_ranking": forward_used_for_ranking,
                    "quantile_enabled": quantile_enabled,
                    "calibrated_probability": False,
                    "model": dict(forward_model) if forward_model else None,
                }
            else:
                reason = (
                    "unsupported_calibrated_probability_claim"
                    if probability_claim is not False
                    else str(raw_forward.get("reason") or raw_status)
                )
                forward_stage = {
                    "stage": "forward_scoring",
                    "status": (
                        "skipped"
                        if raw_status
                        in {"unsupported_modality", "disabled", "not_configured"}
                        else "failed"
                    ),
                    "reason_code": reason,
                    "evaluated_candidate_count": 0,
                    "candidate_count": len(pool),
                    "used_for_ranking": False,
                    "calibrated_probability": False,
                }

        for candidate in pool:
            candidate["_forward_evidence"] = forward_by_id.get(
                candidate["candidate_id"]
            )
        pool.sort(
            key=lambda item: _rank_key(
                item,
                has_13c=bool(q13),
                use_forward=forward_used_for_ranking,
            )
        )

        calibration = None
        if pool and forward_used_for_ranking:
            try:
                from app.ml.calibration import calibrate_pool

                calibration = calibrate_pool(
                    pool,
                    context={
                        "formula": (
                            formula_info.canonical
                            if formula_info is not None
                            else None
                        ),
                        "modality": modality,
                        "n_13c": len(q13),
                        "solvent": solvent,
                        "supplied_candidate_count": len(candidate_smiles or ())
                        + len(generated_smiles or ()),
                        "generation_status": generation_stage.get("status"),
                        "provider_count": len(self.candidate_providers),
                        "provider_failures": provider_failures,
                        "provider_ids": [
                            str(
                                getattr(provider, "provider_id", None)
                                or provider.__class__.__name__
                            )
                            for provider in self.candidate_providers
                        ],
                        "generator_version": (
                            "hybrid-index-formula-plus-pubchem-fastformula-v1"
                        ),
                        "qc_passed": False,
                        "forward_model": (
                            dict(forward_model) if forward_model else None
                        ),
                    },
                )
            except Exception:
                calibration = None

        rank_basis = (
            "13c_forward_then_13c_reference_then_1h_reference"
            if q13 and forward_used_for_ranking
            else "13c_reference_then_1h_reference"
            if q13
            else "1h_reference"
        )
        output_candidates = []
        for rank, candidate in enumerate(pool[:result_limit], start=1):
            reference = candidate.pop("_reference_evidence")
            forward = candidate.pop("_forward_evidence")
            candidate.pop("_reference", None)
            candidate.pop("_legacy_reference_rank", None)
            evidence_state = _candidate_evidence_state(reference, forward)
            uncertainty_kind = _uncertainty_kind(reference, forward)
            assignment = {
                "reference_13c": (
                    reference.get("score_breakdown", {})
                    .get("c13", {})
                    .get("assignments", [])
                ),
                "reference_1h": (
                    reference.get("score_breakdown", {})
                    .get("h1", {})
                    .get("assignments", [])
                ),
                "forward": (
                    {
                        "mode": forward.get("assignment_mode"),
                        "matched_count": forward.get("matched_count"),
                        "unmatched_observed_count": forward.get(
                            "unmatched_observed_count"
                        ),
                        "unmatched_predicted_atom_count": forward.get(
                            "unmatched_predicted_atom_count"
                        ),
                    }
                    if forward
                    else None
                ),
            }
            output_candidates.append(
                {
                    **candidate,
                    "rank": rank,
                    "evidence_state": evidence_state,
                    "reference_evidence": reference,
                    "forward_evidence": forward,
                    "assignment": assignment,
                    "ranking_evidence": {
                        "basis": rank_basis,
                        "relative_only": True,
                        "calibrated_probability": (
                            calibration is not None and rank == 1
                        ),
                    },
                    "uncertainty_kind": uncertainty_kind,
                    "calibrated_probability": (
                        calibration["probability"]
                        if calibration is not None and rank == 1
                        else False
                    ),
                }
            )

        top_state = (
            output_candidates[0]["evidence_state"]
            if output_candidates
            else "no_reference"
        )
        top_uncertainty = (
            output_candidates[0]["uncertainty_kind"]
            if output_candidates
            else "unavailable"
        )
        if not output_candidates:
            decision_reason = "no_candidates"
        elif top_state == "no_reference":
            decision_reason = "no_spectral_reference"
        elif top_state == "no_match":
            decision_reason = "no_valid_spectral_match"
        elif calibration is not None:
            decision_reason = "correctness_probability_calibrated_abstain"
        else:
            decision_reason = "correctness_probability_not_calibrated"
        decision = {
            "action": "abstain",
            "reason_code": decision_reason,
            "selected_candidate_id": None,
            "leading_hypothesis_candidate_id": (
                output_candidates[0]["candidate_id"]
                if output_candidates
                else None
            ),
            "evidence_state": top_state,
            "uncertainty_kind": top_uncertainty,
            "calibrated_probability": calibration is not None,
            "top1_probability": (
                calibration["probability"] if calibration is not None else None
            ),
        }
        assignment_stage = {
            "stage": "assignment",
            "status": "completed" if output_candidates else "skipped",
            "reference_assignment_method": "hungarian_fixed_tolerance",
            "forward_assignment_method": (
                "provider_reported_atom_level"
                if forward_by_id
                else "not_available"
            ),
            "calibrated_probability": calibration is not None,
        }
        ranking_stage = {
            "stage": "ranking",
            "status": "completed" if output_candidates else "skipped",
            "basis": rank_basis,
            "candidate_pool_count": len(pool),
            "returned_candidate_count": len(output_candidates),
            "relative_only": True,
            "calibrated_probability": calibration is not None,
        }
        decision_stage = {
            "stage": "decision",
            "status": "completed",
            **decision,
        }

        warnings = [
            str(item)
            for item in legacy_result.get("warnings", [])
            if isinstance(item, str)
        ]
        if calibration is not None:
            warnings.append(
                "Calibrated conditional Top-1 probability is available "
                "(conditional on retrieval); automatic structure selection "
                "remains disabled."
            )
        else:
            warnings.append(
                "Hybrid v1 ranks hypotheses but has no calibrated structure-"
                "correctness probability; automatic structure selection is disabled."
            )
        if modality != "1h+13c":
            warnings.append(
                "Only one NMR nucleus is available; structural ambiguity is expected."
            )
        if provider_failures:
            warnings.append("One or more candidate providers failed closed.")
        if reference_error:
            warnings.append("The spectral reference ranker was unavailable.")
        if forward_stage["status"] in {"failed", "partial"}:
            warnings.append(
                "Forward evidence was incomplete and did not affect ranking."
            )
        warnings = list(dict.fromkeys(warnings))
        candidate_pool_status = (
            "formula_match"
            if formula_info is not None and output_candidates
            else "no_formula_match"
            if formula_info is not None
            else str(
                legacy_result.get("candidate_pool_status")
                or ("unconstrained_candidates" if output_candidates else "empty")
            )
        )

        return {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "status": "completed" if output_candidates else "no_candidates",
            "query": {
                "formula": (
                    formula_info.canonical if formula_info is not None else None
                ),
                "formula_input": formula,
                "dbe": formula_info.dbe if formula_info is not None else None,
                "modality": modality,
                "n_13c": len(q13),
                "n_1h": len(q1h),
                "peaks_13c": q13,
                "peaks_1h": q1h,
                "constraints": constraints.as_dict(),
            },
            "stages": [
                input_stage,
                generation_stage,
                reference_stage,
                forward_stage,
                assignment_stage,
                ranking_stage,
                decision_stage,
            ],
            "candidates": output_candidates,
            "decision": decision,
            "evidence_state": top_state,
            "uncertainty_kind": top_uncertainty,
            "calibrated_probability": calibration if calibration is not None else False,
            "top1_calibrated_probability": (
                calibration["probability"] if calibration is not None else None
            ),
            "candidate_pool_status": candidate_pool_status,
            "reference_context": {
                "ranker": legacy_result.get("ranker"),
                "index": legacy_result.get("index"),
                "candidate_pool_status": legacy_result.get(
                    "candidate_pool_status"
                ),
            },
            "forward_context": {
                "model": dict(forward_model) if forward_model else None,
                "used_for_ranking": forward_used_for_ranking,
                "calibrated_probability": calibration is not None,
                "calibration": calibration,
            },
            "warnings": warnings,
        }


def _peak_shifts(
    peaks: Sequence[Mapping[str, Any]] | None,
    *,
    nucleus: str,
) -> list[float]:
    lower, upper = (-20.0, 300.0) if nucleus == "13C" else (-5.0, 30.0)
    shifts = []
    for peak in peaks or ():
        if not isinstance(peak, Mapping):
            continue
        shift = _finite_float(
            peak.get("shift", peak.get("position", peak.get("center_ppm")))
        )
        if shift is not None and lower <= shift <= upper:
            shifts.append(shift)
    return sorted(shifts)


def hybrid_rank_candidates(
    *,
    peaks_13c: Sequence[Mapping[str, Any]] | None = None,
    peaks_1h: Sequence[Mapping[str, Any]] | None = None,
    formula: str | None = None,
    solvent: str | None = None,
    constraints: CandidateConstraints | None = None,
    generated_smiles: Sequence[str] | None = None,
    candidate_smiles: Sequence[str] | None = None,
    top_k: int = 10,
    reference_ranker: ReferenceRanker = rank_candidates,
    candidate_providers: Sequence[CandidateProvider] = (),
    forward_scorer: ForwardScorer | None = None,
    config: HybridPredictorConfig | None = None,
) -> dict[str, Any]:
    """Convenience entry point for the hybrid v1 pipeline."""

    effective_config = config or HybridPredictorConfig(result_limit=top_k)
    predictor = HybridPredictorV1(
        reference_ranker=reference_ranker,
        candidate_providers=candidate_providers,
        forward_scorer=forward_scorer,
        config=effective_config,
    )
    return predictor.predict(
        peaks_13c=peaks_13c,
        peaks_1h=peaks_1h,
        formula=formula,
        solvent=solvent,
        constraints=constraints,
        generated_smiles=generated_smiles,
        candidate_smiles=candidate_smiles,
        top_k=top_k,
    )


def hybrid_to_legacy_rank_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt a v1 result to the existing ``rank_candidates`` response shape.

    ``ranking_score`` remains the legacy reference-library heuristic.  It is
    never replaced with a fabricated hybrid probability; forward-only
    candidates therefore receive ``0.0`` in the compatibility field.
    """

    if result.get("schema_version") != SCHEMA_VERSION:
        raise HybridPredictorInputError("Unsupported hybrid result schema")
    legacy_candidates = []
    for value in result.get("candidates", []):
        if not isinstance(value, Mapping):
            continue
        reference = value.get("reference_evidence")
        reference = reference if isinstance(reference, Mapping) else {}
        breakdown = reference.get("score_breakdown")
        breakdown = breakdown if isinstance(breakdown, Mapping) else {}
        score = _finite_float(reference.get("relative_score"), 0.0) or 0.0
        legacy_candidates.append(
            {
                "rank": value.get("rank"),
                "smiles": value.get("smiles"),
                "compound_name": value.get("compound_name"),
                "molecular_formula": value.get("molecular_formula"),
                "molecular_weight": value.get("molecular_weight"),
                "source": (
                    value.get("provenance", [{}])[0].get("source", "")
                    if value.get("provenance")
                    else ""
                ),
                "source_id": (
                    value.get("provenance", [{}])[0].get("source_id", "")
                    if value.get("provenance")
                    else ""
                ),
                "ranking_score": round(min(max(score, 0.0), 1.0), 4),
                "ranking_score_semantics": (
                    "legacy_reference_heuristic_not_hybrid_probability"
                ),
                "evidence_level": value.get("evidence_state"),
                "evidence": {
                    "hybrid_schema_version": SCHEMA_VERSION,
                    "uncertainty_kind": value.get("uncertainty_kind"),
                    "calibrated_probability": False,
                },
                "score_breakdown": dict(breakdown),
                "matched_13c": reference.get("matched_13c", 0),
                "matched_1h": reference.get("matched_1h", 0),
                "candidate_id": value.get("candidate_id"),
                "candidate_provenance": value.get("provenance", []),
                "forward_evidence": value.get("forward_evidence"),
                "calibrated_probability": False,
            }
        )
    query = result.get("query")
    query = dict(query) if isinstance(query, Mapping) else {}
    reference_context = result.get("reference_context")
    reference_context = (
        reference_context if isinstance(reference_context, Mapping) else {}
    )
    return {
        "candidates": legacy_candidates,
        "candidate_pool_status": result.get("candidate_pool_status"),
        "query": query,
        "warnings": list(result.get("warnings", [])),
        "index": reference_context.get("index"),
        "ranker": reference_context.get("ranker"),
        "hybrid_schema_version": SCHEMA_VERSION,
        "hybrid_pipeline_version": PIPELINE_VERSION,
        "decision": result.get("decision"),
        "evidence_state": result.get("evidence_state"),
        "uncertainty_kind": result.get("uncertainty_kind"),
        "calibrated_probability": False,
    }


__all__ = [
    "CandidateConstraints",
    "CandidateProvider",
    "EVIDENCE_STATES",
    "ForwardScorer",
    "HybridPredictorConfig",
    "HybridPredictorError",
    "HybridPredictorInputError",
    "HybridPredictorV1",
    "PIPELINE_VERSION",
    "SCHEMA_VERSION",
    "UNCERTAINTY_KINDS",
    "hybrid_rank_candidates",
    "hybrid_to_legacy_rank_result",
]
