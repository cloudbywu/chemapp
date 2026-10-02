"""Input validation and evidence preparation for NMR structure elucidation.

The inverse NMR problem is very sensitive to seemingly small input mistakes:
different molecular-formula ordering, solvent/reference peaks treated as analyte
signals, and individual multiplet lines treated as independent resonances.  This
module keeps those deterministic, auditable steps separate from any ML model.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable


class FormulaError(ValueError):
    """Raised when a molecular formula cannot be parsed unambiguously."""


_ELEMENTS = frozenset(
    """
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni
    Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I
    Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt
    Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr
    Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
    """.split()
)
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_HALOGENS = frozenset({"F", "Cl", "Br", "I", "At", "Ts"})


@dataclass(frozen=True)
class FormulaInfo:
    canonical: str
    elements: dict[str, int]
    dbe: float | None


def parse_formula(formula: str | None) -> FormulaInfo | None:
    """Parse a neutral, non-isotopic formula and return Hill-order notation.

    Parenthesised formulae, salts/adducts, isotope labels, and explicit charges
    are deliberately rejected: silently guessing their meaning would make a
    hard candidate constraint unsafe.
    """

    if formula is None or not formula.strip():
        return None
    compact = re.sub(r"\s+", "", formula)
    tokens = list(_FORMULA_TOKEN.finditer(compact))
    if not tokens or "".join(match.group(0) for match in tokens) != compact:
        raise FormulaError(
            "Use a neutral molecular formula without parentheses, adducts, "
            "isotope labels, or charge annotations (for example C10H15BrN2)."
        )

    counts: dict[str, int] = {}
    for match in tokens:
        element = match.group(1)
        if element not in _ELEMENTS:
            raise FormulaError(f"Unknown element symbol: {element}")
        count = int(match.group(2) or "1")
        if count <= 0:
            raise FormulaError(f"Element count must be positive: {match.group(0)}")
        counts[element] = counts.get(element, 0) + count

    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else [])
        order += sorted(element for element in counts if element not in {"C", "H"})
    else:
        order = sorted(counts)
    canonical = "".join(
        element + (str(counts[element]) if counts[element] != 1 else "")
        for element in order
    )

    # Standard closed-shell DBE approximation.  Oxygen and divalent sulfur do
    # not contribute; halogens count like hydrogen.  Return None for formulae
    # outside the common organic subset instead of presenting false precision.
    supported = set(counts).issubset({"C", "H", "N", "P", "O", "S", *_HALOGENS})
    if supported and "C" in counts:
        halogens = sum(counts.get(element, 0) for element in _HALOGENS)
        dbe = (
            2 * counts.get("C", 0)
            + 2
            + counts.get("N", 0)
            + counts.get("P", 0)
            - counts.get("H", 0)
            - halogens
        ) / 2
        dbe = round(float(dbe), 3)
    else:
        dbe = None

    return FormulaInfo(canonical=canonical, elements=counts, dbe=dbe)


def canonical_formula(formula: str | None) -> str:
    info = parse_formula(formula)
    return info.canonical if info else ""


_SOLVENT_1H_SHIFTS: dict[str, tuple[float, ...]] = {
    "cdcl3": (7.26,),
    "chloroform-d": (7.26,),
    "dmso-d6": (2.50,),
    "d6-dmso": (2.50,),
    "cd3od": (3.31, 4.87),
    "methanol-d4": (3.31, 4.87),
    "acetone-d6": (2.05,),
    "d2o": (4.79,),
    "benzene-d6": (7.16,),
    "c6d6": (7.16,),
    "cd2cl2": (5.32,),
    "thf-d8": (1.72, 3.58),
}


def _normalise_solvent(solvent: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (solvent or "").strip().lower()).strip("-")


def _coerce_peak(item: dict[str, Any], nucleus: str) -> dict[str, Any] | None:
    raw_shift = item.get("shift", item.get("position", item.get("center_ppm")))
    try:
        shift = float(raw_shift)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(shift):
        return None
    lower, upper = (-5.0, 30.0) if nucleus == "1H" else (-20.0, 300.0)
    if not lower <= shift <= upper:
        return None

    try:
        intensity = float(item.get("intensity", item.get("height", 1.0)) or 0.0)
    except (TypeError, ValueError):
        intensity = 1.0
    if not math.isfinite(intensity):
        intensity = 1.0
    integral_raw = item.get("integral", item.get("relative_area"))
    try:
        integral = float(integral_raw) if integral_raw is not None else None
    except (TypeError, ValueError):
        integral = None
    if integral is not None and not math.isfinite(integral):
        integral = None

    return {
        "shift": shift,
        "intensity": intensity,
        "integral": integral,
        "multiplicity": str(item.get("multiplicity", item.get("type", "")) or ""),
        "assignment": str(item.get("assignment", "") or ""),
    }


def _cluster_peaks(peaks: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    if not peaks:
        return []
    ordered = sorted(peaks, key=lambda peak: peak["shift"])
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    for peak in ordered[1:]:
        if peak["shift"] - groups[-1][-1]["shift"] <= tolerance:
            groups[-1].append(peak)
        else:
            groups.append([peak])

    clustered: list[dict[str, Any]] = []
    for group in groups:
        weights = [max(abs(float(peak["intensity"])), 1e-12) for peak in group]
        weight_sum = sum(weights)
        center = sum(peak["shift"] * weight for peak, weight in zip(group, weights, strict=True)) / weight_sum
        integrals = [peak["integral"] for peak in group if peak.get("integral") is not None]
        multiplicities = [peak["multiplicity"] for peak in group if peak.get("multiplicity")]
        assignments = [peak["assignment"] for peak in group if peak.get("assignment")]
        clustered.append(
            {
                "shift": round(float(center), 6),
                "intensity": max(float(peak["intensity"]) for peak in group),
                "integral": sum(integrals) if integrals else None,
                "multiplicity": multiplicities[0] if len(set(multiplicities)) == 1 else "",
                "assignment": assignments[0] if len(set(assignments)) == 1 else "",
                "line_count": len(group),
                "component_shifts": [round(float(peak["shift"]), 6) for peak in group],
            }
        )
    return clustered


def prepare_query_peaks(
    raw_peaks: Iterable[dict[str, Any]] | None,
    *,
    nucleus: str,
    solvent: str | None = None,
    exclude_references: bool = True,
    cluster_1h_lines: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate, annotate, filter, and optionally cluster query peaks."""

    coerced = [
        peak
        for item in (raw_peaks or [])
        if (peak := _coerce_peak(item, nucleus)) is not None
    ]
    excluded: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    solvent_key = _normalise_solvent(solvent)
    solvent_shifts = _SOLVENT_1H_SHIFTS.get(solvent_key, ()) if nucleus == "1H" else ()

    for peak in coerced:
        assignment = peak["assignment"].lower()
        reason = ""
        if exclude_references and nucleus == "1H":
            if (
                "tms" in assignment
                or "reference" in assignment
                or abs(peak["shift"]) <= 0.035
            ):
                reason = "reference_peak"
            elif "solvent" in assignment or any(
                abs(peak["shift"] - shift) <= 0.04 for shift in solvent_shifts
            ):
                reason = "solvent_peak"
        if reason:
            excluded.append({**peak, "reason": reason})
        else:
            retained.append(peak)

    prepared = (
        _cluster_peaks(retained, tolerance=0.035)
        if nucleus == "1H" and cluster_1h_lines
        else sorted(retained, key=lambda peak: peak["shift"])
    )
    audit = {
        "nucleus": nucleus,
        "received": len(list(coerced)),
        "retained": len(prepared),
        "excluded": excluded,
        "clustered": nucleus == "1H" and cluster_1h_lines,
        "solvent": solvent or "",
    }
    return prepared, audit


def build_generation_prompt(
    peaks_13c: Iterable[dict[str, Any]],
    peaks_1h: Iterable[dict[str, Any]],
    *,
    formula: str | None = None,
) -> str:
    """Build an information-preserving, deterministic generation prompt."""

    parts: list[str] = []
    formula_info = parse_formula(formula)
    if formula_info is not None:
        parts.append(f"formula: {formula_info.canonical}")

    p13_tokens = [f"{float(peak['shift']):.2f}" for peak in peaks_13c]
    if p13_tokens:
        parts.append("13C NMR: " + "; ".join(p13_tokens))

    p1h_tokens = []
    for peak in peaks_1h:
        token = f"{float(peak['shift']):.3f}"
        if peak.get("integral") is not None:
            token += f" integral={float(peak['integral']):.2f}"
        if peak.get("multiplicity"):
            token += f" {str(peak['multiplicity']).lower()}"
        p1h_tokens.append(token)
    if p1h_tokens:
        parts.append("1H NMR: " + "; ".join(p1h_tokens))
    return "predict SMILES: " + " | ".join(parts)


def validate_generated_smiles(
    smiles_values: Iterable[str | dict[str, Any]],
    *,
    formula: str | None = None,
    generator: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate structures while retaining their experimental origin.

    Only provenance fields are copied. Generator-supplied scores, ranks, and
    probability claims cannot bypass the validation/evidence boundary.
    """

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except Exception:
        return [], {
            "status": "rdkit_unavailable",
            "accepted": 0,
            "rejected": [],
        }

    formula_info = parse_formula(formula)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    provenance_keys = (
        "model", "input_mode", "prompt_schema", "requested_variant",
        "provided_modalities", "used_modalities", "ignored_modalities",
        "input_warnings", "observed_carbon_lower_bound", "heavy_atoms",
    )
    for raw_value in smiles_values:
        candidate = raw_value if isinstance(raw_value, dict) else {}
        raw = str(candidate.get("smiles", "") if candidate else raw_value or "").strip()
        if not raw:
            continue
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            rejected.append({"smiles": raw, "reason": "invalid_smiles"})
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)
        generated_formula = rdMolDescriptors.CalcMolFormula(mol)
        try:
            generated_formula = canonical_formula(generated_formula)
        except FormulaError:
            rejected.append({"smiles": raw, "reason": "unsupported_generated_formula"})
            continue
        if formula_info is not None and generated_formula != formula_info.canonical:
            rejected.append(
                {
                    "smiles": raw,
                    "reason": (
                        f"formula_mismatch:{generated_formula}!="
                        f"{formula_info.canonical}"
                    ),
                }
            )
            continue
        candidate_provenance = {
            key: deepcopy((provenance or {}).get(key, candidate.get(key)))
            for key in provenance_keys
            if key in (provenance or {}) or key in candidate
        }
        if candidate.get("origin"):
            candidate_provenance["origin"] = str(candidate["origin"])
        if generator:
            candidate_provenance["generator"] = generator
        accepted.append(
            {
                **candidate_provenance,
                "rank": len(accepted) + 1,
                "smiles": canonical,
                "molecular_formula": generated_formula,
                "molecular_weight": round(float(Descriptors.MolWt(mol)), 3),
                "source": f"{generator}-generation-experimental" if generator else "generation-experimental",
                "evidence_level": "unverified",
                "calibrated_probability": False,
            }
        )

    return accepted, {
        "status": "validated",
        "accepted": len(accepted),
        "rejected": rejected,
        "formula_constraint": formula_info.canonical if formula_info else None,
    }
