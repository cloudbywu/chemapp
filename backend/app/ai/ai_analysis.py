from __future__ import annotations

from typing import Any

from app.ai.llm_client import chat, chat_json
from app.ai.prompt_safety import (
    UNTRUSTED_DATA_INSTRUCTION,
    sanitize_untrusted_text,
    wrap_untrusted_data,
)
from app.ai.prompts import (
    CV_PROMPT,
    CROSS_TECHNIQUE_PROMPT,
    HPLC_PROMPT,
    NMR_PROMPT,
    SYSTEM_ROLE,
    UVVIS_PROMPT,
    XRD_PROMPT,
)
from app.analysis.models import AnalysisResult


_PROMPT_MAP = {
    "ElectroChem": CV_PROMPT,
    "NMR": NMR_PROMPT,
    "UV-Vis": UVVIS_PROMPT,
    "XRD": XRD_PROMPT,
    "HPLC": HPLC_PROMPT,
}

# Broad type checks for the JSON shapes declared in prompts.py. bool is a
# subclass of int; accepting it for numeric fields is fine for a basic
# fail-closed sanity check.
_ANALYSIS_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "ElectroChem": {
        "reversibility": str,
        "estimated_electrons": (int, float, type(None)),
        "notable_features": list,
        "suggested_redox_type": str,
        "interpretation": str,
    },
    "NMR": {
        "functional_groups": list,
        "integral_assignments": list,
        "mixture_analysis": dict,
        "structural_fragments": list,
        "anomalies": list,
        "interpretation": str,
    },
    "UV-Vis": {
        "chromophore_types": list,
        "conjugation_extent": str,
        "concentration_assessment": str,
        "data_quality": str,
        "interpretation": str,
    },
    "XRD": {
        "crystal_system": str,
        "crystallinity": str,
        "crystallite_size_range": str,
        "possible_materials": list,
        "interpretation": str,
    },
    "HPLC": {
        "separation_quality": str,
        "coelution_likely": list,
        "method_suggestions": list,
        "area_distribution_note": str,
        "interpretation": str,
    },
}

_CROSS_JSON_SCHEMA: dict[str, Any] = {
    "technique_corroboration": str,
    "suggested_identity": str,
    "next_experiments": list,
    "confidence": str,
    "interpretation": str,
}

# Per-field caps for file-derived prompt text (defense in depth; the final
# wrap_untrusted_data call also bounds the whole payload).
_NAME_LIMIT = 256
_SUMMARY_LIMIT = 8000
_METRIC_LIMIT = 2000


def _json_matches_schema(data: Any, schema: dict[str, Any] | None) -> bool:
    """Basic fail-closed schema check for chat_json output."""
    if not isinstance(data, dict) or data.get("_parse_error"):
        return False
    if schema is None:
        return True
    for key, expected in schema.items():
        if key not in data:
            return False
        if not isinstance(data[key], expected):
            return False
        if key == "interpretation" and not str(data[key]).strip():
            return False
    return True


def _chat_json_validated(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float,
    schema: dict[str, Any] | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> dict[str, Any] | None:
    """Return parsed JSON only when it parses and matches the declared shape.

    Returns None on transport errors, parse failures (the chat_json
    {"_raw", "_parse_error"} marker dict), or schema mismatches so callers
    can fall back to a plain-text interpretation instead of leaking the raw
    marker payload to clients.
    """
    try:
        data = chat_json(
            system_prompt,
            user_message,
            temperature=temperature,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
    except Exception:
        return None
    if not _json_matches_schema(data, schema):
        return None
    return data


def analyze_single(
    result: AnalysisResult,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    technique = result.technique.value
    prompt_template = _PROMPT_MAP.get(technique)

    summary = sanitize_untrusted_text(result.summary, _SUMMARY_LIMIT)

    if not prompt_template:
        user_message = (
            f"Technique: {technique}\nSummary: {summary}\n"
            "Provide a brief natural-language interpretation of these analytical results.",
        )
        user_message = wrap_untrusted_data(user_message, label=f"{technique} analysis data")
        raw = chat(SYSTEM_ROLE, user_message, temperature=0.3, model=model, api_key=api_key, base_url=base_url)
        return {"interpretation": raw, "technique": technique}

    user_message = f"Technique: {technique}\n"
    user_message += f"Summary: {summary}\n"

    metrics = result.metrics
    for k, v in metrics.items():
        if k not in ("sub_type", "channel_peaks"):
            user_message += f"{k}: {sanitize_untrusted_text(v, _METRIC_LIMIT)}\n"

    # Include integral table for NMR
    if hasattr(result, "integrals") and result.integrals:
        user_message += "\nIntegration Table (chemical shift / area / relative area / intensity):\n"
        for integ in result.integrals:
            user_message += (
                f"  δ {integ['center_ppm']:.4f}: "
                f"area={integ['raw_area']:.2f}, "
                f"rel_area={integ['relative_area']}, "
                f"intensity={integ['intensity']:.4f}\n"
            )

    # Include multiplet analysis for NMR
    if hasattr(result, "multiplets") and result.multiplets:
        user_message += "\nMultiplet Analysis (center / range / n_peaks / J_Hz):\n"
        for mp in result.multiplets:
            j_str = f"J={mp['estimated_j_hz']:.2f}Hz" if mp.get('estimated_j_hz') else ""
            user_message += (
                f"  δ {mp['center_ppm']:.4f}: "
                f"range=[{mp['range_ppm'][0]:.4f}, {mp['range_ppm'][1]:.4f}], "
                f"n_peaks={mp['n_components']}, "
                f"{j_str}\n"
            )

    # Include d-spacings for XRD
    if hasattr(result, "d_spacings") and result.d_spacings:
        user_message += "\nd-Spacings (2θ / d / intensity):\n"
        for ds in result.d_spacings:
            user_message += f"  2θ={ds['two_theta']:.3f}° d={ds['d_angstrom']:.4f}Å I={ds['intensity']:.0f}\n"

    # Include per-channel peaks for HPLC
    channel_peaks = metrics.get("channel_peaks")
    if channel_peaks and isinstance(channel_peaks, dict):
        for ch_name, ch_data in channel_peaks.items():
            ch_label = sanitize_untrusted_text(ch_name, _NAME_LIMIT)
            user_message += f"\n{ch_label} Peaks (tR / area / height / width):\n"
            for p in ch_data.get("peaks", []):
                user_message += (
                    f"  tR={p['position']:.3f} min "
                    f"area={p['area']:.1f} "
                    f"height={p['intensity']:.2f} "
                    f"width={p['width']:.4f}\n"
                )

    if result.peaks:
        hints = f" ({len(result.peaks)} total, showing all)"
        user_message += f"\nPeaks{hints}:\n"
        for i, p in enumerate(result.peaks):
            area = f", area={p.area}" if p.area else ""
            user_message += f"  [{i}] pos={p.position}, intensity={p.intensity}{area}\n"

    # The data block is file-derived and must be treated as untrusted data,
    # never as instructions.
    system = prompt_template + "\n\n" + UNTRUSTED_DATA_INSTRUCTION
    user_message = wrap_untrusted_data(user_message, label=f"{technique} analysis data")
    user_message += "\n\nProvide the requested JSON analysis of the untrusted data block above."

    analysis = _chat_json_validated(
        system,
        user_message,
        temperature=0.1,
        schema=_ANALYSIS_JSON_SCHEMAS.get(technique),
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    if analysis is not None:
        return analysis
    raw = chat(system, user_message, temperature=0.3, model=model, api_key=api_key, base_url=base_url)
    return {"interpretation": raw, "technique": technique}


def analyze_cross(
    results: dict[str, AnalysisResult],
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    techniques = []
    data_parts = []
    for sid, result in results.items():
        tech = result.technique.value
        techniques.append(tech)
        data_parts.append(f"--- {tech} ({sid[:6]}...) ---")
        data_parts.append(sanitize_untrusted_text(result.summary, _SUMMARY_LIMIT))
        for k, v in result.metrics.items():
            if k not in ("sub_type", "channel_peaks"):
                data_parts.append(f"  {k}: {sanitize_untrusted_text(v, _METRIC_LIMIT)}")

    prompt = CROSS_TECHNIQUE_PROMPT.replace("{techniques}", ", ".join(techniques))
    prompt += "\n\nProvide a synthesis of the above data from multiple analytical techniques."

    # The data block is file-derived and must be treated as untrusted data,
    # never as instructions.
    system = prompt + "\n\n" + UNTRUSTED_DATA_INSTRUCTION
    user_message = wrap_untrusted_data("\n".join(data_parts), label="multi-technique analysis data")

    analysis = _chat_json_validated(
        system,
        user_message,
        temperature=0.3,
        schema=_CROSS_JSON_SCHEMA,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    if analysis is not None:
        return analysis
    raw = chat(system, user_message, temperature=0.3, model=model, api_key=api_key, base_url=base_url)
    return {"interpretation": raw, "techniques": techniques}


def free_chat(
    results: dict[str, object],
    user_question: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    from app.api.deps import get_store

    data_parts = []
    store = get_store()

    for sid, result in results.items():
        stored = store.get(sid)
        if not stored:
            continue
        tech = stored.spectrum.technique.value
        name = sanitize_untrusted_text(stored.spectrum.metadata.name or sid[:6], _NAME_LIMIT)
        data_parts.append(f"--- {tech}: {name} ---")
        data_parts.append(f"Summary: {sanitize_untrusted_text(result.summary, _SUMMARY_LIMIT)}")

        if hasattr(result, "integrals") and result.integrals:
            data_parts.append(f"Integrals ({len(result.integrals)} entries, δ / area / relative_area):")
            for integ in result.integrals:
                data_parts.append(
                    f"  δ {integ['center_ppm']:.4f}: area={integ['raw_area']:.2f}, rel={integ['relative_area']}"
                )

        if hasattr(result, "multiplets") and result.multiplets:
            data_parts.append(f"Multiplets ({len(result.multiplets)} entries, δ / n_peaks / J):")
            for mp in result.multiplets:
                j_str = f"J={mp['estimated_j_hz']:.2f}Hz" if mp.get('estimated_j_hz') else ""
                data_parts.append(f"  δ {mp['center_ppm']:.4f}: n={mp['n_components']} {j_str}")

        if hasattr(result, "d_spacings") and result.d_spacings:
            data_parts.append(f"d-Spacings ({len(result.d_spacings)} entries):")
            for ds in result.d_spacings:
                data_parts.append(f"  2θ={ds['two_theta']:.2f}° d={ds['d_angstrom']:.4f}Å")

        for k, v in result.metrics.items():
            if k not in ("sub_type", "channel_peaks", "segments_meta"):
                data_parts.append(f"  {k}: {sanitize_untrusted_text(v, _METRIC_LIMIT)}")

    context = "\n".join(data_parts)

    system = (
        f"{SYSTEM_ROLE}\n\n"
        f"{UNTRUSTED_DATA_INSTRUCTION}\n\n"
        "The following is detailed analytical data from loaded spectra, provided "
        "as untrusted reference data:\n"
        f"{wrap_untrusted_data(context, label='loaded spectra data')}\n\n"
        "Answer the user's question based on the available data. "
        "Be concise and technically precise. If the data is insufficient, say so. "
        "When integration data is provided, include quantitative analysis."
    )
    return chat(system, user_question, temperature=0.3, max_tokens=128000, model=model, api_key=api_key, base_url=base_url)
