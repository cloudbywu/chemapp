from __future__ import annotations

from typing import Any

from app.analysis.models import AnalysisResult
from app.integration.cross_validator import run_all_validations
from app.integration.models import InferenceResult


def _extract_technique_data(technique: str, result: AnalysisResult) -> dict[str, Any]:
    data: dict[str, Any] = {
        "n_peaks": len(result.peaks),
        "summary": result.summary,
    }

    if hasattr(result, "integrals"):
        aromatic_range = [i for i in result.integrals if 6.5 < i["center_ppm"] < 9.0]
        data["aromatic_peaks"] = len(aromatic_range)
        data["aliphatic_peaks"] = len([i for i in result.integrals if 0.5 < i["center_ppm"] < 4.5])
        data["total_integral"] = result.total_integral
        data["n_multiplets"] = len(result.multiplets)

    if hasattr(result, "lambda_max"):
        data["lambda_max"] = result.lambda_max
        data["calibration"] = result.calibration
        data["concentration"] = result.sample_concentration

    if hasattr(result, "ex_peak"):
        data["excitation_peak"] = result.ex_peak
        data["emission_peak"] = result.em_peak
        data["stokes_shift_nm"] = result.stokes_shift_nm
        data["stokes_shift_cm1"] = result.stokes_shift_cm1

    if technique == "HPLC":
        channel_peaks = result.metrics.get("channel_peaks") or {}
        data["channels"] = {
            name: {
                "n_peaks": len(ch.get("peaks", [])),
                "total_area": ch.get("total_area"),
                "main_peak": max(ch.get("peaks", []), key=lambda p: p.get("area", 0), default=None),
            }
            for name, ch in channel_peaks.items()
        }

    if technique == "XRD":
        data["phase_matches"] = getattr(result, "phase_matches", [])[:5]
        data["crystallinity"] = getattr(result, "crystallinity", {})

    return data


def _build_evidence_table(technique_data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for technique, data in technique_data.items():
        if technique == "NMR":
            evidence.append({
                "technique": technique,
                "evidence": "1D NMR integrals and multiplets",
                "support": f"{data.get('n_peaks', 0)} peaks, {data.get('n_multiplets', 0)} multiplets, total integral {data.get('total_integral', 0):.3g}",
                "confidence": 0.85 if data.get("total_integral", 0) else 0.55,
            })
        elif technique == "HPLC":
            channels = data.get("channels", {})
            support = "; ".join(
                f"{name}: {ch.get('n_peaks', 0)} peaks, total area {ch.get('total_area', 0)}"
                for name, ch in channels.items()
            )
            evidence.append({
                "technique": technique,
                "evidence": "Chromatographic peak table",
                "support": support or f"{data.get('n_peaks', 0)} peaks",
                "confidence": 0.9 if channels else 0.6,
            })
        elif technique == "XRD":
            phases = data.get("phase_matches", [])
            top = phases[0] if phases else {}
            evidence.append({
                "technique": technique,
                "evidence": "Powder diffraction phase match",
                "support": f"{top.get('phase_name', 'No phase')} score {top.get('match_score', 0):.1f}%" if top else "No phase match",
                "confidence": min(float(top.get("match_score", 0)) / 100, 0.95) if top else 0.25,
            })
        elif technique == "UV-Vis":
            evidence.append({
                "technique": technique,
                "evidence": "Absorption maxima and calibration",
                "support": f"lambda max: {data.get('lambda_max', [])}",
                "confidence": 0.75 if data.get("lambda_max") else 0.45,
            })
        elif technique == "Fluorescence":
            evidence.append({
                "technique": technique,
                "evidence": "Excitation/emission relationship",
                "support": f"Stokes shift {data.get('stokes_shift_nm')} nm",
                "confidence": 0.75 if data.get("stokes_shift_nm") else 0.45,
            })
        else:
            evidence.append({
                "technique": technique,
                "evidence": "Technique summary",
                "support": data.get("summary", ""),
                "confidence": 0.5,
            })
    return evidence


class InferenceEngine:
    def analyze(self, results: dict[str, AnalysisResult]) -> InferenceResult:
        technique_data: dict[str, dict[str, Any]] = {}

        for sid, result in results.items():
            tech = result.technique.value
            existing = technique_data.get(tech, {})
            data = _extract_technique_data(tech, result)
            if tech == "Fluorescence" and existing:
                for k, v in data.items():
                    if existing.get(k) is None and v is not None:
                        existing[k] = v
                technique_data[tech] = existing
            else:
                technique_data[tech] = data

        validations = run_all_validations(technique_data)
        evidence_table = _build_evidence_table(technique_data)

        scores = [cv.score for cv in validations] if validations else [0.5]
        consistency_score = round(sum(scores) / len(scores), 2)

        n_techniques = len(technique_data)
        tech_factor = min(n_techniques / 3, 1.0)
        has_nmr = "NMR" in technique_data
        has_uv = "UV-Vis" in technique_data
        has_fluor = "Fluorescence" in technique_data
        data_completeness = (int(has_nmr) * 0.4 + int(has_uv) * 0.3 + int(has_fluor) * 0.3)
        confidence = round(0.3 * tech_factor + 0.3 * data_completeness + 0.4 * consistency_score, 2)

        anomalies: list[str] = []
        for cv in validations:
            if cv.score < 0.4:
                anomalies.append(f"[{cv.pair[0]} ↔ {cv.pair[1]}] {cv.metric}: {cv.detail}")

        conclusions = []
        if consistency_score >= 0.75:
            conclusions.append("High cross-technique consistency — data from different instruments corroborate each other.")
        elif consistency_score >= 0.50:
            conclusions.append("Moderate consistency — some techniques align, further investigation recommended.")
        else:
            conclusions.append("Low consistency — significant discrepancies between techniques; verify data quality.")

        if has_nmr and has_uv and consistency_score >= 0.6:
            conclusions.append("NMR and UV-Vis data are compatible. The aromatic proton signals align with the UV absorption profile.")
        if has_fluor:
            conclusions.append("Fluorescence data provides complementary information about the excited-state properties.")

        overall = f"Comprehensive analysis with {n_techniques} technique(s). "
        overall += f"Consistency score: {consistency_score:.2f}, Confidence: {confidence:.2f}. "
        if anomalies:
            overall += f"{len(anomalies)} anomaly(s) detected."
        else:
            overall += "No significant anomalies detected."

        return InferenceResult(
            technique_results={**technique_data, "_evidence_table": {"items": evidence_table}},
            cross_validations=validations,
            consistency_score=consistency_score,
            confidence=confidence,
            anomalies=anomalies,
            conclusions=conclusions,
            overall_assessment=overall,
        )
