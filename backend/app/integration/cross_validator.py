from __future__ import annotations

from typing import Any

from app.integration.models import CrossValidationItem


def validate_nmr_uv(techniques: dict[str, dict[str, Any]]) -> list[CrossValidationItem]:
    results = []
    nmr = techniques.get("NMR", {})
    uv = techniques.get("UV-Vis", {})

    if not nmr or not uv:
        return results

    aromatic_count = nmr.get("aromatic_peaks", 0)
    lambda_max = uv.get("lambda_max", [])
    has_uv_absorption = any(200 < wl < 350 for wl in lambda_max)

    if aromatic_count > 0:
        results.append(CrossValidationItem(
            pair=("NMR", "UV-Vis"),
            metric="NMR芳香区 → UV吸收",
            description=f"NMR detected {aromatic_count} aromatic peaks → UV should show 250-350nm absorption",
            score=0.85 if has_uv_absorption else 0.25,
            detail=f"UV λmax={lambda_max}" if has_uv_absorption else "No UV absorption detected in aromatic range",
        ))

    nmr_integral_total = nmr.get("total_integral", 0)
    uv_concentration = uv.get("concentration", None)
    if nmr_integral_total > 0 and uv_concentration is not None:
        results.append(CrossValidationItem(
            pair=("NMR", "UV-Vis"),
            metric="NMR积分 ↔ UV浓度",
            description="NMR total integral vs UV calibration concentration",
            score=0.70,
            detail=f"NMR integral={nmr_integral_total:.2e}, UV concentration={uv_concentration:.2f}",
        ))

    return results


def validate_uv_fluorescence(techniques: dict[str, dict[str, Any]]) -> list[CrossValidationItem]:
    results = []
    uv = techniques.get("UV-Vis", {})
    fluor = techniques.get("Fluorescence", {})

    if not uv or not fluor:
        return results

    lambda_max = uv.get("lambda_max", [])
    ex_peak = fluor.get("excitation_peak", None)

    if lambda_max and ex_peak:
        closest_uv = min(lambda_max, key=lambda w: abs(w - ex_peak))
        diff = abs(closest_uv - ex_peak)
        score = 1.0 if diff < 20 else (0.6 if diff < 50 else 0.3)
        results.append(CrossValidationItem(
            pair=("UV-Vis", "Fluorescence"),
            metric="UV λmax ↔ 荧光 EX峰",
            description="UV absorption max vs fluorescence excitation peak overlap",
            score=score,
            detail=f"UV λmax={closest_uv}nm, EX peak={ex_peak}nm, Δ={diff:.0f}nm",
        ))

    return results


def validate_ex_em_pair(techniques: dict[str, dict[str, Any]]) -> list[CrossValidationItem]:
    results = []
    fluor = techniques.get("Fluorescence", {})

    ex_peak = fluor.get("excitation_peak")
    em_peak = fluor.get("emission_peak")
    stokes = fluor.get("stokes_shift_nm")

    if ex_peak and em_peak and stokes is not None:
        valid_stokes = 20 < stokes < 250
        score = 1.0 if valid_stokes else (0.5 if stokes > 0 else 0.1)
        results.append(CrossValidationItem(
            pair=("Fluorescence", "Fluorescence"),
            metric="Stokes位移合理性",
            description="Excitation vs Emission: Stokes shift should be 20-250 nm",
            score=score,
            detail=f"EX={ex_peak}nm, EM={em_peak}nm, Stokes={stokes}nm",
        ))

        ex_lt_em = ex_peak < em_peak
        results.append(CrossValidationItem(
            pair=("Fluorescence", "Fluorescence"),
            metric="EX < EM 校验",
            description="Excitation wavelength must be shorter than emission",
            score=1.0 if ex_lt_em else 0.0,
            detail=f"EX={ex_peak}nm {'<' if ex_lt_em else '≥'} EM={em_peak}nm",
        ))

    return results


def validate_nmr_fluorescence(techniques: dict[str, dict[str, Any]]) -> list[CrossValidationItem]:
    results = []
    nmr = techniques.get("NMR", {})
    fluor = techniques.get("Fluorescence", {})

    if not nmr or not fluor:
        return results

    aromatic_count = nmr.get("aromatic_peaks", 0)
    stokes = fluor.get("stokes_shift_nm")

    if aromatic_count > 0 and stokes is not None:
        large_stokes = stokes > 50
        delocalized = large_stokes and aromatic_count > 0
        results.append(CrossValidationItem(
            pair=("NMR", "Fluorescence"),
            metric="芳香区 ↔ 荧光Stokes",
            description="Aromatic protons suggest extended conjugation → large Stokes shift",
            score=0.75 if delocalized else 0.50,
            detail=f"Aromatic protons={aromatic_count}, Stokes={stokes}nm",
        ))

    return results


def run_all_validations(techniques: dict[str, dict[str, Any]]) -> list[CrossValidationItem]:
    all_results = []
    all_results.extend(validate_nmr_uv(techniques))
    all_results.extend(validate_uv_fluorescence(techniques))
    all_results.extend(validate_ex_em_pair(techniques))
    all_results.extend(validate_nmr_fluorescence(techniques))
    return all_results
