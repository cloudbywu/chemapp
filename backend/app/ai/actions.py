from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from app.analysis import AnalyzerRegistry
from app.analysis.models import AnalysisResult
from app.analysis.quality import assess_quality
from app.analysis.utils import trapezoidal_integrate
from app.api.deps import get_store
from app.api.store import RevisionConflict, StoredSpectrum
from app.integration import InferenceEngine, build_report


@dataclass(frozen=True)
class AIActionSpec:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    destructive: bool = False


class AIActionError(ValueError):
    pass


_ACTION_LOCK = threading.RLock()
_PROCESS_PREVIEW_SECRET = secrets.token_bytes(32)
_PREVIEW_TOKEN_VERSION = 1
_DEFAULT_PREVIEW_TOKEN_TTL_SECONDS = 300


def _technique_key(value: str) -> str | None:
    return {
        "nmr": "nmr",
        "uvvis": "uvvis",
        "uv-vis": "uvvis",
        "fluorescence": "fluorescence",
        "xrd": "xrd",
        "hplc": "hplc",
        "electrochem": "electrochem",
    }.get(value.lower().replace("-", ""))


def _ensure_result(sid: str):
    store = get_store()
    stored = store.get(sid)
    if stored is None:
        raise AIActionError(f"Spectrum {sid} not found")
    if stored.result is None:
        key = _technique_key(stored.spectrum.technique.value)
        if key is None:
            raise AIActionError(f"No analyzer for {stored.spectrum.technique.value}")
        stored.result = AnalyzerRegistry.get(key)().analyze(stored.spectrum)
        stored.result.metrics["quality"] = assess_quality(stored.spectrum, stored.result)
    return stored


def _preview_secret() -> bytes:
    """Return a stable deployment secret, with a process-local test fallback."""

    configured = (
        os.environ.get("CHEMAPP_AI_PREVIEW_SECRET")
        or os.environ.get("CHEMAPP_ACCESS_TOKEN")
        or os.environ.get("CHEMAPP_ADMIN_TOKEN")
    )
    if configured:
        return hashlib.sha256(
            b"chemapp-ai-preview-v1\0" + configured.encode("utf-8")
        ).digest()
    return _PROCESS_PREVIEW_SECRET


def _preview_token_ttl_seconds() -> int:
    try:
        requested = int(
            os.environ.get(
                "CHEMAPP_AI_PREVIEW_TTL_SECONDS",
                str(_DEFAULT_PREVIEW_TOKEN_TTL_SECONDS),
            )
        )
    except ValueError:
        requested = _DEFAULT_PREVIEW_TOKEN_TTL_SECONDS
    return max(1, min(requested, 3600))


def _canonical_action_payload(name: str, args: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            {"name": name, "args": args},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AIActionError("AI action arguments must be valid JSON") from exc


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError) as exc:
        raise AIActionError("Invalid AI action preview_token") from exc


def _issue_preview_token(name: str, args: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_action_payload(name, args)).hexdigest()
    payload = json.dumps(
        {
            "v": _PREVIEW_TOKEN_VERSION,
            "exp": int(time.time()) + _preview_token_ttl_seconds(),
            "digest": digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    signature = hmac.new(_preview_secret(), payload, hashlib.sha256).digest()
    return f"{_b64url_encode(payload)}.{_b64url_encode(signature)}"


def _verify_preview_token(
    token: str | None,
    name: str,
    args: dict[str, Any],
) -> None:
    if not token:
        raise AIActionError(
            "preview_token is required for destructive AI actions; preview the "
            "same action first"
        )
    if not isinstance(token, str) or len(token) > 2048:
        raise AIActionError("Invalid AI action preview_token")
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        payload = _b64url_decode(encoded_payload)
        signature = _b64url_decode(encoded_signature)
        decoded = json.loads(payload)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise AIActionError("Invalid AI action preview_token") from exc
    expected_signature = hmac.new(_preview_secret(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected_signature):
        raise AIActionError("Invalid AI action preview_token")
    if not isinstance(decoded, dict) or decoded.get("v") != _PREVIEW_TOKEN_VERSION:
        raise AIActionError("Invalid AI action preview_token")
    expires_at = decoded.get("exp")
    if not isinstance(expires_at, int) or expires_at < int(time.time()):
        raise AIActionError("AI action preview_token has expired; preview again")
    digest = hashlib.sha256(_canonical_action_payload(name, args)).hexdigest()
    if not hmac.compare_digest(str(decoded.get("digest", "")), digest):
        raise AIActionError(
            "AI action or arguments changed after preview; preview again"
        )


def _infer_spectrum_id(args: dict[str, Any]) -> str | None:
    sid = args.get("spectrum_id")
    if sid is not None:
        return str(sid)
    ids = args.get("ids")
    if isinstance(ids, list) and len(ids) == 1:
        return str(ids[0])
    return None


def _expected_revision(args: dict[str, Any]) -> int | None:
    raw = args.get("expected_revision")
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise AIActionError("expected_revision must be a non-negative integer")
    try:
        revision = int(raw)
    except (TypeError, ValueError) as exc:
        raise AIActionError("expected_revision must be a non-negative integer") from exc
    if revision < 0 or (isinstance(raw, float) and not raw.is_integer()):
        raise AIActionError("expected_revision must be a non-negative integer")
    return revision


def _require_destructive_revision(
    spec: AIActionSpec,
    args: dict[str, Any],
) -> tuple[str | None, Any | None, int | None]:
    """Validate one destructive action against a persisted result snapshot."""

    if not spec.destructive:
        return None, None, None
    sid = _infer_spectrum_id(args)
    if not sid:
        raise AIActionError(
            "A destructive AI action must identify exactly one spectrum_id"
        )
    expected_revision = _expected_revision(args)
    if expected_revision is None:
        raise AIActionError(
            "expected_revision is required for destructive AI actions"
        )
    stored = get_store().get(sid)
    if stored is None:
        raise AIActionError(f"Spectrum {sid} not found")
    if stored.result is None:
        raise AIActionError(
            "Analyze the spectrum before previewing or executing a destructive "
            "AI action"
        )
    if stored.result_revision != expected_revision:
        raise RevisionConflict(stored.result_revision)
    return sid, stored, expected_revision


def _preview_update_integral(args: dict[str, Any]) -> dict[str, Any]:
    sid = str(args["spectrum_id"])
    index = int(args["index"])
    stored = _ensure_result(sid)
    result = stored.result
    if result is None or not hasattr(result, "integrals"):
        raise AIActionError("Selected result has no NMR integrals")
    if index < 0 or index >= len(result.integrals):
        raise AIActionError(f"Integral index {index} out of range")
    current = dict(result.integrals[index])
    target = {
        "start_ppm": float(args["start"]),
        "end_ppm": float(args["end"]),
        "center_ppm": float(args.get("center", (float(args["start"]) + float(args["end"])) / 2)),
    }
    return {
        "spectrum_id": sid,
        "action": "update_nmr_integral_range",
        "target": f"NMR integral #{index + 1}",
        "current": current,
        "proposed": target,
        "risks": ["积分归一化会随该区间重新计算", "该动作会保存为人工确认/AI 修改结果"],
    }


def _preview_delete_peak(args: dict[str, Any]) -> dict[str, Any]:
    sid = str(args["spectrum_id"])
    stored = _ensure_result(sid)
    result = stored.result
    if result is None:
        raise AIActionError("No result available")
    index = args.get("index")
    position = args.get("position")
    channel = args.get("channel")
    target: dict[str, Any] | None = None
    if channel and isinstance(result.metrics.get("channel_peaks"), dict):
        ch = result.metrics["channel_peaks"].get(str(channel))
        if not ch:
            raise AIActionError(f"Channel {channel} not found")
        peaks = ch.get("peaks", [])
        remove_idx = int(index) if index is not None else -1
        if position is not None and peaks:
            target_pos = float(position)
            remove_idx = min(range(len(peaks)), key=lambda i: abs(float(peaks[i].get("position", 0)) - target_pos))
        if remove_idx < 0 or remove_idx >= len(peaks):
            raise AIActionError("Peak index out of range")
        target = dict(peaks[remove_idx])
    else:
        peaks = result.peaks
        remove_idx = int(index) if index is not None else -1
        if position is not None and peaks:
            target_pos = float(position)
            remove_idx = min(range(len(peaks)), key=lambda i: abs(peaks[i].position - target_pos))
        if remove_idx < 0 or remove_idx >= len(peaks):
            raise AIActionError("Peak index out of range")
        peak = peaks[remove_idx]
        target = {"position": peak.position, "intensity": peak.intensity, "area": peak.area, "width": peak.width}
    return {
        "spectrum_id": sid,
        "action": "delete_peak",
        "target": f"{channel + ' ' if channel else ''}peak",
        "current": target,
        "proposed": {"remove": True},
        "risks": ["峰面积百分比和峰数会重新计算", "该动作可通过撤销恢复上一版本"],
    }


def _preview_hplc_reintegrate(args: dict[str, Any]) -> dict[str, Any]:
    sid = str(args["spectrum_id"])
    channel = str(args["channel"])
    index = int(args["index"])
    start = float(args["start"])
    end = float(args["end"])
    stored = _ensure_result(sid)
    result = stored.result
    channel_peaks = result.metrics.get("channel_peaks") if result else None
    if not isinstance(channel_peaks, dict) or channel not in channel_peaks:
        raise AIActionError(f"Channel {channel} not found")
    peaks = channel_peaks[channel].get("peaks", [])
    if index < 0 or index >= len(peaks):
        raise AIActionError("Peak index out of range")
    return {
        "spectrum_id": sid,
        "action": "hplc_reintegrate_peak",
        "target": f"{channel} peak #{index + 1}",
        "current": dict(peaks[index]),
        "proposed": {"begin_time": start, "end_time": end, "baseline": "linear"},
        "risks": ["峰高、面积、宽度和面积百分比会重新计算"],
    }


def preview_ai_action(name: str, args: dict[str, Any]) -> dict[str, Any]:
    with _ACTION_LOCK:
        spec = _ACTIONS.get(name)
        if spec is None:
            raise AIActionError(f"Unknown AI action: {name}")
        _, _, expected_revision = _require_destructive_revision(spec, args)
        if name == "update_nmr_integral_range":
            preview = _preview_update_integral(args)
        elif name == "delete_peak":
            preview = _preview_delete_peak(args)
        elif name == "hplc_reintegrate_peak":
            preview = _preview_hplc_reintegrate(args)
        elif name == "nmr_rebuild_multiplets":
            ranges = args.get("ranges") or []
            preview = {
                "spectrum_id": str(args.get("spectrum_id", "")),
                "action": name,
                "target": "NMR multiplet table",
                "current": {"mode": "current automatic/manual result"},
                "proposed": {"ranges": ranges, "n_ranges": len(ranges)},
                "risks": ["multiplet 类型和 J 值会按新区间重建"],
            }
        elif name == "undo_last_ai_action":
            preview = {
                "spectrum_id": str(args.get("spectrum_id", "")),
                "action": name,
                "target": "last AI-modified result",
                "current": {
                    "last_ai_action": get_store().get_last_ai_action(
                        str(args.get("spectrum_id", ""))
                    )
                },
                "proposed": {"restore_previous_version": True},
                "risks": ["只撤销最近一个尚未撤销的 AI 数据修改"],
            }
        else:
            preview = {
                "action": name,
                "target": "analysis workflow",
                "current": {},
                "proposed": args,
                "risks": [
                    "非破坏性动作"
                    if not spec.destructive
                    else "执行后会保存结果版本"
                ],
            }
        if expected_revision is not None:
            preview["expected_revision"] = expected_revision
            preview["preview_token"] = _issue_preview_token(name, args)
        return preview


def _normalize_integrals(integrals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive = [float(item.get("raw_area", 0)) for item in integrals if float(item.get("raw_area", 0)) > 0]
    min_area = min(positive) if positive else 0.0
    for item in integrals:
        raw = float(item.get("raw_area", 0))
        item["raw_area"] = round(raw, 4)
        item["relative_area"] = round(raw / min_area, 2) if min_area > 0 and raw > 0 else 0
    return integrals


def _integrate_xy(x, y, start: float, end: float, technique: str) -> dict[str, float]:
    lo, hi = sorted((start, end))
    mask = (x >= lo) & (x <= hi)
    seg_x = x[mask]
    seg_y = y[mask]
    if len(seg_x) < 2:
        return {"area": 0.0, "height": 0.0, "apex_time": (lo + hi) / 2}
    if technique == "HPLC":
        baseline = np.linspace(float(seg_y[0]), float(seg_y[-1]), len(seg_y))
        corrected = np.maximum(seg_y - baseline, 0)
        apex_idx = int(np.argmax(corrected))
        area = float(np.trapezoid(corrected, seg_x)) * 60.0
        height = float(corrected[apex_idx])
    else:
        apex_idx = int(np.argmax(seg_y))
        area = float(trapezoidal_integrate(x, y, lo, hi))
        height = float(seg_y[apex_idx])
    return {
        "area": round(area, 4),
        "height": round(height, 4),
        "apex_time": round(float(seg_x[apex_idx]), 4),
    }


def _save_ai_result(
    action_name: str,
    args: dict[str, Any],
    stored: StoredSpectrum,
    result: AnalysisResult,
    note: str,
) -> dict[str, Any]:
    expected_revision = _expected_revision(args)
    saved = get_store().commit_ai_action(
        stored.id,
        result,
        action_name,
        args,
        note=note,
        expected_revision=(
            stored.result_revision if expected_revision is None else expected_revision
        ),
        expected_spectrum_revision=stored.spectrum_revision,
    )
    if saved is None:
        raise AIActionError(f"Spectrum {stored.id} not found")
    return {
        "ok": True,
        "spectrum_id": stored.id,
        "version": saved.version,
        "result_revision": saved.result_revision,
        "ai_action_id": saved.action_id,
        "previous_version": saved.previous_version,
    }


def action_update_integral_range(args: dict[str, Any]) -> dict[str, Any]:
    sid = str(args["spectrum_id"])
    index = int(args["index"])
    start = float(args["start"])
    end = float(args["end"])
    stored = _ensure_result(sid)
    result = stored.result
    if result is None or not hasattr(result, "integrals"):
        raise AIActionError("Selected result has no NMR integrals")
    integrals = [dict(item) for item in result.integrals]
    if index < 0 or index >= len(integrals):
        raise AIActionError(f"Integral index {index} out of range")
    x = np.asarray(stored.spectrum.x_data, dtype=float)
    y = np.asarray(stored.spectrum.y_data, dtype=float)
    center = float(args.get("center", (start + end) / 2))
    calc = _integrate_xy(x, np.abs(y), start, end, stored.spectrum.technique.value)
    integrals[index].update({
        "start_ppm": round(start, 4),
        "end_ppm": round(end, 4),
        "center_ppm": round(center, 4),
        "raw_area": calc["area"],
        "intensity": calc["height"],
    })
    result.integrals = _normalize_integrals(integrals)
    result.total_integral = sum(float(item.get("raw_area", 0)) for item in result.integrals)
    result.metrics["manual_confirmed"] = True
    result.metrics["ai_modified"] = True
    result.metrics["quality"] = assess_quality(stored.spectrum, result)
    saved = _save_ai_result(
        "update_nmr_integral_range",
        args,
        stored,
        result,
        note="AI updated NMR integral range",
    )
    return {**saved, "integral": result.integrals[index]}


def action_delete_peak(args: dict[str, Any]) -> dict[str, Any]:
    sid = str(args["spectrum_id"])
    index = args.get("index")
    position = args.get("position")
    channel = args.get("channel")
    stored = _ensure_result(sid)
    result = stored.result
    if result is None:
        raise AIActionError("No result available")

    removed: dict[str, Any] | None = None
    channel_peaks = result.metrics.get("channel_peaks")
    if channel and isinstance(channel_peaks, dict):
        ch = channel_peaks.get(str(channel))
        if not ch:
            raise AIActionError(f"Channel {channel} not found")
        peaks = [dict(p) for p in ch.get("peaks", [])]
        remove_idx = int(index) if index is not None else -1
        if position is not None:
            target = float(position)
            remove_idx = min(range(len(peaks)), key=lambda i: abs(float(peaks[i].get("position", 0)) - target))
        if remove_idx < 0 or remove_idx >= len(peaks):
            raise AIActionError("Peak index out of range")
        removed = peaks.pop(remove_idx)
        total = sum(float(p.get("area", 0) or 0) for p in peaks)
        for p in peaks:
            p["area_percent"] = round(float(p.get("area", 0) or 0) / total * 100, 2) if total > 0 else 0
        ch["peaks"] = peaks
        ch["total_area"] = round(total, 2)
        ch["source"] = "ai_modified"
        result.metrics["channel_peaks"] = channel_peaks
    else:
        peaks = list(result.peaks)
        remove_idx = int(index) if index is not None else -1
        if position is not None:
            target = float(position)
            remove_idx = min(range(len(peaks)), key=lambda i: abs(peaks[i].position - target))
        if remove_idx < 0 or remove_idx >= len(peaks):
            raise AIActionError("Peak index out of range")
        peak = peaks.pop(remove_idx)
        removed = {"position": peak.position, "intensity": peak.intensity, "area": peak.area}
        result.peaks = peaks

    result.metrics["n_peaks"] = len(result.peaks)
    result.metrics["manual_confirmed"] = True
    result.metrics["ai_modified"] = True
    result.metrics["quality"] = assess_quality(stored.spectrum, result)
    saved = _save_ai_result(
        "delete_peak",
        args,
        stored,
        result,
        note="AI deleted peak",
    )
    return {**saved, "removed": removed}


def action_hplc_reintegrate_peak(args: dict[str, Any]) -> dict[str, Any]:
    sid = str(args["spectrum_id"])
    channel = str(args["channel"])
    index = int(args["index"])
    start = float(args["start"])
    end = float(args["end"])
    stored = _ensure_result(sid)
    if stored.spectrum.technique.value != "HPLC":
        raise AIActionError("HPLC reintegration requires an HPLC spectrum")
    result = stored.result
    channel_peaks = result.metrics.get("channel_peaks") if result else None
    if not isinstance(channel_peaks, dict) or channel not in channel_peaks:
        raise AIActionError(f"Channel {channel} not found")
    peaks = [dict(p) for p in channel_peaks[channel].get("peaks", [])]
    if index < 0 or index >= len(peaks):
        raise AIActionError("Peak index out of range")
    channels = stored.spectrum.parameters.get("channels") or []
    ch_data = next((ch for ch in channels if ch.get("name") == channel), None)
    if ch_data is None:
        raise AIActionError(f"Raw channel {channel} not found")
    x = np.asarray(stored.spectrum.x_data, dtype=float)
    y = np.asarray(ch_data["y_data"], dtype=float)
    calc = _integrate_xy(x, y, start, end, "HPLC")
    peaks[index].update({
        "begin_time": round(start, 4),
        "end_time": round(end, 4),
        "position": calc["apex_time"],
        "area": calc["area"],
        "intensity": calc["height"],
        "height": calc["height"],
        "width": round(abs(end - start), 4),
    })
    total = sum(float(p.get("area", 0) or 0) for p in peaks)
    for p in peaks:
        p["area_percent"] = round(float(p.get("area", 0) or 0) / total * 100, 2) if total > 0 else 0
    channel_peaks[channel]["peaks"] = peaks
    channel_peaks[channel]["total_area"] = round(total, 2)
    channel_peaks[channel]["source"] = "ai_modified"
    result.metrics["channel_peaks"] = channel_peaks
    result.metrics["manual_confirmed"] = True
    result.metrics["ai_modified"] = True
    result.metrics["quality"] = assess_quality(stored.spectrum, result)
    saved = _save_ai_result(
        "hplc_reintegrate_peak",
        args,
        stored,
        result,
        note="AI reintegrated HPLC peak",
    )
    return {**saved, "peak": peaks[index], "total_area": round(total, 2)}


def action_nmr_rebuild_multiplets(args: dict[str, Any]) -> dict[str, Any]:
    sid = str(args["spectrum_id"])
    ranges = args.get("ranges") or []
    stored = _ensure_result(sid)
    if stored.spectrum.technique.value != "NMR":
        raise AIActionError("Multiplet rebuild requires an NMR spectrum")
    options = dict(stored.result.metrics.get("analysis_options", {}) if stored.result else {})
    options["multiplet_ranges"] = ranges
    result = AnalyzerRegistry.get("nmr")().analyze(stored.spectrum, options=options)
    result.metrics["manual_confirmed"] = True
    result.metrics["ai_modified"] = True
    result.metrics["quality"] = assess_quality(stored.spectrum, result)
    saved = _save_ai_result(
        "nmr_rebuild_multiplets",
        args,
        stored,
        result,
        note="AI rebuilt NMR multiplets",
    )
    return {**saved, "n_multiplets": len(result.multiplets), "multiplets": result.multiplets}


def action_apply_hplc_integration_events(args: dict[str, Any]) -> dict[str, Any]:
    sid = str(args["spectrum_id"])
    events = args.get("events") or []
    stored = _ensure_result(sid)
    if stored.spectrum.technique.value != "HPLC":
        raise AIActionError("Integration events require an HPLC spectrum")
    options = dict(stored.result.metrics.get("analysis_options", {}) if stored.result else {})
    options["integration_events"] = events
    result = AnalyzerRegistry.get("hplc")().analyze(stored.spectrum, options=options)
    result.metrics["manual_confirmed"] = True
    result.metrics["ai_modified"] = True
    result.metrics["quality"] = assess_quality(stored.spectrum, result)
    saved = _save_ai_result(
        "apply_hplc_integration_events",
        args,
        stored,
        result,
        note="AI applied HPLC integration events",
    )
    return {
        **saved,
        "events": events,
        "n_peaks": result.metrics.get("n_peaks"),
        "channel_peaks": result.metrics.get("channel_peaks", {}),
    }


def action_nmr_phase_correct(args: dict[str, Any]) -> dict[str, Any]:
    sid = str(args["spectrum_id"])
    stored = _ensure_result(sid)
    if stored.spectrum.technique.value != "NMR":
        raise AIActionError("Phase correction requires an NMR spectrum")
    options = dict(stored.result.metrics.get("analysis_options", {}) if stored.result else {})
    options["phase_zero_deg"] = float(args.get("phase_zero_deg", args.get("zero", 0)) or 0)
    options["phase_first_deg"] = float(args.get("phase_first_deg", args.get("first", 0)) or 0)
    if args.get("phase_pivot_ppm") is not None:
        options["phase_pivot_ppm"] = float(args["phase_pivot_ppm"])
    result = AnalyzerRegistry.get("nmr")().analyze(stored.spectrum, options=options)
    result.metrics["manual_confirmed"] = True
    result.metrics["ai_modified"] = True
    result.metrics["quality"] = assess_quality(stored.spectrum, result)
    saved = _save_ai_result(
        "nmr_phase_correct",
        args,
        stored,
        result,
        note="AI applied NMR phase correction",
    )
    return {
        **saved,
        "phase": {
            "zero_deg": options["phase_zero_deg"],
            "first_deg": options["phase_first_deg"],
            "pivot_ppm": options.get("phase_pivot_ppm"),
        },
        "summary": result.summary,
    }


def action_xrd_rietveld_refine(args: dict[str, Any]) -> dict[str, Any]:
    sid = str(args["spectrum_id"])
    stored = _ensure_result(sid)
    if stored.spectrum.technique.value != "XRD":
        raise AIActionError("Rietveld refinement requires an XRD spectrum")
    options = dict(stored.result.metrics.get("analysis_options", {}) if stored.result else {})
    options["rietveld_enabled"] = True
    if args.get("sigma_deg") is not None:
        options["rietveld_sigma_deg"] = float(args["sigma_deg"])
    result = AnalyzerRegistry.get("xrd")().analyze(stored.spectrum, options=options)
    result.metrics["manual_confirmed"] = True
    result.metrics["ai_modified"] = True
    result.metrics["quality"] = assess_quality(stored.spectrum, result)
    saved = _save_ai_result(
        "xrd_rietveld_refine",
        args,
        stored,
        result,
        note="AI ran XRD Rietveld refinement",
    )
    return {
        **saved,
        "rietveld_refinement": getattr(result, "rietveld_refinement", {}),
    }


def action_run_cross_inference(args: dict[str, Any]) -> dict[str, Any]:
    ids = [str(sid) for sid in args.get("ids", [])]
    if not ids:
        raise AIActionError("Provide ids")
    results = {}
    techniques = []
    sample_names = []
    for sid in ids:
        stored = _ensure_result(sid)
        results[sid] = stored.result
        techniques.append(stored.spectrum.technique.value)
        sample_names.append(stored.spectrum.metadata.name or stored.spectrum.source_file or sid)
    inference = InferenceEngine().analyze(results)
    report = build_report(", ".join(sample_names), sorted(set(techniques)), inference.technique_results, inference)
    return report.to_dict()


def action_undo_last_ai_action(args: dict[str, Any]) -> dict[str, Any]:
    sid = str(args["spectrum_id"])
    store = get_store()
    stored = store.get(sid)
    if stored is None:
        raise AIActionError(f"Spectrum {sid} not found")
    expected_revision = _expected_revision(args)
    try:
        restored = store.undo_last_ai_action(
            sid,
            expected_revision=(
                stored.result_revision if expected_revision is None else expected_revision
            ),
        )
    except KeyError as exc:
        raise AIActionError(str(exc.args[0])) from exc
    if restored is None:
        raise AIActionError("No AI action available to undo")
    return {
        "ok": True,
        "spectrum_id": sid,
        "undone_action": restored.action_name,
        "restored_from_version": restored.previous_version,
        "version": restored.version,
        "result_revision": restored.result_revision,
    }


def suggest_ai_actions(ids: list[str]) -> list[dict[str, Any]]:
    store = get_store()
    suggestions: list[dict[str, Any]] = []
    target_ids = ids or [item.id for item in store.list_all()[:10]]
    for sid in target_ids:
        stored = store.get(str(sid))
        if stored is None:
            continue
        try:
            stored = _ensure_result(str(sid))
        except AIActionError:
            continue
        result = stored.result
        if result is None:
            continue
        quality = result.metrics.get("quality") or assess_quality(stored.spectrum, result)
        if quality.get("status") in {"review", "poor"}:
            suggestions.append({
                "title": "质量状态需要复核",
                "reason": "; ".join((quality.get("warnings") or [])[:2]) or "质量评分偏低",
                "action": {"name": "run_cross_inference", "args": {"ids": [str(sid)]}},
                "severity": quality.get("status"),
            })
        if stored.spectrum.technique.value == "NMR":
            integrals = getattr(result, "integrals", []) or []
            if integrals:
                narrow = min(integrals, key=lambda item: abs(float(item.get("start_ppm", 0)) - float(item.get("end_ppm", 0))))
                if abs(float(narrow.get("start_ppm", 0)) - float(narrow.get("end_ppm", 0))) < 0.015:
                    suggestions.append({
                        "title": "NMR 积分区间过窄",
                        "reason": f"δ {narrow.get('center_ppm')} ppm 的积分窗口小于 0.015 ppm",
                        "action": {
                            "name": "update_nmr_integral_range",
                            "args": {
                                "spectrum_id": str(sid),
                                "index": integrals.index(narrow),
                                "start": round(float(narrow.get("center_ppm", 0)) + 0.03, 4),
                                "end": round(float(narrow.get("center_ppm", 0)) - 0.03, 4),
                            },
                        },
                        "severity": "review",
                    })
            if not getattr(result, "multiplets", []):
                suggestions.append({
                    "title": "NMR 尚未解析 multiplet",
                    "reason": "当前结果没有 multiplet 表，可按峰群重建",
                    "action": {"name": "nmr_rebuild_multiplets", "args": {"spectrum_id": str(sid), "ranges": []}},
                    "severity": "review",
                })
        if stored.spectrum.technique.value == "HPLC":
            channel_peaks = result.metrics.get("channel_peaks") or {}
            if isinstance(channel_peaks, dict):
                for channel, channel_data in channel_peaks.items():
                    for index, peak in enumerate(channel_data.get("peaks", [])[:10]):
                        peak_type = str(peak.get("type", ""))
                        area_percent = float(peak.get("area_percent", 0) or 0)
                        if peak_type in {"VV", "BV", "VB"} or area_percent < 0.1:
                            suggestions.append({
                                "title": "HPLC 峰建议复核积分",
                                "reason": f"{channel} tR={float(peak.get('position', 0)):.3f} min 类型 {peak_type or '未知'}",
                                "action": {
                                    "name": "hplc_reintegrate_peak",
                                    "args": {
                                        "spectrum_id": str(sid),
                                        "channel": channel,
                                        "index": index,
                                        "start": peak.get("begin_time", peak.get("position", 0)),
                                        "end": peak.get("end_time", peak.get("position", 0)),
                                    },
                                },
                                "severity": "review",
                            })
                            break
        if stored.spectrum.technique.value == "XRD" and hasattr(result, "phase_matches"):
            matches = getattr(result, "phase_matches", []) or []
            if matches and float(matches[0].get("match_score", 0)) >= 20:
                suggestions.append({
                    "title": "XRD 可执行 Rietveld 精修",
                    "reason": f"主相 {matches[0].get('phase_name')} 已匹配，可进一步拟合尺度、背景和峰宽",
                    "action": {"name": "xrd_rietveld_refine", "args": {"spectrum_id": str(sid)}},
                    "severity": "info",
                })
        if len(suggestions) >= 20:
            break
    return suggestions


_ACTIONS: dict[str, AIActionSpec] = {
    "update_nmr_integral_range": AIActionSpec(
        "update_nmr_integral_range",
        "Modify one NMR integral range and recalculate its area.",
        {"spectrum_id": "string", "index": "integer", "start": "number", "end": "number", "center": "number optional"},
        action_update_integral_range,
        destructive=True,
    ),
    "delete_peak": AIActionSpec(
        "delete_peak",
        "Delete/cancel a peak by index or nearest position. For HPLC, include channel.",
        {"spectrum_id": "string", "index": "integer optional", "position": "number optional", "channel": "string optional"},
        action_delete_peak,
        destructive=True,
    ),
    "hplc_reintegrate_peak": AIActionSpec(
        "hplc_reintegrate_peak",
        "Reintegrate an HPLC channel peak over a new time range.",
        {"spectrum_id": "string", "channel": "string", "index": "integer", "start": "number", "end": "number"},
        action_hplc_reintegrate_peak,
        destructive=True,
    ),
    "nmr_rebuild_multiplets": AIActionSpec(
        "nmr_rebuild_multiplets",
        "Rebuild NMR multiplets from manual ranges.",
        {"spectrum_id": "string", "ranges": [{"start": "number", "end": "number"}]},
        action_nmr_rebuild_multiplets,
        destructive=True,
    ),
    "run_cross_inference": AIActionSpec(
        "run_cross_inference",
        "Run structured cross-technique inference and evidence table.",
        {"ids": ["string"]},
        action_run_cross_inference,
        destructive=False,
    ),
    "apply_hplc_integration_events": AIActionSpec(
        "apply_hplc_integration_events",
        "Apply HPLC integration events such as delete, forced type, labels, baseline mode, or area factors.",
        {"spectrum_id": "string", "events": [{"channel": "string optional", "start": "number", "end": "number", "mode": "delete|force_bb|force_vv|name", "name": "string optional"}]},
        action_apply_hplc_integration_events,
        destructive=True,
    ),
    "nmr_phase_correct": AIActionSpec(
        "nmr_phase_correct",
        "Reanalyze NMR with zero/first-order phase correction parameters.",
        {"spectrum_id": "string", "phase_zero_deg": "number", "phase_first_deg": "number", "phase_pivot_ppm": "number optional"},
        action_nmr_phase_correct,
        destructive=True,
    ),
    "xrd_rietveld_refine": AIActionSpec(
        "xrd_rietveld_refine",
        "Run ChemApp reference-pattern constrained Rietveld-like refinement for an XRD result.",
        {"spectrum_id": "string", "sigma_deg": "number optional"},
        action_xrd_rietveld_refine,
        destructive=True,
    ),
    "undo_last_ai_action": AIActionSpec(
        "undo_last_ai_action",
        "Undo the latest AI data operation for one spectrum by restoring the previous result version.",
        {"spectrum_id": "string"},
        action_undo_last_ai_action,
        destructive=True,
    ),
}


def action_is_destructive(name: str) -> bool:
    """Whether executing action name mutates stored analysis data.

    Used by the execute route to require an admin token for destructive
    actions; preview_token + revision checks still apply unchanged.
    """
    spec = _ACTIONS.get(name)
    return bool(spec and spec.destructive)


def list_ai_actions() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "schema": {
                **spec.schema,
                **(
                    {"expected_revision": "non-negative integer required"}
                    if spec.destructive
                    else {}
                ),
            },
            "destructive": spec.destructive,
        }
        for spec in _ACTIONS.values()
    ]


def execute_ai_action(
    name: str,
    args: dict[str, Any],
    *,
    preview_token: str | None = None,
) -> dict[str, Any]:
    with _ACTION_LOCK:
        spec = _ACTIONS.get(name)
        if spec is None:
            raise AIActionError(f"Unknown AI action: {name}")
        if spec.destructive:
            _verify_preview_token(preview_token, name, args)
        _require_destructive_revision(spec, args)
        return spec.handler(args)
