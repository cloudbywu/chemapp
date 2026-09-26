from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from app.analysis.base import BaseAnalyzer
from app.analysis.models import AnalysisResult
from app.analysis.utils import estimate_noise_region
from app.core.models import Peak, Spectrum


def _detect_channel_peaks(x: np.ndarray, y_raw: np.ndarray, options: dict[str, Any] | None = None) -> list[dict]:
    options = options or {}
    y = _subtract_background(y_raw)
    noise = estimate_noise_region(y, fraction=0.05)
    height_factor = float(options.get("height_factor", 1.5))
    prominence_factor = float(options.get("prominence_factor", 2))
    min_peak_distance = int(options.get("min_peak_distance", 2))

    peak_indices, peak_props = signal.find_peaks(
        y, height=noise * height_factor, prominence=noise * prominence_factor, distance=min_peak_distance, width=(1, 200),
    )

    dt = x[1] - x[0] if len(x) > 1 else 0.0
    peaks: list[dict] = []
    for i in range(len(peak_indices)):
        idx = int(peak_indices[i])
        width_pts = float(peak_props.get("widths", [0])[i] if "widths" in peak_props else 0)
        fwhm_pts = width_pts if width_pts > 0 else 10
        fwhm_min = fwhm_pts * dt if dt > 0 else 0.0
        height = float(peak_props.get("peak_heights", [0])[i] if "peak_heights" in peak_props else y[idx])

        # Integrate over 4×FWHM window, area in mAU·sec (matches instrument report)
        half_win = int(fwhm_pts * 2) if fwhm_pts > 0 else 10
        half_win = max(half_win, 8)
        lo = max(0, idx - half_win)
        hi = min(len(y), idx + half_win)
        if hi > lo:
            area = float(np.trapezoid(y[lo:hi], x[lo:hi])) * 60.0
        else:
            area = 0.0

        peaks.append({
            "position": round(float(x[idx]), 4),
            "intensity": round(height, 4),
            "width": round(fwhm_min, 4) if fwhm_min > 0 else None,
            "area": round(area, 2),
            "area_percent": 0.0,
            "type": "",
            "name": "",
        })

    total_area = sum(p["area"] for p in peaks)
    for peak in peaks:
        peak["area_percent"] = round(peak["area"] / total_area * 100, 2) if total_area > 0 else 0.0

    peaks.sort(key=lambda p: p["position"])
    return peaks


def apply_integration_events(
    x: np.ndarray,
    peaks: list[dict],
    events: list[dict[str, Any]],
) -> list[dict]:
    """Apply simple time-window integration events to an instrument/computed peak table."""
    if not events:
        return peaks
    updated = [dict(p) for p in peaks]
    for event in sorted(events, key=lambda item: float(item.get("time", item.get("start", 0)) or 0)):
        mode = str(event.get("mode") or event.get("type") or "").lower()
        start = event.get("start", event.get("time"))
        end = event.get("end", start)
        try:
            lo, hi = sorted((float(start), float(end)))
        except (TypeError, ValueError):
            continue
        for peak in updated:
            rt = float(peak.get("position", 0) or 0)
            if not (lo <= rt <= hi):
                continue
            if mode in {"delete", "suppress", "ignore", "off"}:
                peak["_deleted_by_event"] = True
            elif mode in {"force_bb", "bb"}:
                peak["type"] = "BB"
            elif mode in {"force_vv", "vv"}:
                peak["type"] = "VV"
            elif mode in {"name", "label"} and event.get("name"):
                peak["name"] = str(event["name"])
            if event.get("baseline"):
                peak["baseline"] = str(event["baseline"])
            if event.get("area_factor") not in (None, ""):
                try:
                    factor = float(event["area_factor"])
                    peak["area"] = round(float(peak.get("area", 0) or 0) * factor, 2)
                except (TypeError, ValueError):
                    pass
    filtered = [p for p in updated if not p.pop("_deleted_by_event", False)]
    total_area = sum(float(p.get("area", 0) or 0) for p in filtered)
    for peak in filtered:
        peak["area_percent"] = round(float(peak.get("area", 0) or 0) / total_area * 100, 2) if total_area > 0 else 0.0
    return filtered


def _subtract_background(y: np.ndarray) -> np.ndarray:
    window = max(len(y) // 10, 5)
    bg = np.zeros_like(y)
    for i in range(len(y)):
        lo = max(0, i - window)
        hi = min(len(y), i + window + 1)
        bg[i] = np.min(y[lo:hi])
    bg_smooth = np.convolve(bg, np.ones(window) / window, mode="same")
    return np.maximum(y - bg_smooth, 0)


class HPLCAnalyzer(BaseAnalyzer):
    technique = "hplc"

    def analyze(self, spectrum: Spectrum, options: dict[str, Any] | None = None) -> AnalysisResult:
        options = options or {}
        x = spectrum.x_data
        channels = spectrum.parameters.get("channels", [])
        integration_events = list(options.get("integration_events") or [])

        if not channels:
            y = spectrum.y_data
            peaks = apply_integration_events(x, _detect_channel_peaks(x, y, options), integration_events)
            main_peaks = [Peak(
                position=p["position"], intensity=p["intensity"],
                width=p["width"], area=p["area"],
            ) for p in peaks]

            return AnalysisResult(
                technique=spectrum.technique,
                peaks=main_peaks,
                metrics={
                    "n_peaks": len(main_peaks),
                    "time_range_min": (round(float(x[0]), 3), round(float(x[-1]), 3)),
                    "max_intensity_mau": round(float(y.max()), 2),
                    "total_area": round(sum(p["area"] for p in peaks), 2),
                    "channel_peaks": {},
                    "analysis_options": options,
                    "integration_events": integration_events,
                },
                summary=f"HPLC: {len(main_peaks)} peaks detected.",
            )

        channel_peaks: dict[str, dict] = {}
        all_peaks: list[Peak] = []
        total_peaks = 0

        for ch in channels:
            ch_name = ch["name"]
            instrument_peaks = ch.get("instrument_peaks") or []
            ch_peaks = (
                [dict(p) for p in instrument_peaks]
                if instrument_peaks
                else _detect_channel_peaks(x, np.array(ch["y_data"]), options)
            )
            channel_events = [
                event for event in integration_events
                if not event.get("channel") or str(event.get("channel")) == ch_name
            ]
            ch_peaks = apply_integration_events(x, ch_peaks, channel_events)
            total_area = round(sum(float(p.get("area", 0.0)) for p in ch_peaks), 2)
            channel_peaks[ch_name] = {
                "wavelength_nm": ch.get("wavelength_nm"),
                "color": ch.get("color", "#3b82f6"),
                "peaks": ch_peaks,
                "total_area": total_area,
                "source": "instrument_record" if instrument_peaks else "computed",
            }
            total_peaks += len(ch_peaks)

            for p in ch_peaks:
                all_peaks.append(Peak(
                    position=p["position"], intensity=p["intensity"],
                    width=p["width"], area=p["area"],
                ))

        ch_names = ", ".join(
            f"{ch['name']} ({ch['wavelength_nm']}nm)"
            for ch in channels
        )
        ch_summaries = []
        for ch_name, ch_data in channel_peaks.items():
            n = len(ch_data["peaks"])
            if n > 0:
                top = max(ch_data["peaks"], key=lambda p: p.get("area", 0.0))
                ch_summaries.append(
                    f"{ch_name}: {n} peaks, main @ {top['position']:.3f} min"
                )
        summary = (
            f"HPLC ({ch_names}): {total_peaks} peaks total. "
            + "; ".join(ch_summaries) + "."
        )

        return AnalysisResult(
            technique=spectrum.technique,
            peaks=all_peaks,
            metrics={
                "n_peaks": total_peaks,
                "time_range_min": (round(float(x[0]), 3), round(float(x[-1]), 3)),
                "n_channels": len(channels),
                "channel_peaks": channel_peaks,
                "integration_source": (
                    "instrument_record"
                    if any(ch.get("instrument_peaks") for ch in channels)
                    else "computed"
                ),
                "instrument_result_source": spectrum.parameters.get("instrument_result_source", ""),
                "analysis_options": options,
                "integration_events": integration_events,
            },
            summary=summary,
        )
