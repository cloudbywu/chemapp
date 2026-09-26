from __future__ import annotations

from typing import Any

import numpy as np

from app.analysis.base import BaseAnalyzer
from app.analysis.models import AnalysisResult
from app.core.models import Peak, Spectrum


class ElectrochemAnalyzer(BaseAnalyzer):
    technique = "electrochem"

    def analyze(self, spectrum: Spectrum, options: dict[str, Any] | None = None) -> AnalysisResult:
        sub_type = spectrum.parameters.get("sub_type", "CV")

        if sub_type == "EIS":
            return self._analyze_eis(spectrum)
        return self._analyze_cv(spectrum)

    def _analyze_cv(self, spectrum: Spectrum) -> AnalysisResult:
        x = np.asarray(spectrum.x_data, dtype=float)
        y = np.asarray(spectrum.y_data, dtype=float)
        scan_rate = spectrum.parameters.get("scan_rate_v_s", 0.05)

        # Split at the scan vertex (potential reversal point) instead of the
        # midpoint: the vertex is the first extremum in the sweep direction.
        if len(x) < 2:
            return AnalysisResult(
                technique=spectrum.technique,
                peaks=[],
                metrics={
                    "sub_type": "CV",
                    "scan_rate_v_s": scan_rate,
                    "error": "insufficient_points",
                },
                summary="CV: insufficient data points.",
            )
        # Determine initial sweep direction from the first nonzero step so a
        # full cycle that returns to the starting potential is still split at
        # the first reversal (the true vertex), not at the midpoint or start.
        direction = 0
        for index in range(1, len(x)):
            if x[index] != x[0]:
                direction = 1 if x[index] > x[0] else -1
                break
        if direction >= 0:
            idx_vertex = int(np.argmax(x))
        else:
            idx_vertex = int(np.argmin(x))
        x_forward = x[: idx_vertex + 1]
        y_forward = y[: idx_vertex + 1]
        x_reverse = x[idx_vertex:]
        y_reverse = y[idx_vertex:]

        # Forward scan (anodic): find maximum (oxidation peak)
        # Reverse scan (cathodic): find minimum (reduction peak)
        if len(y_forward) > 5:
            idx_fwd_max = int(np.argmax(y_forward))
            peak_anodic = Peak(
                position=float(x_forward[idx_fwd_max]),
                intensity=float(y_forward[idx_fwd_max]),
                assignment="oxidation",
            )
        else:
            peak_anodic = Peak(position=0, intensity=0, assignment="oxidation")

        if len(y_reverse) > 5:
            idx_rev_min = int(np.argmin(y_reverse))
            peak_cathodic = Peak(
                position=float(x_reverse[idx_rev_min]),
                intensity=float(y_reverse[idx_rev_min]),
                assignment="reduction",
            )
        else:
            peak_cathodic = Peak(position=0, intensity=0, assignment="reduction")

        ep_a = peak_anodic.position
        ep_c = peak_cathodic.position
        ip_a = peak_anodic.intensity
        ip_c = peak_cathodic.intensity

        delta_ep = abs(ep_a - ep_c) if (ep_a and ep_c) else None
        e_formal = (ep_a + ep_c) / 2 if (ep_a and ep_c) else None
        ip_ratio = abs(ip_a / ip_c) if (ip_c and abs(ip_c) > 1e-20) else None

        peaks = [peak_anodic, peak_cathodic]
        peaks = [p for p in peaks if abs(p.intensity) > 0]

        # Charge integration (coulomb)
        if len(x) > 1:
            q_forward = abs(float(np.trapezoid(y_forward[y_forward > 0], x_forward[y_forward > 0]))) if np.any(y_forward > 0) else 0
            q_reverse = abs(float(np.trapezoid(y_reverse[y_reverse < 0], x_reverse[y_reverse < 0]))) if np.any(y_reverse < 0) else 0
        else:
            q_forward = q_reverse = 0

        segments_meta = spectrum.parameters.get("segments_meta", [])
        instrument_ep = None
        instrument_ip = None
        if segments_meta and len(segments_meta) >= 1:
            seg1 = segments_meta[0]
            instrument_ep = seg1.get("Ep")
            instrument_ip = seg1.get("ip_a")

        metrics = {
            "scan_rate_v_s": scan_rate,
            "ep_anodic_v": round(ep_a, 4) if ep_a is not None else None,
            "ep_cathodic_v": round(ep_c, 4) if ep_c is not None else None,
            "ip_anodic_a": round(ip_a, 10) if ip_a else None,
            "ip_cathodic_a": round(ip_c, 10) if ip_c else None,
            "delta_ep_v": round(delta_ep, 4) if delta_ep is not None else None,
            "e_formal_v": round(e_formal, 4) if e_formal is not None else None,
            "ip_ratio": round(ip_ratio, 3) if ip_ratio else None,
            "q_forward_c": round(q_forward, 8),
            "q_reverse_c": round(q_reverse, 8),
            "instrument_ep_v": instrument_ep,
            "instrument_ip_a": instrument_ip,
            "sub_type": "CV",
        }

        summary_parts = [
            f"CV at {scan_rate} V/s: ",
            f"Ep,a={ep_a:.3f}V" if ep_a is not None else "",
            f"Ep,c={ep_c:.3f}V" if ep_c is not None else "",
            f"ip,a={ip_a:.2e}A" if ip_a else "",
            f"ip,c={ip_c:.2e}A" if ip_c else "",
        ]
        summary = ", ".join([s for s in summary_parts if s]) + "."

        return AnalysisResult(
            technique=spectrum.technique,
            peaks=peaks,
            metrics=metrics,
            summary=summary,
        )

    def _analyze_eis(self, spectrum: Spectrum) -> AnalysisResult:
        z_prime = np.asarray(spectrum.x_data, dtype=float)
        z_double = np.asarray(spectrum.y_data, dtype=float)
        if len(z_prime) == 0:
            return AnalysisResult(
                technique=spectrum.technique,
                peaks=[],
                metrics={
                    "sub_type": "EIS",
                    "error": "insufficient_points",
                },
                summary="EIS: insufficient data points.",
            )

        # High-frequency intercept: leftmost Z' value (best available proxy
        # for the solution resistance Rs without frequency metadata).
        rs = float(np.min(z_prime))
        # Rct = total polarization resistance minus Rs = semicircle diameter.
        rct = max(0.0, float(np.max(z_prime)) - rs)

        # W_max = frequency at max -Z"
        idx_max_zd = int(np.argmax(z_double)) if len(z_double) > 0 else 0
        zd_max = float(z_double[idx_max_zd]) if idx_max_zd < len(z_double) else 0

        metrics = {
            "sub_type": "EIS",
            "rs_ohm": round(rs, 1),
            "rct_ohm": round(rct, 1),
            "zd_max_ohm": round(zd_max, 1),
            "z_range_real": (round(float(z_prime.min()), 1), round(float(z_prime.max()), 1)),
            "z_range_imag": (round(float(z_double.min()), 1), round(float(z_double.max()), 1)),
        }

        summary = (
            f"EIS: Rs ≈ {rs:.1f} Ω, Rct ≈ {rct:.1f} Ω, "
            f"max -Z\" = {zd_max:.1f} Ω"
        )

        return AnalysisResult(
            technique=spectrum.technique,
            peaks=[],
            metrics=metrics,
            summary=summary,
        )
