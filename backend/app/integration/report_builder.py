from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.integration.models import ComprehensiveReport, InferenceResult


def build_report(
    sample_name: str,
    techniques: list[str],
    technique_data: dict[str, dict[str, Any]],
    inference: InferenceResult,
) -> ComprehensiveReport:
    lines: list[str] = []

    lines.append("# Comprehensive Analysis Report")
    lines.append("")
    lines.append(f"**Sample:** {sample_name}")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Techniques:** {', '.join(techniques)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Per-Technique Summary")
    lines.append("")
    for tech, data in technique_data.items():
        if tech.startswith("_"):
            continue
        lines.append(f"### {tech}")
        lines.append("")
        summary = data.get("summary", "No summary available.")
        lines.append(f"{summary}")
        lines.append("")

        metrics = {k: v for k, v in data.items() if k not in ("summary",)}
        if metrics:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in metrics.items():
                val_str = _format_value(v)
                lines.append(f"| {k} | {val_str} |")
            lines.append("")

    evidence = technique_data.get("_evidence_table", {}).get("items", [])
    if evidence:
        lines.append("---")
        lines.append("")
        lines.append("## Evidence Table")
        lines.append("")
        lines.append("| Technique | Evidence | Support | Confidence |")
        lines.append("|---|---|---|---:|")
        for item in evidence:
            lines.append(
                f"| {item.get('technique', '')} | {item.get('evidence', '')} | "
                f"{item.get('support', '')} | {float(item.get('confidence', 0)):.2f} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Cross-Validation Results")
    lines.append("")
    if inference.cross_validations:
        lines.append("| Pair | Metric | Score | Detail |")
        lines.append("|------|--------|-------|--------|")
        for cv in inference.cross_validations:
            pair_str = f"{cv.pair[0]} ↔ {cv.pair[1]}"
            lines.append(f"| {pair_str} | {cv.metric} | {cv.score:.2f} | {cv.detail} |")
        lines.append("")
    else:
        lines.append("No cross-validation performed (need 2+ techniques).")
        lines.append("")

    lines.append("---")
    lines.append("")

    lines.append("## Scores")
    lines.append("")
    lines.append(f"- **Consistency Score:** {inference.consistency_score:.2f}")
    lines.append(f"- **Confidence:** {inference.confidence:.2f}")
    lines.append("")

    if inference.anomalies:
        lines.append("## Anomalies & Warnings")
        lines.append("")
        for a in inference.anomalies:
            lines.append(f"- {a}")
        lines.append("")

    if inference.conclusions:
        lines.append("## Conclusions")
        lines.append("")
        for c in inference.conclusions:
            lines.append(f"- {c}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Overall Assessment")
    lines.append("")
    lines.append(inference.overall_assessment)
    lines.append("")

    markdown = "\n".join(lines)

    return ComprehensiveReport(
        sample_name=sample_name,
        techniques=techniques,
        inference=inference,
        generated_at=datetime.now(timezone.utc).isoformat(),
        report_markdown=markdown,
    )


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, list):
        items = [_format_value(x) for x in v[:5]]
        if len(v) > 5:
            items.append("...")
        return ", ".join(items)
    if v is None:
        return "—"
    return str(v)
