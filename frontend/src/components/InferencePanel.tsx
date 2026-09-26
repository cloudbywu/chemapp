import { useState } from "react";
import { analyzeBatch, compareHplcBatch, downloadBatchCsvZip, downloadDocxReport, downloadHtmlReport, downloadHplcComparisonCsv, downloadMarkdownReport, getBatchQuality, getBatchWorkbench, runInference } from "../services/api";
import { useLang } from "../i18n/LangContext";
import type { InferenceResponse, SpectrumListItem } from "../types/spectrum";

interface Props {
  spectra: SpectrumListItem[];
  onDataChanged?: () => void;
}

export default function InferencePanel({ spectra, onDataChanged }: Props) {
  const { t } = useLang();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [batching, setBatching] = useState(false);
  const [error, setError] = useState("");
  const [batchSummary, setBatchSummary] = useState("");
  const [hplcComparison, setHplcComparison] = useState<Awaited<ReturnType<typeof compareHplcBatch>> | null>(null);
  const [qualitySummary, setQualitySummary] = useState<Awaited<ReturnType<typeof getBatchQuality>> | null>(null);
  const [workbench, setWorkbench] = useState<Awaited<ReturnType<typeof getBatchWorkbench>> | null>(null);

  const toggle = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  };

  const handleRun = async () => {
    if (selected.size === 0) return;
    setError("");
    setRunning(true);
    try {
      const data = await runInference(Array.from(selected));
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.inference.inferenceFailed);
    } finally {
      setRunning(false);
    }
  };

  const handleReport = async () => {
    if (selected.size === 0) return;
    try {
      await downloadMarkdownReport(Array.from(selected), "ChemApp Multi-Spectrum Report");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.inference.reportFailed);
    }
  };

  const handleHtmlReport = async () => {
    if (selected.size === 0) return;
    try {
      await downloadHtmlReport(Array.from(selected), "ChemApp Multi-Spectrum Report");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.inference.htmlReportFailed);
    }
  };

  const handleDocxReport = async () => {
    if (selected.size === 0) return;
    try {
      await downloadDocxReport(Array.from(selected), "ChemApp Multi-Spectrum Report");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.inference.docxReportFailed);
    }
  };

  const handleBatchExport = async () => {
    if (selected.size === 0) return;
    try {
      await downloadBatchCsvZip(Array.from(selected));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.inference.batchExportFailed);
    }
  };

  const handleBatchAnalyze = async () => {
    if (selected.size === 0) return;
    setError("");
    setBatching(true);
    setBatchSummary("");
    try {
      const ids = Array.from(selected);
      const expectedRevisions = Object.fromEntries(
        spectra
          .filter((item) => selected.has(item.id))
          .map((item) => [item.id, item.result_revision ?? 0]),
      );
      const data = await analyzeBatch(ids, expectedRevisions);
      setBatchSummary(t.inference.batchSummary
        .replace("{success}", String(data.results.length))
        .replace("{failed}", String(data.errors.length)));
      if (data.results.length > 0) onDataChanged?.();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.inference.batchAnalysisFailed);
    } finally {
      setBatching(false);
    }
  };

  const handleHplcCompare = async () => {
    setError("");
    setHplcComparison(null);
    try {
      const data = await compareHplcBatch(Array.from(selected));
      setHplcComparison(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.inference.hplcCompareFailed);
    }
  };

  const handleHplcCsv = async () => {
    try {
      await downloadHplcComparisonCsv(Array.from(selected));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.inference.hplcCsvFailed);
    }
  };

  const handleQualitySummary = async () => {
    setError("");
    try {
      const data = await getBatchQuality(Array.from(selected));
      setQualitySummary(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.inference.qualityFailed);
    }
  };

  const handleWorkbench = async () => {
    setError("");
    try {
      const data = await getBatchWorkbench(Array.from(selected));
      setWorkbench(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.inference.workbenchFailed);
    }
  };

  const inference = result?.inference;
  const selectedItems = spectra.filter((s) => selected.has(s.id));
  const canCompareHplc = selectedItems.length >= 2 && selectedItems.every((s) => s.technique === "HPLC");

  return (
    <div className="inference-panel">
      <h3>{t.inference.title}</h3>
      <p className="hint">{t.inference.hint}</p>

      {spectra.length === 0 ? (
        <p className="empty-hint">{t.inference.empty}</p>
      ) : (
        <div className="checkbox-list">
          {spectra.map((s) => (
            <label key={s.id} className="checkbox-item">
              <input
                type="checkbox"
                checked={selected.has(s.id)}
                onChange={() => toggle(s.id)}
              />
              <span className="tech-badge" style={{ background: { NMR: "#3b82f6", "UV-Vis": "#10b981", Fluorescence: "#f59e0b", XRD: "#8b5cf6", HPLC: "#06b6d4", ElectroChem: "#ec4899" }[s.technique] || "#6b7280" }}>
                {s.technique}
              </span>
              <span>{s.name || s.id}</span>
            </label>
          ))}
        </div>
      )}

      <div className="action-bar">
        <button onClick={() => void handleRun()} disabled={running || selected.size < 1}>
          {running ? t.inference.running : t.inference.run}
        </button>
        <button onClick={() => void handleBatchAnalyze()} disabled={batching || selected.size < 1} className="secondary-btn">
          {batching ? t.inference.batchRunning : t.inference.batchAnalyze}
        </button>
        <button onClick={() => void handleReport()} disabled={running || selected.size < 1} className="secondary-btn">
          {t.inference.exportMarkdown}
        </button>
        <button onClick={() => void handleHtmlReport()} disabled={running || selected.size < 1} className="secondary-btn">
          {t.inference.exportHtml}
        </button>
        <button onClick={() => void handleDocxReport()} disabled={running || selected.size < 1} className="secondary-btn">
          {t.inference.exportWord}
        </button>
        <button onClick={() => void handleBatchExport()} disabled={running || selected.size < 1} className="secondary-btn">
          {t.inference.exportBundle}
        </button>
        <button onClick={() => void handleWorkbench()} disabled={running || selected.size < 1} className="secondary-btn">
          {t.inference.batchWorkbench}
        </button>
        <button onClick={() => void handleQualitySummary()} disabled={running || selected.size < 1} className="secondary-btn">
          {t.inference.qualityOverview}
        </button>
        {canCompareHplc && (
          <button onClick={() => void handleHplcCompare()} disabled={running} className="secondary-btn">
            {t.inference.hplcMatch}
          </button>
        )}
        {canCompareHplc && (
          <button onClick={() => void handleHplcCsv()} disabled={running} className="secondary-btn">
            {t.inference.hplcCsv}
          </button>
        )}
        <span className="selection-count">{t.inference.select.replace("{n}", String(selected.size))}</span>
      </div>

      {error && <p className="error" role="alert">{error}</p>}
      {batchSummary && <p className="hint" role="status" aria-live="polite">{batchSummary}</p>}

      {workbench && (
        <div className="table-section batch-workbench">
          <h4>{t.inference.batchWorkbench}</h4>
          <div className="metrics-grid">
            <div className="metric"><span className="metric-label">{t.inference.samples}</span><span className="metric-value">{workbench.summary.count}</span></div>
            <div className="metric"><span className="metric-label">{t.inference.manualConfirmed}</span><span className="metric-value">{workbench.summary.manual_confirmed}</span></div>
            <div className="metric"><span className="metric-label">{t.inference.totalPoints}</span><span className="metric-value">{workbench.summary.total_points}</span></div>
          </div>
          <p className="hint">
            {t.inference.techniqueDistribution}: {Object.entries(workbench.summary.technique_counts).map(([k, v]) => `${k} ${v}`).join(" · ") || "—"}
          </p>
          <table>
            <thead>
              <tr>
                <th>{t.inference.sample}</th>
                <th>{t.inference.technique}</th>
                <th>{t.inference.peakCount}</th>
                <th>{t.inference.quality}</th>
                <th>{t.inference.confirmation}</th>
                <th>{t.inference.summary}</th>
              </tr>
            </thead>
            <tbody>
              {workbench.items.map((item) => {
                const quality = item.quality as { status?: string; score?: number };
                return (
                  <tr key={item.id}>
                    <td>{item.name || item.id}</td>
                    <td>{item.technique}</td>
                    <td>{item.n_peaks}</td>
                    <td>{quality.status || "—"} {typeof quality.score === "number" ? `${(quality.score * 100).toFixed(0)}%` : ""}</td>
                    <td>{item.manual_confirmed ? t.inference.manual : item.ai_modified ? "AI" : t.inference.automatic}</td>
                    <td style={{ fontSize: 12 }}>{item.summary}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {qualitySummary && (
        <div className="table-section">
          <h4>{t.inference.batchQuality}</h4>
          <p className="hint">
            {t.quality.good} {qualitySummary.counts.good || 0} · {t.quality.review} {qualitySummary.counts.review || 0} · {t.quality.poor} {qualitySummary.counts.poor || 0}
          </p>
          <table>
            <thead>
              <tr>
                <th>{t.inference.sample}</th>
                <th>{t.inference.technique}</th>
                <th>{t.inference.quality}</th>
                <th>{t.inference.peakCount}</th>
                <th>{t.inference.manualConfirmed}</th>
                <th>{t.inference.notes}</th>
              </tr>
            </thead>
            <tbody>
              {qualitySummary.items.map((item) => (
                <tr key={item.id}>
                  <td>{item.name || item.id}</td>
                  <td>{item.technique}</td>
                  <td>{item.status} · {(item.score * 100).toFixed(0)}%</td>
                  <td>{item.n_peaks}</td>
                  <td>{item.manual_confirmed ? t.inference.yes : t.inference.no}</td>
                  <td style={{ fontSize: 12 }}>{[...item.warnings, ...item.info].slice(0, 2).join("; ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {hplcComparison && (
        <div className="table-section">
          <h4>{t.inference.hplcDrift}</h4>
          <table>
            <thead>
              <tr>
                <th>{t.inference.sample}</th>
                <th>{t.inference.channel}</th>
                <th>{t.inference.matchedPeaks}</th>
                <th>{t.inference.meanRtShift}</th>
                <th>{t.inference.maxRtShift}</th>
                <th>{t.inference.totalArea}</th>
              </tr>
            </thead>
            <tbody>
              {hplcComparison.drift.map((row) => (
                <tr key={row.id}>
                  <td>{row.name || row.id}</td>
                  <td>{row.channel}</td>
                  <td>{row.matched_peaks}</td>
                  <td>{row.mean_rt_shift != null ? row.mean_rt_shift.toFixed(4) : "—"}</td>
                  <td>{row.max_abs_rt_shift != null ? row.max_abs_rt_shift.toFixed(4) : "—"}</td>
                  <td>{row.total_area.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h4>{t.inference.hplcAreaTrend}</h4>
          <table>
            <thead>
              <tr>
                <th>{t.inference.refTR}</th>
                <th>{t.inference.refType}</th>
                {hplcComparison.drift.map((row) => <th key={row.id}>{row.name || row.id}</th>)}
              </tr>
            </thead>
            <tbody>
              {hplcComparison.rows.slice(0, 20).map((row) => (
                <tr key={row.peak_index}>
                  <td>{row.reference_rt.toFixed(3)}</td>
                  <td>{row.reference_type || "—"}</td>
                  {row.samples.map((sample) => (
                    <td key={sample.id}>{sample.matched ? `${Number(sample.area).toFixed(1)} (${Number(sample.rt_shift).toFixed(3)})` : "—"}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {inference && (
        <div className="inference-result">
          <div className="score-grid">
            <div className="score-card">
              <span className="score-label">{t.inference.consistency}</span>
              <span className="score-value" style={{ color: inference.consistency_score >= 0.7 ? "#6ee7b7" : "#fbbf24" }}>
                {(inference.consistency_score * 100).toFixed(0)}%
              </span>
            </div>
            <div className="score-card">
              <span className="score-label">{t.inference.confidence}</span>
              <span className="score-value" style={{ color: inference.confidence >= 0.7 ? "#6ee7b7" : "#fbbf24" }}>
                {(inference.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {inference.cross_validations.length > 0 && (
            <div className="table-section">
              <h4>{t.inference.crossValidations}</h4>
              <table>
                <thead>
                  <tr>
                    <th>{t.inference.pair}</th>
                    <th>{t.inference.metric}</th>
                    <th>{t.inference.score}</th>
                    <th>{t.inference.detail}</th>
                  </tr>
                </thead>
                <tbody>
                  {inference.cross_validations.map((cv, i) => (
                    <tr key={i}>
                      <td>{cv.pair[0]} ↔ {cv.pair[1]}</td>
                      <td>{cv.metric}</td>
                      <td style={{ color: cv.score >= 0.7 ? "#6ee7b7" : cv.score >= 0.4 ? "#fbbf24" : "#f87171" }}>
                        {(cv.score * 100).toFixed(0)}%
                      </td>
                      <td style={{ fontSize: 12 }}>{cv.detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {inference.technique_results._evidence_table?.items?.length ? (
            <div className="table-section">
              <h4>{t.inference.evidenceTable}</h4>
              <table>
                <thead>
                  <tr>
                    <th>{t.inference.technique}</th>
                    <th>{t.inference.evidence}</th>
                    <th>{t.inference.support}</th>
                    <th>{t.inference.confidence}</th>
                  </tr>
                </thead>
                <tbody>
                  {inference.technique_results._evidence_table.items.map((item, i) => (
                    <tr key={i}>
                      <td>{item.technique}</td>
                      <td>{item.evidence}</td>
                      <td style={{ fontSize: 12 }}>{item.support}</td>
                      <td style={{ color: item.confidence >= 0.7 ? "#6ee7b7" : item.confidence >= 0.4 ? "#fbbf24" : "#f87171" }}>
                        {(item.confidence * 100).toFixed(0)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}

          {inference.anomalies.length > 0 && (
            <div className="anomalies">
              <h4>{t.inference.anomalies}</h4>
              {inference.anomalies.map((a, i) => (
                <p key={i} className="anomaly-item">{a}</p>
              ))}
            </div>
          )}

          <p className="assessment">{inference.overall_assessment}</p>

          {result?.report_markdown && (
            <details className="report-preview">
              <summary>{t.inference.report}</summary>
              <pre>{result.report_markdown}</pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
