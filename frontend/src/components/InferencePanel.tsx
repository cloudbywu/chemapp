import { useEffect, useRef, useState } from "react";
import { analyzeBatch, compareHplcBatch, downloadBatchCsvZip, downloadDocxReport, downloadHtmlReport, downloadHplcComparisonCsv, downloadMarkdownReport, getBatchQuality, getBatchWorkbench, runInference } from "../services/api";
import { useLang } from "../i18n/LangContext";
import type { InferenceResponse, SpectrumListItem } from "../types/spectrum";

interface Props {
  spectra: SpectrumListItem[];
  onDataChanged?: () => void;
}

type Action = "inference" | "batch" | "markdown" | "html" | "docx" | "bundle" | "hplc" | "hplcCsv" | "quality" | "workbench";
type RequestOwner = { scope: object };
type Results = {
  result: InferenceResponse | null;
  batchSummary: string;
  hplcComparison: Awaited<ReturnType<typeof compareHplcBatch>> | null;
  qualitySummary: Awaited<ReturnType<typeof getBatchQuality>> | null;
  workbench: Awaited<ReturnType<typeof getBatchWorkbench>> | null;
};
type ViewState = Results & {
  key: string;
  scope: object;
  busy: Partial<Record<Action, boolean>>;
  latestRequest: RequestOwner | null;
  error: string;
};
const emptyView = (key: string): ViewState => ({
  key, scope: {}, busy: {}, latestRequest: null, error: "", result: null,
  batchSummary: "", hplcComparison: null, qualitySummary: null, workbench: null,
});

export default function InferencePanel({ spectra, onDataChanged }: Props) {
  const { t } = useLang();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const available = new Map(spectra.map((item) => [item.id, item]));
  // Preserve selection order: the first HPLC spectrum is the comparison reference.
  const ids = Array.from(selected).filter((id) => available.has(id));
  const selectedItems = ids.map((id) => available.get(id)!);
  const selectionKey = JSON.stringify(selectedItems.map((item) => [
    item.id, item.spectrum_revision ?? 0, item.result_revision ?? 0, item.technique,
  ]));
  const [view, setView] = useState(() => emptyView(selectionKey));
  const [batching, setBatching] = useState(false);
  const requests = useRef(new Map<Action, RequestOwner>());
  const mounted = useRef(false);
  const onDataChangedRef = useRef(onDataChanged);

  // Adjust during render so neither deleted IDs nor old results can reach a commit.
  // A fresh scope identity also invalidates A -> B -> A requests, not just different keys.
  if (ids.length !== selected.size) setSelected(new Set(ids));
  if (view.key !== selectionKey) setView(emptyView(selectionKey));

  useEffect(() => {
    onDataChangedRef.current = onDataChanged;
  }, [onDataChanged]);
  useEffect(() => {
    mounted.current = true;
    const owners = requests.current;
    return () => {
      mounted.current = false;
      owners.clear();
    };
  }, []);

  const toggle = (id: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const execute = async <T,>(
    action: Action,
    operation: () => Promise<T>,
    failure: string,
    apply: (data: T) => Partial<Results> = () => ({}),
  ) => {
    if (!ids.length || batching || requests.current.has("batch")) return;
    const previous = requests.current.get(action);
    // The ref takes ownership synchronously, including repeated gestures in one render.
    // Mutations stay locked across selections until the server operation actually settles.
    if (previous?.scope === view.scope) return;
    const reset = action === "batch" ? emptyView(selectionKey) : null;
    const request = { scope: reset?.scope ?? view.scope };
    requests.current.set(action, request);
    if (action === "batch") setBatching(true);
    setView((current) => current.scope === view.scope ? {
      ...(reset ?? current), error: "", latestRequest: request,
      busy: { ...(reset?.busy ?? current.busy), [action]: true },
    } : current);
    try {
      const data = await operation();
      if (!mounted.current || requests.current.get(action) !== request) return;
      setView((current) => current.scope === request.scope ? { ...current, ...apply(data) } : current);
    } catch (e: unknown) {
      if (!mounted.current || requests.current.get(action) !== request) return;
      setView((current) => current.scope === request.scope && current.latestRequest === request
        ? { ...current, error: e instanceof Error ? e.message : failure } : current);
    } finally {
      if (mounted.current && requests.current.get(action) === request) {
        requests.current.delete(action);
        if (action === "batch") setBatching(false);
        setView((current) => current.scope === request.scope
          ? { ...current, busy: { ...current.busy, [action]: false } } : current);
      }
    }
  };

  const handleRun = () => execute("inference", () => runInference(ids), t.inference.inferenceFailed, (result) => ({ result }));
  // An initiated download still belongs to the captured IDs; only its UI feedback is invalidated.
  const handleReport = () => execute("markdown", () => downloadMarkdownReport(ids, "ChemApp Multi-Spectrum Report"), t.inference.reportFailed);
  const handleHtmlReport = () => execute("html", () => downloadHtmlReport(ids, "ChemApp Multi-Spectrum Report"), t.inference.htmlReportFailed);
  const handleDocxReport = () => execute("docx", () => downloadDocxReport(ids, "ChemApp Multi-Spectrum Report"), t.inference.docxReportFailed);
  const handleBatchExport = () => execute("bundle", () => downloadBatchCsvZip(ids), t.inference.batchExportFailed);
  const handleBatchAnalyze = () => execute("batch", async () => {
    const expectedRevisions = Object.fromEntries(selectedItems.map((item) => [item.id, item.result_revision ?? 0]));
    const data = await analyzeBatch(ids, expectedRevisions);
    // A completed server mutation must refresh shared data even after selection changes or unmount.
    if (data.results.length > 0) onDataChangedRef.current?.();
    return data;
  }, t.inference.batchAnalysisFailed, (data) => ({
    batchSummary: t.inference.batchSummary
      .replace("{success}", String(data.results.length))
      .replace("{failed}", String(data.errors.length)),
  }));
  const handleHplcCompare = () => execute("hplc", () => compareHplcBatch(ids), t.inference.hplcCompareFailed, (hplcComparison) => ({ hplcComparison }));
  const handleHplcCsv = () => execute("hplcCsv", () => downloadHplcComparisonCsv(ids), t.inference.hplcCsvFailed);
  const handleQualitySummary = () => execute("quality", () => getBatchQuality(ids), t.inference.qualityFailed, (qualitySummary) => ({ qualitySummary }));
  const handleWorkbench = () => execute("workbench", () => getBatchWorkbench(ids), t.inference.workbenchFailed, (workbench) => ({ workbench }));

  const { result, error, batchSummary, hplcComparison, qualitySummary, workbench, busy } = view;
  const running = busy.inference ?? false;
  const inference = result?.inference;
  const canCompareHplc = selectedItems.length >= 2 && selectedItems.every((s) => s.technique === "HPLC");
  const disabled = (action: Action) => batching || running || !!busy[action] || !ids.length;
  const label = (action: Action, title: string) => busy[action] ? `${title} · ${t.action.loading}` : title;

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
                aria-label={`${s.technique} ${s.name || s.id}`}
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
        <button onClick={() => void handleRun()} disabled={disabled("inference")} aria-busy={running}>
          {running ? t.inference.running : t.inference.run}
        </button>
        <button onClick={() => void handleBatchAnalyze()} disabled={batching || !ids.length} aria-busy={batching} className="secondary-btn">
          {batching ? t.inference.batchRunning : t.inference.batchAnalyze}
        </button>
        <button onClick={() => void handleReport()} disabled={disabled("markdown")} aria-busy={!!busy.markdown} className="secondary-btn">
          {label("markdown", t.inference.exportMarkdown)}
        </button>
        <button onClick={() => void handleHtmlReport()} disabled={disabled("html")} aria-busy={!!busy.html} className="secondary-btn">
          {label("html", t.inference.exportHtml)}
        </button>
        <button onClick={() => void handleDocxReport()} disabled={disabled("docx")} aria-busy={!!busy.docx} className="secondary-btn">
          {label("docx", t.inference.exportWord)}
        </button>
        <button onClick={() => void handleBatchExport()} disabled={disabled("bundle")} aria-busy={!!busy.bundle} className="secondary-btn">
          {label("bundle", t.inference.exportBundle)}
        </button>
        <button onClick={() => void handleWorkbench()} disabled={disabled("workbench")} aria-busy={!!busy.workbench} className="secondary-btn">
          {label("workbench", t.inference.batchWorkbench)}
        </button>
        <button onClick={() => void handleQualitySummary()} disabled={disabled("quality")} aria-busy={!!busy.quality} className="secondary-btn">
          {label("quality", t.inference.qualityOverview)}
        </button>
        {canCompareHplc && (
          <button onClick={() => void handleHplcCompare()} disabled={disabled("hplc")} aria-busy={!!busy.hplc} className="secondary-btn">
            {label("hplc", t.inference.hplcMatch)}
          </button>
        )}
        {canCompareHplc && (
          <button onClick={() => void handleHplcCsv()} disabled={disabled("hplcCsv")} aria-busy={!!busy.hplcCsv} className="secondary-btn">
            {label("hplcCsv", t.inference.hplcCsv)}
          </button>
        )}
        <span className="selection-count">{t.inference.select.replace("{n}", String(ids.length))}</span>
      </div>

      {(batching || Object.values(busy).some(Boolean)) && <p className="visually-hidden" role="status">{batching ? t.inference.batchRunning : t.action.loading}</p>}
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
                const quality = item.quality as { status?: string; score?: number } | null;
                return (
                  <tr key={item.id}>
                    <td>{item.name || item.id}</td>
                    <td>{item.technique}</td>
                    <td>{item.n_peaks}</td>
                    <td>{quality?.status || "—"} {typeof quality?.score === "number" ? `${(quality.score * 100).toFixed(0)}%` : ""}</td>
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
