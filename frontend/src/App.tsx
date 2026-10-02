import { lazy, Suspense, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import "./App.css";
import { useLang } from "./i18n/LangContext";
import FileUpload, { type FileUploadHandle } from "./components/FileUpload";

import SpectrumList from "./components/SpectrumList";
import SpectrumViewer, { type IntegrationSelection } from "./components/SpectrumViewer";
import AnalysisPanel from "./components/AnalysisPanel";
const CompareView = lazy(() => import("./components/CompareView"));
import ErrorBoundary from "./components/ErrorBoundary";
const InferencePanel = lazy(() => import("./components/InferencePanel"));
const AIChatSidebar = lazy(() => import("./components/AIChatSidebar"));
const SettingsPanel = lazy(() => import("./components/SettingsPanel"));
import MLPredictionPanel from "./components/MLPredictionPanel";
const MLTrainingPanel = lazy(() => import("./components/MLTrainingPanel"));
import AnalysisControls from "./components/AnalysisControls";
import QualityPanel from "./components/QualityPanel";
import ManualReviewPanel from "./components/ManualReviewPanel";
import NMRWorkbench from "./components/NMRWorkbench";
const SpectrumReviewQueuePanel = lazy(() =>
  import("./components/SpectrumGoldReviewPanel").then((module) => ({
    default: module.SpectrumReviewQueuePanel,
  })),
);
const SpectrumGoldReviewPanel = lazy(() =>
  import("./components/SpectrumGoldReviewPanel").then((module) => ({
    default: module.default,
  })),
);
import ConfirmDialog from "./components/ConfirmDialog";
import {
  listSpectra,
  getSpectrum,
  deleteSpectrum,
  analyzeSpectrum,
  downloadMarkdownReport,
  getResult,
  ApiError,
  downloadSpectrumCsv,
} from "./services/api";
import type { AnalysisOptions, AnalysisResult, SpectrumData, SpectrumListItem } from "./types/spectrum";
import { displayText, text } from "./utils/number";

type Tab = "spectra" | "review" | "compare" | "inference" | "settings";
type PlotPickTarget = { mode: "nmr" | "hplc"; index: number; label: string } | null;
type PlotPickedRange = { mode: "nmr" | "hplc"; index: number; start: number; end: number; nonce: number } | null;
type PendingNavigation = { kind: "spectrum"; id: string } | { kind: "tab"; tab: Tab } | null;
type DirtySource = "nmr-workbench" | "manual-review" | "hplc-events" | "gold-review";

export default function App() {
  const { t, lang, toggleLang } = useLang();
  const [items, setItems] = useState<SpectrumListItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [spectrum, setSpectrum] = useState<SpectrumData | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<Tab>("spectra");
  const [analysisOptions, setAnalysisOptions] = useState<AnalysisOptions>({});
  const [error, setError] = useState("");
  const [plotPickTarget, setPlotPickTarget] = useState<PlotPickTarget>(null);
  const [plotPickedRange, setPlotPickedRange] = useState<PlotPickedRange>(null);
  const [deleteTarget, setDeleteTarget] = useState<SpectrumListItem | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [dirtySources, setDirtySources] = useState<ReadonlySet<DirtySource>>(
    () => new Set(),
  );
  const [pendingNavigation, setPendingNavigation] = useState<PendingNavigation>(null);
  const [confirmAnalysisOverwrite, setConfirmAnalysisOverwrite] = useState(false);
  const [confirmDiscardForAnalysis, setConfirmDiscardForAnalysis] = useState(false);
  const [editorDraftEpoch, setEditorDraftEpoch] = useState(0);
  const selectionRequestRef = useRef<{ id: string | null; sequence: number; controller: AbortController | null }>({
    id: null,
    sequence: 0,
    controller: null,
  });
  const analysisRequestRef = useRef<AbortController | null>(null);
  const listRequestRef = useRef<AbortController | null>(null);
  const dirtySourcesRef = useRef(dirtySources);
  const fileUploadRef = useRef<FileUploadHandle>(null);

  useLayoutEffect(() => {
    dirtySourcesRef.current = dirtySources;
  }, [dirtySources]);

  const invalidateAnalysis = useCallback(() => {
    analysisRequestRef.current?.abort();
    analysisRequestRef.current = null;
    setAnalyzing(false);
  }, []);

  useEffect(() => () => {
    selectionRequestRef.current.controller?.abort();
    selectionRequestRef.current = {
      id: null,
      sequence: selectionRequestRef.current.sequence + 1,
      controller: null,
    };
    analysisRequestRef.current?.abort();
    analysisRequestRef.current = null;
    listRequestRef.current?.abort();
    listRequestRef.current = null;
  }, []);

  const refreshList = useCallback(async (reportError = false) => {
    listRequestRef.current?.abort();
    const controller = new AbortController();
    listRequestRef.current = controller;
    try {
      const data = await listSpectra(controller.signal);
      if (listRequestRef.current === controller && !controller.signal.aborted) setItems(data);
    } catch (cause: unknown) {
      if (reportError && listRequestRef.current === controller && !controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : t.error.network);
      }
    } finally {
      if (listRequestRef.current === controller) listRequestRef.current = null;
    }
  }, [t.error.network]);

  const updateDirtySource = useCallback((source: DirtySource, dirty: boolean) => {
    setDirtySources((previous) => {
      if (previous.has(source) === dirty) return previous;
      const next = new Set(previous);
      if (dirty) next.add(source);
      else next.delete(source);
      return next;
    });
  }, []);

  const setNmrWorkbenchDirty = useCallback(
    (dirty: boolean) => updateDirtySource("nmr-workbench", dirty),
    [updateDirtySource],
  );
  const setManualReviewDirty = useCallback(
    (dirty: boolean) => updateDirtySource("manual-review", dirty),
    [updateDirtySource],
  );
  const setHplcEventsDirty = useCallback(
    (dirty: boolean) => updateDirtySource("hplc-events", dirty),
    [updateDirtySource],
  );
  const setGoldReviewDirty = useCallback(
    (dirty: boolean) => updateDirtySource("gold-review", dirty),
    [updateDirtySource],
  );
  const clearDirtySources = useCallback(() => setDirtySources(new Set()), []);

  useEffect(() => {
    listRequestRef.current?.abort();
    const controller = new AbortController();
    listRequestRef.current = controller;
    listSpectra(controller.signal).then((data) => {
      if (listRequestRef.current === controller && !controller.signal.aborted) setItems(data);
    }).catch((cause: unknown) => {
      if (listRequestRef.current === controller && !controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : t.error.network);
      }
    }).finally(() => {
      if (listRequestRef.current === controller) listRequestRef.current = null;
    });
    return () => {
      controller.abort();
      if (listRequestRef.current === controller) listRequestRef.current = null;
    };
  }, [t.error.network]);

  const hasUnsavedChanges = dirtySources.size > 0;

  useEffect(() => {
    if (!hasUnsavedChanges) return;
    const protectUnsavedReview = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", protectUnsavedReview);
    return () => window.removeEventListener("beforeunload", protectUnsavedReview);
  }, [hasUnsavedChanges]);

  const performSelect = useCallback(async (id: string) => {
    selectionRequestRef.current.controller?.abort();
    invalidateAnalysis();
    const controller = new AbortController();
    const sequence = selectionRequestRef.current.sequence + 1;
    selectionRequestRef.current = { id, sequence, controller };
    setSelectedId(id);
    setSpectrum(null);
    setResult(null);
    setLoading(true);
    setError("");
    setAnalysisOptions({});
    setPlotPickTarget(null);
    setPlotPickedRange(null);
    setConfirmAnalysisOverwrite(false);
    setConfirmDiscardForAnalysis(false);
    try {
      const [spec, existingResult] = await Promise.all([
        getSpectrum(id, controller.signal),
        getResult(id, controller.signal).catch((cause: unknown) => {
          if (cause instanceof ApiError && cause.status === 404) return null;
          throw cause;
        }),
      ]);
      if (selectionRequestRef.current.sequence !== sequence) return;
      setSpectrum(spec);
      setResult(existingResult);
      clearDirtySources();
    } catch (cause: unknown) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      if (selectionRequestRef.current.sequence !== sequence) return;
      setSpectrum(null);
      setResult(null);
      setError(cause instanceof Error ? cause.message : t.action.loadFailed);
    } finally {
      if (selectionRequestRef.current.sequence === sequence) setLoading(false);
    }
  }, [clearDirtySources, invalidateAnalysis, t.action.loadFailed]);

  const handleAiSpectrumMutated = useCallback(async (id: string) => {
    const sequence = selectionRequestRef.current.sequence;
    await refreshList();
    if (selectionRequestRef.current.id !== id
      || selectionRequestRef.current.sequence !== sequence) return;
    if (dirtySourcesRef.current.size > 0) {
      setError(t.manual.revisionConflict);
      return;
    }
    await performSelect(id);
  }, [performSelect, refreshList, t.manual.revisionConflict]);

  const handleSelect = (id: string) => {
    if (id === selectedId && (loading || spectrum !== null)) {
      requestTab("spectra");
      return;
    }
    if (hasUnsavedChanges) {
      setPendingNavigation({ kind: "spectrum", id });
      return;
    }
    setTab("spectra");
    void performSelect(id);
  };

  const handleDelete = (id: string) => {
    const target = items.find((item) => item.id === id);
    if (target) setDeleteTarget(target);
  };

  const cancelDelete = useCallback(() => {
    if (!deleting) setDeleteTarget(null);
  }, [deleting]);

  const confirmDelete = async () => {
    if (!deleteTarget || deleting) return;
    setDeleting(true);
    setError("");
    try {
      if (
        typeof deleteTarget.spectrum_revision !== "number"
        || typeof deleteTarget.result_revision !== "number"
      ) {
        setDeleteTarget(null);
        await refreshList();
        setError(t.action.revisionUnavailable);
        return;
      }
      await deleteSpectrum(
        deleteTarget.id,
        deleteTarget.spectrum_revision,
        deleteTarget.result_revision,
      );
      if (selectionRequestRef.current.id === deleteTarget.id) {
        selectionRequestRef.current.controller?.abort();
        selectionRequestRef.current = {
          id: null,
          sequence: selectionRequestRef.current.sequence + 1,
          controller: null,
        };
        invalidateAnalysis();
        setSelectedId(null);
        setSpectrum(null);
        setResult(null);
        setLoading(false);
        clearDirtySources();
      }
      setDeleteTarget(null);
      await refreshList();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.error.network);
    } finally {
      setDeleting(false);
    }
  };

  const handleUploaded = (item: SpectrumListItem) => {
    // A list fetched before this upload committed must not erase the new item.
    const interruptedList = listRequestRef.current !== null;
    listRequestRef.current?.abort();
    listRequestRef.current = null;
    setItems((prev) => [...prev.filter((existing) => existing.id !== item.id), item]);
    // If this interrupted initial loading, fetch again so older stored spectra
    // are not omitted from the sidebar while keeping the uploaded item visible.
    if (interruptedList) void refreshList();
    handleSelect(item.id);
  };

  const executeAnalyze = async (forceOverwrite = false) => {
    if (!selectedId || loading || !spectrum || spectrum.id !== selectedId || analysisRequestRef.current) return;
    const controller = new AbortController();
    const selectionSequence = selectionRequestRef.current.sequence;
    analysisRequestRef.current = controller;
    // Aborting only cancels the client wait; the server may already have saved
    // the result. Keep an ownership guard even if a late response still arrives,
    // including when the user leaves and returns to the same spectrum.
    const isCurrentRequest = () => (
      analysisRequestRef.current === controller
      && selectionRequestRef.current.sequence === selectionSequence
      && !controller.signal.aborted
    );
    setConfirmAnalysisOverwrite(false);
    setAnalyzing(true);
    setError("");
    try {
      const res = await analyzeSpectrum(
        selectedId,
        analysisOptions,
        controller.signal,
        result?.result_revision ?? 0,
        forceOverwrite,
      );
      if (!isCurrentRequest()) return;
      setResult(res);
      setEditorDraftEpoch((value) => value + 1);
      updateDirtySource("manual-review", false);
      updateDirtySource("hplc-events", false);
      void refreshList();
    } catch (e: unknown) {
      if (!isCurrentRequest() || (e instanceof DOMException && e.name === "AbortError")) return;
      setError(e instanceof Error ? e.message : t.action.analysisFailed);
    } finally {
      if (analysisRequestRef.current === controller) {
        analysisRequestRef.current = null;
        setAnalyzing(false);
      }
    }
  };

  const handleAnalyze = () => {
    if (
      dirtySources.has("manual-review")
      || dirtySources.has("hplc-events")
    ) {
      setConfirmDiscardForAnalysis(true);
      return;
    }
    if (result?.metrics.manual_confirmed) setConfirmAnalysisOverwrite(true);
    else void executeAnalyze(false);
  };

  const discardEditorsAndAnalyze = () => {
    setConfirmDiscardForAnalysis(false);
    setEditorDraftEpoch((value) => value + 1);
    updateDirtySource("manual-review", false);
    updateDirtySource("hplc-events", false);
    if (result?.metrics.manual_confirmed) {
      setConfirmAnalysisOverwrite(true);
      return;
    }
    void executeAnalyze(false);
  };

  const handleManualResultChanged = (nextResult: AnalysisResult) => {
    setResult(nextResult);
    updateDirtySource("manual-review", false);
    if (!dirtySources.has("hplc-events")) {
      setEditorDraftEpoch((value) => value + 1);
    }
    void refreshList();
  };

  const handleHplcResultChanged = (nextResult: AnalysisResult) => {
    setResult(nextResult);
    updateDirtySource("hplc-events", false);
    if (!dirtySources.has("manual-review")) {
      setEditorDraftEpoch((value) => value + 1);
    }
    void refreshList();
  };

  const handleDownloadReport = async () => {
    if (!selectedId || loading || !spectrum || spectrum.id !== selectedId) return;
    try {
      await downloadMarkdownReport([selectedId], text(spectrum?.metadata?.name) ?? "ChemApp Analysis Report");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.action.reportFailed);
    }
  };

  const handleDownloadCsv = async () => {
    if (!selectedId || loading || !spectrum || spectrum.id !== selectedId) return;
    try {
      await downloadSpectrumCsv(selectedId, `${displayText(spectrum?.metadata?.name) || selectedId}.csv`);
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : t.error.network);
    }
  };

  const handlePlotRangeSelected = useCallback((start: number, end: number) => {
    if (!plotPickTarget) return;
    setPlotPickedRange({
      mode: plotPickTarget.mode,
      index: plotPickTarget.index,
      start,
      end,
      nonce: Date.now(),
    });
    setPlotPickTarget(null);
  }, [plotPickTarget]);

  const integrationSelection: IntegrationSelection | null = plotPickTarget ? {
    label: plotPickTarget.label,
    regions: [],
  } : null;

  const requestTab = (nextTab: Tab) => {
    if (nextTab === tab) return;
    if (hasUnsavedChanges) {
      setPendingNavigation({ kind: "tab", tab: nextTab });
      return;
    }
    setTab(nextTab);
  };

  const discardAndNavigate = () => {
    const pending = pendingNavigation;
    setPendingNavigation(null);
    clearDirtySources();
    if (!pending) return;
    if (pending.kind === "spectrum") {
      setTab("spectra");
      void performSelect(pending.id);
    } else setTab(pending.tab);
  };

  return (
    <div className="app">
      <a className="skip-link" href="#workspace-content">{t.workspace.skip}</a>
      <header className="app-header">
        <div className="app-brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 28 28" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M4 19h5l3-12 4 16 3-9 2 5h3" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          <div><h1>{t.app.title}</h1><span className="subtitle">{t.app.subtitle}</span></div>
        </div>
        <nav className="tabs" aria-label={t.workspace.navigation} onKeyDown={(event) => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          const buttons = Array.from(event.currentTarget.querySelectorAll("button"));
          const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
          if (index < 0) return;
          event.preventDefault();
          const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1
            : (index + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
          buttons[next]?.focus();
        }}>
          <button
            className={tab === "spectra" ? "active" : ""}
            onClick={() => requestTab("spectra")}
            aria-current={tab === "spectra" ? "page" : undefined}
          >
            {t.nav.spectra}
          </button>
          <button
            className={tab === "review" ? "active" : ""}
            onClick={() => requestTab("review")}
            aria-current={tab === "review" ? "page" : undefined}
          >
            {t.nav.review}
          </button>
          <button
            className={tab === "compare" ? "active" : ""}
            onClick={() => requestTab("compare")}
            aria-current={tab === "compare" ? "page" : undefined}
          >
            {t.nav.compare}
          </button>
          <button
            className={tab === "inference" ? "active" : ""}
            onClick={() => requestTab("inference")}
            aria-current={tab === "inference" ? "page" : undefined}
          >
            {t.nav.inference}
          </button>
          <button
            className={tab === "settings" ? "active" : ""}
            onClick={() => requestTab("settings")}
            aria-current={tab === "settings" ? "page" : undefined}
          >
            {t.nav.settings}
          </button>
        </nav>
        <button type="button" className="lang-toggle" onClick={toggleLang} aria-label={lang === "zh" ? "Switch to English" : "切换到中文"} title={lang === "zh" ? "Switch to English" : "切换到中文"}>
          {lang === "zh" ? "EN" : "中"}
        </button>
      </header>

      <main className="app-main">
        <aside className="sidebar">
          <FileUpload ref={fileUploadRef} onUploaded={handleUploaded} />
          <section className="spectrum-library" aria-labelledby="library-heading">
            <div className="section-heading"><h2 id="library-heading">{t.workspace.library}</h2><span className="section-count">{items.length}</span></div>
            <SpectrumList
              spectra={items}
              selectedId={selectedId}
              onSelect={handleSelect}
              onDelete={handleDelete}
            />
          </section>
        </aside>

        <section className="content" id="workspace-content" tabIndex={-1} aria-label={t.nav[tab]}>
          <div className="workspace-heading">
            <div><span className="workspace-eyebrow">{t.workspace.title}</span><h2>{t.nav[tab]}</h2></div>
            {tab === "spectra" && spectrum && <span className="workspace-spectrum" title={displayText(spectrum.metadata?.name) || spectrum.id}>{displayText(spectrum.metadata?.name) || spectrum.id}</span>}
          </div>
          {tab === "spectra" && (
            <>
              {loading && <p className="loading" role="status">{t.action.loading}</p>}
              {spectrum && (
                <ErrorBoundary resetKey={`${selectedId || ""}:${tab}`} resetLabel={t.error.retry} title={t.error.somethingWrong}>
                  <>
                    {spectrum.technique === "NMR" ? (
                      <NMRWorkbench
                        key={spectrum.id}
                        spectrum={spectrum}
                        result={result}
                        spectra={items}
                        onSpectrumChanged={(nextSpectrum) => {
                          setSpectrum(nextSpectrum);
                          void refreshList();
                        }}
                        onResultChanged={(nextResult) => {
                          setResult(nextResult);
                          void refreshList();
                        }}
                        onError={setError}
                        onDirtyChange={setNmrWorkbenchDirty}
                      />
                    ) : (
                      <>
                        <SpectrumViewer
                          spectrum={spectrum}
                          result={result}
                          integrationSelection={integrationSelection}
                          onIntegrationRangeSelected={handlePlotRangeSelected}
                        />
                        <AnalysisControls
                          technique={spectrum.technique}
                          value={analysisOptions}
                          onChange={setAnalysisOptions}
                          disabled={analyzing}
                        />
                        <div className="action-bar">
                          <button type="button" onClick={handleAnalyze} disabled={analyzing || loading || !spectrum}>
                            {analyzing ? t.action.analyzing : t.action.analyze}
                          </button>
                          <button type="button" onClick={() => void handleDownloadCsv()} disabled={!selectedId || loading || !spectrum} className="secondary-btn">
                            {t.action.exportCsv}
                          </button>
                          <button onClick={() => void handleDownloadReport()} disabled={!result || !selectedId || loading || !spectrum} className="secondary-btn">
                            {t.action.report}
                          </button>
                        </div>
                        {error && <p className="error">{error}</p>}
                        {result && (
                          <>
                            <QualityPanel quality={result.metrics.quality} />
                            <ManualReviewPanel
                              key={`manual:${spectrum.id}:${editorDraftEpoch}`}
                              spectrum={spectrum}
                              result={result}
                              onSaved={handleManualResultChanged}
                              activePlotPick={plotPickTarget}
                              pickedPlotRange={plotPickedRange}
                              onRequestPlotPick={setPlotPickTarget}
                              onDirtyChange={setManualReviewDirty}
                            />
                            <AnalysisPanel
                              key={`analysis:${spectrum.id}:${editorDraftEpoch}`}
                              result={result}
                              spectrumId={selectedId}
                              onResultChanged={handleHplcResultChanged}
                              onDirtyChange={setHplcEventsDirty}
                            />
                            <MLPredictionPanel spectrumId={selectedId} hasResult={true} technique={result.technique} result={result}
                              nucleus={text(spectrum?.parameters?.nucleus) ?? ""} spectra={items} />
                          </>
                        )}
                      </>
                    )}
                    {spectrum.technique === "NMR" && (
                      <SpectrumGoldReviewPanel
                        key={`gold-review-${spectrum.id}`}
                        spectrum={spectrum}
                        onDirtyChange={setGoldReviewDirty}
                      />
                    )}
                    {error && spectrum.technique === "NMR" && <p className="error">{error}</p>}
                  </>
                </ErrorBoundary>
              )}
              {error && !spectrum && <p className="error" role="alert">{error}</p>}
              {!spectrum && !loading && (
                <div className="empty-state workspace-welcome">
                  <div className="welcome-spectrum" aria-hidden="true">
                    <svg viewBox="0 0 400 96" fill="none">
                      <path className="welcome-grid" d="M0 24h400M0 48h400M0 72h400M50 0v96M100 0v96M150 0v96M200 0v96M250 0v96M300 0v96M350 0v96" />
                      <path className="welcome-signal" d="M0 76h56l6-4 5 4h22l7-9 6 9h33l7-42 7 42h22l8-61 8 61h34l6-18 6 18h35l8-39 8 39h60l5-7 5 7h37" />
                    </svg>
                  </div>
                  <span className="welcome-kicker">CHEMAPP</span>
                  <h3>{t.workspace.welcome}</h3>
                  <p>{t.workspace.description}</p>
                  <div className="welcome-actions">
                    <button type="button" className="welcome-primary" onClick={() => fileUploadRef.current?.chooseFiles()}>{t.workspace.import}<span aria-hidden="true"> ↗</span></button>
                    <button type="button" className="welcome-secondary" onClick={() => fileUploadRef.current?.showExamples()}>{t.workspace.examples}</button>
                  </div>
                  {items.length > 0 && <button type="button" className="welcome-resume" onClick={() => handleSelect(items[0].id)}>{t.workspace.resume}<span aria-hidden="true"> →</span></button>}
                  <ol className="welcome-steps">
                    <li><span>01</span><strong>{t.workspace.stepImport}</strong><p>{t.workspace.stepImportHint}</p></li>
                    <li><span>02</span><strong>{t.workspace.stepReview}</strong><p>{t.workspace.stepReviewHint}</p></li>
                    <li><span>03</span><strong>{t.workspace.stepAnalyze}</strong><p>{t.workspace.stepAnalyzeHint}</p></li>
                  </ol>
                  <p className="welcome-footnote">{t.workspace.localHint}</p>
                </div>
              )}
            </>
          )}
          {tab === "compare" && (
            <Suspense fallback={null}>
              <CompareView spectra={items} />
            </Suspense>
          )}
          {tab === "review" && (
            <Suspense fallback={null}>
              <SpectrumReviewQueuePanel
                onOpenSpectrum={(id) => {
                  setTab("spectra");
                  if (id !== selectedId) void performSelect(id);
                }}
              />
            </Suspense>
          )}
          {tab === "inference" && (
            <Suspense fallback={null}>
              <InferencePanel spectra={items} onDataChanged={() => void refreshList()} />
            </Suspense>
          )}
          {tab === "settings" && (
            <Suspense fallback={null}>
              <SettingsPanel />
              <div style={{ marginTop: 20 }}>
                <MLTrainingPanel />
              </div>
            </Suspense>
          )}
        </section>
        <Suspense fallback={null}>
          <AIChatSidebar
            spectra={items}
            protectedSpectrumId={hasUnsavedChanges ? selectedId : null}
            onSpectrumMutated={handleAiSpectrumMutated}
          />
        </Suspense>
      </main>
      <ConfirmDialog
        open={deleteTarget !== null}
        title={t.action.deleteTitle}
        message={t.action.deleteMessage.replace("{name}", () => deleteTarget?.name || deleteTarget?.id || "")}
        confirmLabel={t.action.confirmDelete}
        cancelLabel={t.action.cancel}
        busyLabel={t.action.deleting}
        busy={deleting}
        danger
        onConfirm={() => void confirmDelete()}
        onCancel={cancelDelete}
      />
      <ConfirmDialog
        open={pendingNavigation !== null}
        title={t.action.unsavedTitle}
        message={t.action.unsavedMessage}
        confirmLabel={t.action.discard}
        cancelLabel={t.action.keepEditing}
        danger
        onConfirm={discardAndNavigate}
        onCancel={() => setPendingNavigation(null)}
      />
      <ConfirmDialog
        open={confirmDiscardForAnalysis}
        title={t.action.unsavedTitle}
        message={t.action.unsavedMessage}
        confirmLabel={t.action.discard}
        cancelLabel={t.action.keepEditing}
        danger
        onConfirm={discardEditorsAndAnalyze}
        onCancel={() => setConfirmDiscardForAnalysis(false)}
      />
      <ConfirmDialog
        open={confirmAnalysisOverwrite}
        title={t.workbench.confirmReanalysisTitle}
        message={t.workbench.confirmReanalysisMessage}
        confirmLabel={t.workbench.overwriteAndAnalyze}
        cancelLabel={t.action.cancel}
        busy={analyzing}
        busyLabel={t.action.analyzing}
        danger
        onConfirm={() => void executeAnalyze(true)}
        onCancel={() => setConfirmAnalysisOverwrite(false)}
      />
    </div>
  );
}
