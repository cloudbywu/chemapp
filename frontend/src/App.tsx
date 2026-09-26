import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import { useLang } from "./i18n/LangContext";
import FileUpload from "./components/FileUpload";

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
import { text } from "./utils/number";

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
  const selectionRequestRef = useRef<{ sequence: number; controller: AbortController | null }>({
    sequence: 0,
    controller: null,
  });

  const refreshList = useCallback(async () => {
    try {
      const data = await listSpectra();
      setItems(data);
    } catch {
      // backend may not be running
    }
  }, []);

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
    const controller = new AbortController();
    listSpectra(controller.signal)
      .then(setItems)
      .catch((cause: unknown) => {
        if (!(cause instanceof DOMException && cause.name === "AbortError")) {
          setError(cause instanceof Error ? cause.message : t.error.network);
        }
      });
    return () => controller.abort();
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
    const controller = new AbortController();
    const sequence = selectionRequestRef.current.sequence + 1;
    selectionRequestRef.current = { sequence, controller };
    setSelectedId(id);
    setSpectrum(null);
    setResult(null);
    setLoading(true);
    setError("");
    setAnalysisOptions({});
    setPlotPickTarget(null);
    setPlotPickedRange(null);
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
  }, [clearDirtySources, t.action.loadFailed]);

  const handleAiSpectrumMutated = useCallback(async (id: string) => {
    await refreshList();
    if (id === selectedId) await performSelect(id);
  }, [performSelect, refreshList, selectedId]);

  const handleSelect = (id: string) => {
    if (id === selectedId && (loading || spectrum !== null)) return;
    if (hasUnsavedChanges) {
      setPendingNavigation({ kind: "spectrum", id });
      return;
    }
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
      if (selectedId === deleteTarget.id) {
        selectionRequestRef.current.controller?.abort();
        setSelectedId(null);
        setSpectrum(null);
        setResult(null);
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
    setItems((prev) => [...prev.filter((existing) => existing.id !== item.id), item]);
    handleSelect(item.id);
  };

  const executeAnalyze = async (forceOverwrite = false) => {
    if (!selectedId || loading || !spectrum || spectrum.id !== selectedId) return;
    setConfirmAnalysisOverwrite(false);
    setAnalyzing(true);
    setError("");
    try {
      const res = await analyzeSpectrum(
        selectedId,
        analysisOptions,
        undefined,
        result?.result_revision ?? 0,
        forceOverwrite,
      );
      setResult(res);
      setEditorDraftEpoch((value) => value + 1);
      updateDirtySource("manual-review", false);
      updateDirtySource("hplc-events", false);
      void refreshList();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.action.analysisFailed);
    } finally {
      setAnalyzing(false);
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
      await downloadSpectrumCsv(selectedId, `${spectrum?.metadata?.name || selectedId}.csv`);
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
    if (pending.kind === "spectrum") void performSelect(pending.id);
    else setTab(pending.tab);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>{t.app.title}</h1>
        <span className="subtitle">{t.app.subtitle}</span>
        <nav className="tabs">
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
          <FileUpload onUploaded={handleUploaded} />
          <SpectrumList
            spectra={items}
            selectedId={selectedId}
            onSelect={handleSelect}
            onDelete={handleDelete}
          />
        </aside>

        <section className="content">
          {tab === "spectra" && (
            <>
              {loading && <p className="loading">{t.action.loading}</p>}
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
                          <button type="button" onClick={handleDownloadCsv} disabled={!selectedId || loading || !spectrum} className="secondary-btn">
                            {t.action.exportCsv}
                          </button>
                          <button onClick={handleDownloadReport} disabled={!result || !selectedId || loading || !spectrum} className="secondary-btn">
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
                <div className="empty-state">
                  <p>{t.action.empty}</p>
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
        onConfirm={confirmDelete}
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
