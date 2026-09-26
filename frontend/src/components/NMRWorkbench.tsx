import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AnalysisControls from "./AnalysisControls";
import ManualReviewPanel from "./ManualReviewPanel";
import MLPredictionPanel from "./MLPredictionPanel";
import NMRPanel from "./NMRPanel";
import QualityPanel from "./QualityPanel";
import SpectrumViewer, { type IntegrationSelection } from "./SpectrumViewer";
import ConfirmDialog from "./ConfirmDialog";
import {
  analyzeSpectrum,
  ApiError,
  downloadMarkdownReport,
  getNmrSpectrumView,
  processNmrSpectrum,
  resetNmrSpectrum,
  saveManualResult,
} from "../services/api";
import type { AnalysisOptions, AnalysisResult, SpectrumData, SpectrumListItem } from "../types/spectrum";
import { useLang } from "../i18n/LangContext";
import { displayText } from "../utils/number";

type PlotPickTarget = { mode: "nmr" | "hplc"; index: number; label: string } | null;
type PlotPickedRange = { mode: "nmr" | "hplc"; index: number; start: number; end: number; nonce: number } | null;
type PendingMultipletRebuild = {
  ranges: Array<{ start: number; end: number }>;
  expectedRevision: number;
} | null;
type NmrViewSpectrum = SpectrumData & {
  view?: "original" | "current";
  quality_metrics?: Record<string, unknown>;
};

interface Props {
  spectrum: SpectrumData;
  result: AnalysisResult | null;
  spectra: SpectrumListItem[];
  onSpectrumChanged: (spectrum: SpectrumData) => void;
  onResultChanged: (result: AnalysisResult | null) => void;
  onError: (message: string) => void;
  onDirtyChange?: (dirty: boolean) => void;
}

function initialProcessing(spectrum: SpectrumData) {
  return {
    baseline_correct: !spectrum.parameters.baseline_corrected,
    baseline_method: "asymmetric_least_squares" as "asymmetric_least_squares" | "percentile",
    baseline_smoothness: 1e7,
    baseline_asymmetry: 0.001,
    baseline_iterations: 8,
    baseline_percentile: 10,
    auto_phase: false,
    auto_phase_first_order: true,
    auto_phase_max_first_deg: 720,
    auto_reference: false,
    reference_window_ppm: 0.12,
    reference_min_snr: 5,
    reference_current_ppm: "",
    reference_target_ppm: "7.26",
    normalize: false,
    normalize_ppm: "",
    normalize_window_points: 3,
    invert: false,
    smoothing_window: 0,
    crop_min_ppm: "",
    crop_max_ppm: "",
    phase_zero_deg: 0,
    phase_first_deg: 0,
    phase_pivot_ppm: "",
  };
}

export function optionalFiniteNumber(value: string | number): number | undefined {
  if (typeof value === "string" && value.trim() === "") return undefined;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function formatQualityValue(value: unknown, digits = 3): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const absolute = Math.abs(value);
  if (absolute !== 0 && (absolute < 0.001 || absolute >= 10000)) {
    return value.toExponential(2);
  }
  return value.toFixed(digits);
}

export default function NMRWorkbench({
  spectrum,
  result,
  spectra,
  onSpectrumChanged,
  onResultChanged,
  onError,
  onDirtyChange,
}: Props) {
  const { t } = useLang();
  const [analysisOptions, setAnalysisOptions] = useState<AnalysisOptions>({ auto_reference: false });
  const [processing, setProcessing] = useState(() => initialProcessing(spectrum));
  const [processingBaseline, setProcessingBaseline] = useState(() => JSON.stringify(initialProcessing(spectrum)));
  const [busy, setBusy] = useState(false);
  const [plotPickTarget, setPlotPickTarget] = useState<PlotPickTarget>(null);
  const [plotPickedRange, setPlotPickedRange] = useState<PlotPickedRange>(null);
  const [activeTab, setActiveTab] = useState<"process" | "review" | "multiplets" | "report">("process");
  const [notice, setNotice] = useState("");
  const [multipletRangeText, setMultipletRangeText] = useState("");
  const [multipletRangeBaseline, setMultipletRangeBaseline] = useState("");
  const [manualDraftEpoch, setManualDraftEpoch] = useState(0);
  const [confirmProcessing, setConfirmProcessing] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const [confirmReanalysis, setConfirmReanalysis] = useState(false);
  const [confirmTabLeave, setConfirmTabLeave] = useState(false);
  const [pendingTab, setPendingTab] = useState<typeof activeTab | null>(null);
  const [confirmDiscardForAnalysis, setConfirmDiscardForAnalysis] = useState(false);
  const [pendingMultipletRebuild, setPendingMultipletRebuild] = useState<PendingMultipletRebuild>(null);
  const [previewSpectrum, setPreviewSpectrum] = useState<SpectrumData | null>(null);
  const [previewSourceRevision, setPreviewSourceRevision] = useState<number | null>(null);
  const [originalSpectrum, setOriginalSpectrum] = useState<NmrViewSpectrum | null>(null);
  const [spectrumView, setSpectrumView] = useState<"original" | "current" | "preview">("current");
  const [qualityBefore, setQualityBefore] = useState<Record<string, unknown> | null>(null);
  const [qualityAfter, setQualityAfter] = useState<Record<string, unknown> | null>(null);
  const [processingWarnings, setProcessingWarnings] = useState<string[]>([]);
  const [manualDirty, setManualDirty] = useState(false);
  const resultRevision = result?.result_revision ?? 0;
  const previousResultRevision = useRef(resultRevision);
  const phaseAvailable = spectrum.parameters.quadrature_available === true
    || spectrum.parameters.phase_correction_available === true
    || spectrum.parameters.has_complex_data === true
    || spectrum.parameters.complex_data_preserved === true;
  const processingDirty = useMemo(
    () => JSON.stringify(processing) !== processingBaseline,
    [processing, processingBaseline],
  );
  const multipletDirty = multipletRangeText !== multipletRangeBaseline;
  const draftDirty = processingDirty || manualDirty || multipletDirty;

  useEffect(() => {
    onDirtyChange?.(draftDirty);
  }, [draftDirty, onDirtyChange]);

  useEffect(() => () => onDirtyChange?.(false), [onDirtyChange]);

  useEffect(() => {
    if (previousResultRevision.current === resultRevision) return;
    previousResultRevision.current = resultRevision;
    setManualDraftEpoch((value) => value + 1);
    setManualDirty(false);
    setMultipletRangeText("");
    setMultipletRangeBaseline("");
    setPendingMultipletRebuild(null);
    setPreviewSpectrum(null);
    setPreviewSourceRevision(null);
    setConfirmProcessing(false);
    setSpectrumView("current");
    setQualityBefore(null);
    setQualityAfter(null);
    setProcessingWarnings([]);
  }, [resultRevision]);

  const processingPayload = (previewOnly: boolean, expectedRevision: number) => {
    const referenceCurrent = optionalFiniteNumber(processing.reference_current_ppm);
    const useAls = processing.baseline_method === "asymmetric_least_squares";
    const useAutoPhase = phaseAvailable && processing.auto_phase;
    const useAutoReference = processing.auto_reference;
    return {
      baseline_correct: processing.baseline_correct,
      baseline_method: processing.baseline_method,
      baseline_smoothness: processing.baseline_correct && useAls
        ? processing.baseline_smoothness
        : undefined,
      baseline_asymmetry: processing.baseline_correct && useAls
        ? processing.baseline_asymmetry
        : undefined,
      baseline_iterations: processing.baseline_correct && useAls
        ? processing.baseline_iterations
        : undefined,
      baseline_percentile: processing.baseline_correct && !useAls
        ? processing.baseline_percentile
        : undefined,
      auto_phase: useAutoPhase,
      auto_phase_first_order: useAutoPhase
        ? processing.auto_phase_first_order
        : undefined,
      auto_phase_max_first_deg: useAutoPhase
        ? processing.auto_phase_max_first_deg
        : undefined,
      auto_reference: useAutoReference,
      reference_solvent: useAutoReference
        ? displayText(spectrum.metadata?.solvent || spectrum.parameters.solvent).trim() || undefined
        : undefined,
      reference_window_ppm: useAutoReference
        ? processing.reference_window_ppm
        : undefined,
      reference_min_snr: useAutoReference
        ? processing.reference_min_snr
        : undefined,
      reference_current_ppm: useAutoReference ? undefined : referenceCurrent,
      reference_target_ppm: useAutoReference || referenceCurrent === undefined
        ? undefined
        : optionalFiniteNumber(processing.reference_target_ppm),
      normalize: processing.normalize,
      normalize_ppm: processing.normalize
        ? optionalFiniteNumber(processing.normalize_ppm)
        : undefined,
      normalize_window_points: processing.normalize_window_points,
      invert: processing.invert,
      smoothing_window: processing.smoothing_window,
      crop_min_ppm: optionalFiniteNumber(processing.crop_min_ppm),
      crop_max_ppm: optionalFiniteNumber(processing.crop_max_ppm),
      phase_zero_deg: phaseAvailable && !useAutoPhase && processing.phase_zero_deg !== 0
        ? optionalFiniteNumber(processing.phase_zero_deg)
        : undefined,
      phase_first_deg: phaseAvailable && !useAutoPhase && processing.phase_first_deg !== 0
        ? optionalFiniteNumber(processing.phase_first_deg)
        : undefined,
      phase_pivot_ppm: phaseAvailable && !useAutoPhase && processing.phase_pivot_ppm
        ? optionalFiniteNumber(processing.phase_pivot_ppm)
        : undefined,
      replay_from_original: true,
      preview_only: previewOnly,
      expected_revision: expectedRevision,
    };
  };

  const currentSpectrumRevision = (): number | null => {
    const revision = spectrum.spectrum_revision;
    if (typeof revision !== "number" || revision < 1) {
      onError(t.manual.revisionConflict);
      return null;
    }
    return revision;
  };

  const previewProcessing = async () => {
    const expectedRevision = currentSpectrumRevision();
    if (expectedRevision === null) return;
    setBusy(true);
    setNotice("");
    setPreviewSpectrum(null);
    setPreviewSourceRevision(null);
    setSpectrumView("current");
    setQualityBefore(null);
    setQualityAfter(null);
    setProcessingWarnings([]);
    onError("");
    try {
      const preview = await processNmrSpectrum(
        spectrum.id,
        processingPayload(true, expectedRevision),
      );
      setPreviewSourceRevision(expectedRevision);
      setPreviewSpectrum(preview);
      setSpectrumView("preview");
      setQualityBefore(preview.quality_before || null);
      setQualityAfter(preview.quality_after || null);
      setProcessingWarnings(preview.warnings || []);
      setConfirmProcessing(true);
      setNotice(t.workbench.previewReady);
    } catch (cause: unknown) {
      onError(cause instanceof ApiError && cause.status === 409
        ? t.manual.revisionConflict
        : cause instanceof Error ? cause.message : t.workbench.processingFailed);
    } finally {
      setBusy(false);
    }
  };

  const applyProcessing = async () => {
    const expectedRevision = previewSourceRevision;
    if (expectedRevision === null) {
      setConfirmProcessing(false);
      onError(t.manual.revisionConflict);
      return;
    }
    setConfirmProcessing(false);
    setBusy(true);
    setNotice("");
    setProcessingWarnings([]);
    onError("");
    try {
      const updated = await processNmrSpectrum(
        spectrum.id,
        processingPayload(false, expectedRevision),
      );
      onSpectrumChanged(updated);
      onResultChanged(null);
      setPreviewSpectrum(null);
      setPreviewSourceRevision(null);
      setSpectrumView("current");
      setQualityBefore(updated.quality_before || null);
      setQualityAfter(updated.quality_after || null);
      setProcessingWarnings(updated.warnings || []);
      const applied = Array.isArray(updated.processing_applied) ? updated.processing_applied : [];
      if (applied.length === 0) {
        setNotice(t.workbench.noProcessingApplied);
      } else {
        setNotice(t.workbench.processingApplied.replace("{count}", String(applied.length)));
      }
      const reset = initialProcessing(updated);
      setProcessing(reset);
      setProcessingBaseline(JSON.stringify(reset));
      setManualDirty(false);
    } catch (cause: unknown) {
      onError(cause instanceof ApiError && cause.status === 409
        ? t.manual.revisionConflict
        : cause instanceof Error ? cause.message : t.workbench.processingFailed);
    } finally {
      setBusy(false);
    }
  };

  const runAnalysis = async (forceOverwrite = false) => {
    setConfirmReanalysis(false);
    setBusy(true);
    setNotice("");
    setProcessingWarnings([]);
    onError("");
    try {
      const data = await analyzeSpectrum(
        spectrum.id,
        analysisOptions,
        undefined,
        result?.result_revision ?? 0,
        forceOverwrite,
      );
      onResultChanged(data);
      setNotice(t.workbench.analysisUpdated);
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : t.workbench.analysisFailed);
    } finally {
      setBusy(false);
    }
  };

  const discardLocalDrafts = () => {
    const reset = initialProcessing(spectrum);
    setProcessing(reset);
    setProcessingBaseline(JSON.stringify(reset));
    setManualDraftEpoch((value) => value + 1);
    setManualDirty(false);
    setMultipletRangeText(multipletRangeBaseline);
  };

  const requestAnalysis = () => {
    if (draftDirty) {
      setConfirmDiscardForAnalysis(true);
      return;
    }
    if (result?.metrics.manual_confirmed) setConfirmReanalysis(true);
    else void runAnalysis(false);
  };

  const confirmDiscardManualForAnalysis = () => {
    setConfirmDiscardForAnalysis(false);
    discardLocalDrafts();
    if (result?.metrics.manual_confirmed) {
      setConfirmReanalysis(true);
      return;
    }
    void runAnalysis(false);
  };

  const requestTab = (next: typeof activeTab) => {
    if (next === activeTab) return;
    if (draftDirty) {
      setPendingTab(next);
      setConfirmTabLeave(true);
      return;
    }
    setActiveTab(next);
  };

  const confirmLeaveTab = () => {
    const next = pendingTab;
    setConfirmTabLeave(false);
    setPendingTab(null);
    if (!next) return;
    setActiveTab(next);
    discardLocalDrafts();
  };

  const resetProcessing = async () => {
    const expectedRevision = currentSpectrumRevision();
    if (expectedRevision === null) {
      setConfirmReset(false);
      return;
    }
    setConfirmReset(false);
    setBusy(true);
    setNotice("");
    onError("");
    try {
      const reset = await resetNmrSpectrum(spectrum.id, expectedRevision);
      onSpectrumChanged(reset);
      onResultChanged(null);
      setPreviewSpectrum(null);
      setPreviewSourceRevision(null);
      setOriginalSpectrum(reset);
      setSpectrumView("current");
      setQualityBefore(null);
      setQualityAfter(null);
      setProcessingWarnings(reset.warnings || []);
      const initial = initialProcessing(reset);
      setProcessing(initial);
      setProcessingBaseline(JSON.stringify(initial));
      setNotice(t.workbench.resetApplied);
    } catch (cause: unknown) {
      onError(cause instanceof ApiError && cause.status === 409
        ? t.manual.revisionConflict
        : cause instanceof Error ? cause.message : t.workbench.processingFailed);
    } finally {
      setBusy(false);
    }
  };

  const selectSpectrumView = async (
    nextView: "original" | "current" | "preview",
  ) => {
    if (nextView === "preview") {
      if (previewSpectrum) setSpectrumView("preview");
      return;
    }
    if (nextView === "current") {
      setSpectrumView("current");
      return;
    }
    if (originalSpectrum) {
      setSpectrumView("original");
      return;
    }
    setBusy(true);
    onError("");
    try {
      const original = await getNmrSpectrumView(spectrum.id, "original");
      setOriginalSpectrum(original);
      setSpectrumView("original");
    } catch (cause: unknown) {
      onError(cause instanceof Error ? cause.message : t.workbench.processingFailed);
    } finally {
      setBusy(false);
    }
  };

  const applyAnalysisPreset = (preset: "mnova" | "auto-reference") => {
    if (preset === "mnova") {
      setAnalysisOptions({
        auto_reference: false,
        baseline_percentile: 10,
        noise_factor: 3,
        prominence_factor: 8,
        min_peak_distance: 5,
      });
      setNotice(t.workbench.mnovaPreset);
      return;
    }

    setAnalysisOptions({
      auto_reference: true,
      baseline_percentile: 10,
      noise_factor: 3,
      prominence_factor: 8,
      min_peak_distance: 5,
    });
    setNotice(t.workbench.autoReferencePreset);
  };

  const rebuildMultiplets = async (
    ranges: Array<{ start: number; end: number }>,
    expectedRevision: number,
    forceOverwrite: boolean,
  ) => {
    setAnalysisOptions((prev) => ({ ...prev, multiplet_ranges: ranges }));
    setBusy(true);
    onError("");
    try {
      const data = await analyzeSpectrum(
        spectrum.id,
        { ...analysisOptions, multiplet_ranges: ranges },
        undefined,
        expectedRevision,
        forceOverwrite,
      );
      setMultipletRangeBaseline(multipletRangeText);
      onResultChanged(data);
      setNotice(t.workbench.multipletsRebuilt.replace("{count}", String(ranges.length)));
    } catch (cause: unknown) {
      onError(cause instanceof ApiError && cause.status === 409
        ? t.manual.revisionConflict
        : cause instanceof Error ? cause.message : t.workbench.multipletFailed);
    } finally {
      setBusy(false);
    }
  };

  const applyMultipletRanges = () => {
    const ranges = multipletRangeText
      .split(/[;\n]+/)
      .map((chunk) => chunk.trim())
      .filter(Boolean)
      .map((chunk) => {
        const parts = chunk.split(/\.\.|,|\s+/).map((p) => Number(p.trim())).filter(Number.isFinite);
        return parts.length >= 2 ? { start: parts[0], end: parts[1] } : null;
      })
      .filter((item): item is { start: number; end: number } => item !== null);
    if (!ranges.length) {
      setNotice(t.workbench.invalidMultipletRanges);
      return;
    }
    const expectedRevision = result?.result_revision ?? 0;
    if (result?.metrics.manual_confirmed) {
      setPendingMultipletRebuild({ ranges, expectedRevision });
      return;
    }
    void rebuildMultiplets(ranges, expectedRevision, false);
  };

  const confirmMultipletRebuild = () => {
    const pending = pendingMultipletRebuild;
    setPendingMultipletRebuild(null);
    if (!pending) return;
    void rebuildMultiplets(pending.ranges, pending.expectedRevision, true);
  };

  const saveMultipletConfirmation = async () => {
    if (!result) return;
    setBusy(true);
    setNotice("");
    onError("");
    try {
      const saved = await saveManualResult(spectrum.id, {
        peaks: result.peaks,
        integrals: result.integrals,
        multiplets: result.multiplets,
        metrics: {
          ...result.metrics,
          n_multiplets: result.multiplets?.length || 0,
        },
        summary: result.summary,
        note: "NMR multiplet confirmation",
        expected_revision: result.result_revision ?? 0,
      });
      onResultChanged(saved);
      setMultipletRangeBaseline(multipletRangeText);
      setNotice(t.workbench.multipletSaved);
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : t.workbench.multipletSaveFailed);
    } finally {
      setBusy(false);
    }
  };

  const exportReport = async () => {
    onError("");
    try {
      await downloadMarkdownReport([spectrum.id], "ChemApp 1D NMR Report");
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : t.action.reportFailed);
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
    regions: result?.integrals?.map((item) => ({
      start: item.start_ppm,
      end: item.end_ppm,
      label: String(item.center_ppm),
    })) || [],
  } : null;

  const history = (spectrum.parameters.processing_history as Array<Record<string, unknown>> | undefined) || [];
  const autoPhaseApplied = spectrum.parameters.phase_corrected === true;
  const displayedSpectrum = spectrumView === "original" && originalSpectrum
    ? originalSpectrum
    : spectrumView === "preview" && previewSpectrum
      ? previewSpectrum
      : spectrum;
  const displayedResult = spectrumView === "current" ? result : null;
  const displayedViewQuality = (
    displayedSpectrum as NmrViewSpectrum
  ).quality_metrics || null;
  const comparisonQualityBefore = qualityBefore;
  const comparisonQualityAfter = qualityAfter || displayedViewQuality;
  const qualityRows = [
    ["noise_sigma", t.workbench.qualityNoise],
    ["max_abs_signal_to_noise", t.workbench.qualitySnr],
    ["baseline_offset", t.workbench.qualityBaselineOffset],
    ["baseline_span", t.workbench.qualityBaselineSpan],
    ["baseline_rms", t.workbench.qualityBaselineRms],
    ["baseline_slope_per_x", t.workbench.qualityBaselineSlope],
    ["negative_energy_fraction", t.workbench.qualityNegativeEnergy],
    ["imaginary_energy_fraction", t.workbench.qualityImaginaryEnergy],
  ] as const;
  const qualityFlags = Array.isArray(comparisonQualityAfter?.quality_flags)
    ? comparisonQualityAfter.quality_flags.map(String)
    : [];
  const qualityFlagLabels: Record<string, string> = {
    empty_spectrum: t.workbench.qualityFlagEmpty,
    low_point_count: t.workbench.qualityFlagLowPoints,
    non_uniform_axis: t.workbench.qualityFlagNonUniformAxis,
    baseline_drift: t.workbench.qualityFlagBaselineDrift,
    high_negative_energy: t.workbench.qualityFlagNegativeEnergy,
    high_imaginary_energy: t.workbench.qualityFlagImaginaryEnergy,
    low_signal_to_noise: t.workbench.qualityFlagLowSnr,
  };
  const processingSummary = [
    processing.baseline_correct ? t.workbench.baselineCorrection : "",
    processing.auto_phase ? t.workbench.autoPhase : "",
    processing.auto_reference ? t.workbench.autoReferenceProcessing : "",
    processing.reference_current_ppm ? t.workbench.reference : "",
    processing.normalize ? t.workbench.normalize : "",
    processing.smoothing_window > 0 ? t.workbench.smoothing : "",
    processing.crop_min_ppm || processing.crop_max_ppm ? t.workbench.crop : "",
    processing.invert ? t.workbench.invert : "",
    phaseAvailable && (processing.phase_zero_deg !== 0 || processing.phase_first_deg !== 0)
      ? t.workbench.phase
      : "",
  ].filter(Boolean).join(", ") || t.workbench.noOperations;

  return (
    <div className="nmr-workbench">
      <div className="workbench-topbar">
        <div>
          <h2>{t.workbench.title}</h2>
          <p>{spectrum.parameters.nucleus as string || "1H"} · {spectrum.parameters.frequency_mhz as number || ""} MHz · {spectrum.metadata?.solvent as string || spectrum.parameters.solvent as string || t.workbench.unknownSolvent}</p>
        </div>
        <div className="workbench-actions">
          <button type="button" onClick={requestAnalysis} disabled={busy}>{busy ? t.workbench.busy : t.workbench.analyze}</button>
          <button type="button" onClick={() => void exportReport()} disabled={!result}>{t.workbench.exportReport}</button>
        </div>
      </div>

      <SpectrumViewer
        spectrum={displayedSpectrum}
        result={displayedResult}
        integrationSelection={spectrumView === "current" ? integrationSelection : null}
        onIntegrationRangeSelected={handlePlotRangeSelected}
      />
      <div className="spectrum-view-switcher" role="group" aria-label={t.workbench.spectrumViews}>
        <button
          type="button"
          aria-pressed={spectrumView === "original"}
          onClick={() => void selectSpectrumView("original")}
          disabled={busy}
        >
          {t.workbench.originalView}
        </button>
        <button
          type="button"
          aria-pressed={spectrumView === "current"}
          onClick={() => void selectSpectrumView("current")}
          disabled={busy}
        >
          {t.workbench.currentView}
        </button>
        {previewSpectrum && (
          <button
            type="button"
            aria-pressed={spectrumView === "preview"}
            onClick={() => void selectSpectrumView("preview")}
            disabled={busy}
          >
            {t.workbench.previewView}
          </button>
        )}
      </div>
      {spectrumView === "original" && (
        <p className="preview-banner" role="status">{t.workbench.originalBanner}</p>
      )}
      {spectrumView === "preview" && previewSpectrum && (
        <p className="preview-banner" role="status">{t.workbench.previewBanner}</p>
      )}

      <div className="workbench-tabs" role="tablist" aria-label={t.workbench.tabsLabel}>
        <button type="button" role="tab" aria-selected={activeTab === "process"} className={activeTab === "process" ? "active" : ""} onClick={() => requestTab("process")}>{t.workbench.process}</button>
        <button type="button" role="tab" aria-selected={activeTab === "review"} className={activeTab === "review" ? "active" : ""} onClick={() => requestTab("review")} disabled={!result}>{t.workbench.review}</button>
        <button type="button" role="tab" aria-selected={activeTab === "multiplets"} className={activeTab === "multiplets" ? "active" : ""} onClick={() => requestTab("multiplets")} disabled={!result}>{t.workbench.multiplets}</button>
        <button type="button" role="tab" aria-selected={activeTab === "report"} className={activeTab === "report" ? "active" : ""} onClick={() => requestTab("report")} disabled={!result}>{t.workbench.confirmReport}</button>
      </div>

      {notice && <p className="workbench-notice" role="status" aria-live="polite">{notice}</p>}
      {processingWarnings.length > 0 && (
        <ul className="prediction-warnings" aria-label={t.prediction.warnings}>
          {processingWarnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      )}

      {activeTab === "process" && (
        <div className="workbench-grid">
          <section className="workbench-section">
            <h3>{t.workbench.processingTitle}</h3>
            {autoPhaseApplied && (
              <p className="summary">
                {t.workbench.autoPhaseSummary
                  .replace("{p0}", Number(spectrum.parameters.auto_phase_zero_deg || 0).toFixed(2))
                  .replace("{p1}", Number(spectrum.parameters.auto_phase_first_deg || 0).toFixed(2))
                  .replace("{delay}", Number(spectrum.parameters.digital_filter_points || 0).toFixed(3))}
              </p>
            )}
            {!phaseAvailable && (
              <p className="warning-note" role="note">{t.workbench.phaseUnavailable}</p>
            )}
            <div className="processing-grid">
              <label className="checkbox-control">
                <input type="checkbox" checked={processing.baseline_correct}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, baseline_correct: e.target.checked }))} />
                {t.workbench.baselineCorrection}
              </label>
              <label>
                {t.workbench.baselineMethod}
                <select
                  value={processing.baseline_method}
                  disabled={!processing.baseline_correct}
                  onChange={(e) => setProcessing((prev) => ({
                    ...prev,
                    baseline_method: e.target.value as "asymmetric_least_squares" | "percentile",
                  }))}
                >
                  <option value="asymmetric_least_squares">{t.workbench.baselineAls}</option>
                  <option value="percentile">{t.workbench.baselinePercentileMethod}</option>
                </select>
              </label>
              {processing.baseline_method === "asymmetric_least_squares" ? (
                <>
                  <label>
                    {t.workbench.baselineSmoothness}
                    <input type="number" min={1} step={1000000} value={processing.baseline_smoothness}
                      disabled={!processing.baseline_correct}
                      onChange={(e) => setProcessing((prev) => ({ ...prev, baseline_smoothness: Number(e.target.value) }))} />
                  </label>
                  <label>
                    {t.workbench.baselineAsymmetry}
                    <input type="number" min={0.000001} max={0.999999} step={0.001} value={processing.baseline_asymmetry}
                      disabled={!processing.baseline_correct}
                      onChange={(e) => setProcessing((prev) => ({ ...prev, baseline_asymmetry: Number(e.target.value) }))} />
                  </label>
                  <label>
                    {t.workbench.baselineIterations}
                    <input type="number" min={1} max={50} value={processing.baseline_iterations}
                      disabled={!processing.baseline_correct}
                      onChange={(e) => setProcessing((prev) => ({ ...prev, baseline_iterations: Number(e.target.value) }))} />
                  </label>
                </>
              ) : (
                <label>
                  {t.workbench.baselinePercentile}
                  <input type="number" min={0} max={100} value={processing.baseline_percentile}
                    disabled={!processing.baseline_correct}
                    onChange={(e) => setProcessing((prev) => ({ ...prev, baseline_percentile: Number(e.target.value) }))} />
                </label>
              )}
              <label className="checkbox-control">
                <input type="checkbox" checked={processing.auto_reference}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, auto_reference: e.target.checked }))} />
                {t.workbench.autoReferenceProcessing}
              </label>
              <label>
                {t.workbench.referenceWindow}
                <input type="number" min={0.01} max={1} step={0.01}
                  value={processing.reference_window_ppm} disabled={!processing.auto_reference}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, reference_window_ppm: Number(e.target.value) }))} />
              </label>
              <label>
                {t.workbench.referenceMinSnr}
                <input type="number" min={1} max={1000} step={1}
                  value={processing.reference_min_snr} disabled={!processing.auto_reference}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, reference_min_snr: Number(e.target.value) }))} />
              </label>
              <label>
                {t.workbench.currentReference}
                <input value={processing.reference_current_ppm} disabled={processing.auto_reference}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, reference_current_ppm: e.target.value }))} placeholder={t.workbench.referenceExample} />
              </label>
              <label>
                {t.workbench.targetPpm}
                <input value={processing.reference_target_ppm} disabled={processing.auto_reference}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, reference_target_ppm: e.target.value }))} />
              </label>
              <label className="checkbox-control">
                <input type="checkbox" checked={processing.normalize}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, normalize: e.target.checked }))} />
                {t.workbench.normalize}
              </label>
              <label>
                {t.workbench.normalizePpm}
                <input value={processing.normalize_ppm}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, normalize_ppm: e.target.value }))} placeholder={t.workbench.normalizeHint} />
              </label>
              <label>
                {t.workbench.smoothingWindow}
                <input type="number" min={0} max={101} step={2} value={processing.smoothing_window}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, smoothing_window: Number(e.target.value) }))} />
              </label>
              <label>
                {t.workbench.cropMin}
                <input value={processing.crop_min_ppm}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, crop_min_ppm: e.target.value }))} placeholder={t.workbench.optional} />
              </label>
              <label>
                {t.workbench.cropMax}
                <input value={processing.crop_max_ppm}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, crop_max_ppm: e.target.value }))} placeholder={t.workbench.optional} />
              </label>
              <label className="checkbox-control">
                <input type="checkbox" checked={processing.invert}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, invert: e.target.checked }))} />
                {t.workbench.invert}
              </label>
              <label className="checkbox-control">
                <input type="checkbox" checked={processing.auto_phase} disabled={!phaseAvailable}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, auto_phase: e.target.checked }))} />
                {t.workbench.autoPhase}
              </label>
              <label className="checkbox-control">
                <input type="checkbox" checked={processing.auto_phase_first_order}
                  disabled={!phaseAvailable || !processing.auto_phase}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, auto_phase_first_order: e.target.checked }))} />
                {t.workbench.autoPhaseFirstOrder}
              </label>
              <label>
                {t.workbench.phaseZero}
                <input type="number" step={0.5} value={processing.phase_zero_deg} disabled={!phaseAvailable || processing.auto_phase}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, phase_zero_deg: Number(e.target.value) }))} />
              </label>
              <label>
                {t.workbench.phaseFirst}
                <input type="number" step={0.5} value={processing.phase_first_deg} disabled={!phaseAvailable || processing.auto_phase}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, phase_first_deg: Number(e.target.value) }))} />
              </label>
              <label>
                {t.workbench.phasePivot}
                <input value={processing.phase_pivot_ppm} disabled={!phaseAvailable || processing.auto_phase}
                  onChange={(e) => setProcessing((prev) => ({ ...prev, phase_pivot_ppm: e.target.value }))} placeholder={t.workbench.phasePivotHint} />
              </label>
            </div>
            <div className="preset-row">
              <button type="button" className="primary-inline" onClick={() => void previewProcessing()} disabled={busy}>{t.workbench.previewApply}</button>
              <button
                type="button"
                onClick={() => {
                  setProcessing(initialProcessing(spectrum));
                  setPreviewSpectrum(null);
                  setSpectrumView("current");
                  setQualityBefore(null);
                  setQualityAfter(null);
                  setProcessingWarnings([]);
                  setNotice("");
                }}
                disabled={busy || !processingDirty}
              >
                {t.workbench.resetDraft}
              </button>
              <button
                type="button"
                className="danger-outline"
                onClick={() => setConfirmReset(true)}
                disabled={busy || !spectrum.spectrum_revision}
              >
                {t.workbench.restoreOriginal}
              </button>
            </div>
            {(comparisonQualityBefore || comparisonQualityAfter) && (
              <section className="processing-quality" aria-labelledby="processing-quality-title">
                <h4 id="processing-quality-title">{t.workbench.qualityTitle}</h4>
                <p>{t.workbench.qualityHint}</p>
                <div className="processing-quality-table" role="table" aria-label={t.workbench.qualityTitle}>
                  <div className={`processing-quality-row processing-quality-heading${comparisonQualityBefore ? "" : " single"}`} role="row">
                    <span role="columnheader">{t.workbench.qualityMetric}</span>
                    {comparisonQualityBefore && (
                      <span role="columnheader">{t.workbench.qualityBefore}</span>
                    )}
                    <span role="columnheader">{comparisonQualityBefore
                      ? t.workbench.qualityAfter
                      : t.workbench.qualityCurrent}</span>
                  </div>
                  {qualityRows.map(([key, label]) => (
                    <div className={`processing-quality-row${comparisonQualityBefore ? "" : " single"}`} role="row" key={key}>
                      <span role="rowheader">{label}</span>
                      {comparisonQualityBefore && (
                        <span role="cell">{formatQualityValue(comparisonQualityBefore[key])}</span>
                      )}
                      <span role="cell">{formatQualityValue(comparisonQualityAfter?.[key])}</span>
                    </div>
                  ))}
                </div>
                {qualityFlags.length > 0 ? (
                  <div className="processing-quality-flags">
                    <strong>{t.workbench.qualityFlags}</strong>
                    <ul>
                      {qualityFlags.map((flag) => (
                        <li key={flag}>{qualityFlagLabels[flag] || flag}</li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <p className="processing-quality-clear">{t.workbench.qualityNoFlags}</p>
                )}
              </section>
            )}
            {history.length > 0 && (
              <div className="processing-history">
                <h4>{t.workbench.history}</h4>
                {history.slice(-6).map((item, index) => (
                  <span key={index}>{displayText(item.type) || "operation"}</span>
                ))}
              </div>
            )}
          </section>

          <section className="workbench-section">
            <h3>{t.workbench.peakPicking}</h3>
            <div className="preset-row">
              <button type="button" onClick={() => applyAnalysisPreset("mnova")} disabled={busy}>{t.workbench.mnovaAlign}</button>
              <button type="button" onClick={() => applyAnalysisPreset("auto-reference")} disabled={busy}>{t.workbench.autoSolvent}</button>
            </div>
            <AnalysisControls technique="NMR" value={analysisOptions} onChange={setAnalysisOptions} disabled={busy} />
          </section>
        </div>
      )}

      {result && activeTab === "review" && (
        <>
          <QualityPanel quality={result.metrics.quality} />
          <ManualReviewPanel
            key={`${spectrum.id}:${result.result_revision ?? 0}:${manualDraftEpoch}`}
            spectrum={spectrum}
            result={result}
            onSaved={onResultChanged}
            activePlotPick={plotPickTarget}
            pickedPlotRange={plotPickedRange}
            onRequestPlotPick={setPlotPickTarget}
            onDirtyChange={setManualDirty}
          />
        </>
      )}

      {result && activeTab === "multiplets" && (
        <>
          <div className="workbench-section">
            <h3>{t.workbench.manualMultipletRanges}</h3>
            <textarea
              aria-label={t.workbench.manualMultipletRanges}
              value={multipletRangeText}
              onChange={(e) => setMultipletRangeText(e.target.value)}
              placeholder={t.workbench.multipletExample}
              rows={3}
            />
            <button type="button" className="primary-inline" onClick={applyMultipletRanges} disabled={busy}>{t.workbench.rebuildMultiplets}</button>
            <button type="button" className="primary-inline" onClick={() => void saveMultipletConfirmation()} disabled={busy || !result.multiplets?.length}>{t.workbench.saveMultiplets}</button>
          </div>
          <NMRPanel result={result} />
        </>
      )}

      {result && activeTab === "report" && (
        <div className="workbench-grid">
          <section className="workbench-section">
            <h3>{t.workbench.reviewStatus}</h3>
            <p className="summary">
              {result.metrics.manual_confirmed
                ? t.workbench.reviewSaved.replace("{version}", displayText(result.metrics.manual_version) || "1")
                : t.workbench.reviewNotSaved}
            </p>
            <button type="button" className="primary-inline" onClick={() => void exportReport()}>{t.workbench.exportMarkdown}</button>
          </section>
          <section className="workbench-section">
            <h3>{t.workbench.structureAssistance}</h3>
            <MLPredictionPanel spectrumId={spectrum.id} hasResult={true} technique={result.technique} result={result}
              nucleus={spectrum.parameters?.nucleus as string || ""} spectra={spectra}
              solvent={displayText(spectrum.metadata?.solvent || spectrum.parameters?.solvent)} />
          </section>
        </div>
      )}
      <ConfirmDialog
        open={confirmProcessing}
        title={t.workbench.confirmProcessingTitle}
        message={t.workbench.confirmProcessingMessage.replace("{operations}", processingSummary)}
        confirmLabel={t.workbench.applyProcessing}
        cancelLabel={t.action.cancel}
        busy={busy}
        busyLabel={t.workbench.busy}
        onConfirm={() => void applyProcessing()}
        onCancel={() => {
          setConfirmProcessing(false);
          setPreviewSpectrum(null);
          setPreviewSourceRevision(null);
          setSpectrumView("current");
          setQualityBefore(null);
          setQualityAfter(null);
          setProcessingWarnings([]);
          setNotice("");
        }}
      />
      <ConfirmDialog
        open={confirmReset}
        title={t.workbench.confirmResetTitle}
        message={t.workbench.confirmResetMessage}
        confirmLabel={t.workbench.restoreOriginal}
        cancelLabel={t.action.cancel}
        busy={busy}
        busyLabel={t.workbench.busy}
        danger
        onConfirm={() => void resetProcessing()}
        onCancel={() => setConfirmReset(false)}
      />
      <ConfirmDialog
        open={confirmReanalysis}
        title={t.workbench.confirmReanalysisTitle}
        message={t.workbench.confirmReanalysisMessage}
        confirmLabel={t.workbench.overwriteAndAnalyze}
        cancelLabel={t.action.cancel}
        busy={busy}
        busyLabel={t.workbench.busy}
        danger
        onConfirm={() => void runAnalysis(true)}
        onCancel={() => setConfirmReanalysis(false)}
      />
      <ConfirmDialog
        open={pendingMultipletRebuild !== null}
        title={t.workbench.confirmReanalysisTitle}
        message={t.workbench.confirmReanalysisMessage}
        confirmLabel={t.workbench.overwriteAndAnalyze}
        cancelLabel={t.action.cancel}
        busy={busy}
        busyLabel={t.workbench.busy}
        danger
        onConfirm={confirmMultipletRebuild}
        onCancel={() => setPendingMultipletRebuild(null)}
      />
      <ConfirmDialog
        open={confirmTabLeave}
        title={t.action.unsavedTitle}
        message={t.action.unsavedMessage}
        confirmLabel={t.action.discard}
        cancelLabel={t.action.keepEditing}
        danger
        onConfirm={confirmLeaveTab}
        onCancel={() => {
          setConfirmTabLeave(false);
          setPendingTab(null);
        }}
      />
      <ConfirmDialog
        open={confirmDiscardForAnalysis}
        title={t.action.unsavedTitle}
        message={t.action.unsavedMessage}
        confirmLabel={t.action.discard}
        cancelLabel={t.action.keepEditing}
        danger
        onConfirm={confirmDiscardManualForAnalysis}
        onCancel={() => setConfirmDiscardForAnalysis(false)}
      />
    </div>
  );
}
