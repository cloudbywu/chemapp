import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StrictMode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LangProvider } from "../i18n/LangContext";
import type * as api from "../services/api";
import type { AnalysisResult, SpectrumData } from "../types/spectrum";
import NMRWorkbench, { optionalFiniteNumber } from "./NMRWorkbench";

const processNmrSpectrum = vi.fn<typeof api.processNmrSpectrum>();
const getNmrSpectrumView = vi.fn<typeof api.getNmrSpectrumView>();
const analyzeSpectrum = vi.fn<typeof api.analyzeSpectrum>();
const resetNmrSpectrum = vi.fn<typeof api.resetNmrSpectrum>();
const saveManualResult = vi.fn<typeof api.saveManualResult>();
const downloadMarkdownReport = vi.fn<typeof api.downloadMarkdownReport>();

vi.mock("../services/api", () => ({
  analyzeSpectrum: (...args: Parameters<typeof api.analyzeSpectrum>) => analyzeSpectrum(...args),
  ApiError: class ApiError extends Error {
    status: number;

    constructor(message: string, status: number) {
      super(message);
      this.status = status;
    }
  },
  downloadMarkdownReport: (...args: Parameters<typeof api.downloadMarkdownReport>) => downloadMarkdownReport(...args),
  getNmrSpectrumView: (...args: Parameters<typeof api.getNmrSpectrumView>) => getNmrSpectrumView(...args),
  processNmrSpectrum: (...args: Parameters<typeof api.processNmrSpectrum>) => processNmrSpectrum(...args),
  resetNmrSpectrum: (...args: Parameters<typeof api.resetNmrSpectrum>) => resetNmrSpectrum(...args),
  saveManualResult: (...args: Parameters<typeof api.saveManualResult>) => saveManualResult(...args),
}));

vi.mock("./AnalysisControls", () => ({ default: () => null }));
vi.mock("./ManualReviewPanel", () => ({
  default: ({ onDirtyChange }: { onDirtyChange?: (dirty: boolean) => void }) => (
    <button type="button" onClick={() => onDirtyChange?.(true)}>Edit manual draft</button>
  ),
}));
vi.mock("./MLPredictionPanel", () => ({ default: () => null }));
vi.mock("./NMRPanel", () => ({ default: () => null }));
vi.mock("./QualityPanel", () => ({ default: () => null }));
vi.mock("./SpectrumViewer", () => ({
  default: ({
    spectrum,
    result,
  }: {
    spectrum: SpectrumData;
    result: unknown;
  }) => (
    <div
      data-testid="spectrum-viewer"
      data-first-x={String(spectrum.x_data[0])}
      data-has-result={String(Boolean(result))}
    />
  ),
}));
vi.mock("./ConfirmDialog", () => ({
  default: ({
    open,
    onCancel,
    onConfirm,
    title,
    confirmLabel,
  }: {
    open: boolean;
    onCancel: () => void;
    onConfirm: () => void;
    title: string;
    confirmLabel: string;
  }) => open
    ? (
      <div role="alertdialog">
        <span>{title}</span>
        <button type="button" onClick={onCancel}>Cancel preview</button>
        <button type="button" onClick={onConfirm}>{confirmLabel}</button>
      </div>
    )
    : null,
}));

const spectrum: SpectrumData = {
  id: "nmr-1",
  spectrum_revision: 4,
  technique: "NMR",
  x_data: [8, 7, 6],
  y_data: [0, 1, 0],
  x_label: "Chemical shift",
  y_label: "Intensity",
  x_unit: "ppm",
  y_unit: "a.u.",
  parameters: {
    nucleus: "1H",
    quadrature_available: true,
  },
  metadata: {},
  peaks: [],
  source_file: "sample.jdf",
};

const defaultResult: AnalysisResult = {
  technique: "NMR",
  result_revision: 3,
  peaks: [],
  metrics: {},
  summary: "reviewed",
};

function renderWorkbench(
  result: AnalysisResult | null = defaultResult,
  callbacks: {
    onSpectrumChanged?: (next: SpectrumData) => void;
    onResultChanged?: (next: AnalysisResult | null) => void;
    onError?: (message: string) => void;
    onDirtyChange?: (dirty: boolean) => void;
  } = {},
  selectedSpectrum = spectrum,
) {
  const onSpectrumChanged = callbacks.onSpectrumChanged ?? vi.fn<(next: SpectrumData) => void>();
  const onResultChanged = callbacks.onResultChanged ?? vi.fn<(next: AnalysisResult | null) => void>();
  const onError = callbacks.onError ?? vi.fn<(message: string) => void>();
  const onDirtyChange = callbacks.onDirtyChange ?? vi.fn<(dirty: boolean) => void>();
  const renderSource = (nextSpectrum: SpectrumData, nextResult: AnalysisResult | null) => (
    <StrictMode>
      <LangProvider>
        <NMRWorkbench
          spectrum={nextSpectrum}
          result={nextResult}
          spectra={[]}
          onSpectrumChanged={onSpectrumChanged}
          onResultChanged={onResultChanged}
          onError={onError}
          onDirtyChange={onDirtyChange}
        />
      </LangProvider>
    </StrictMode>
  );
  const view = render(renderSource(selectedSpectrum, result));
  return {
    ...view,
    updateSource: (nextSpectrum: SpectrumData, nextResult = result) => {
      view.rerender(renderSource(nextSpectrum, nextResult));
    },
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((accept, decline) => {
    resolve = accept;
    reject = decline;
  });
  return { promise, resolve, reject };
}

describe("NMR processing payload numbers", () => {
  beforeEach(() => {
    localStorage.setItem("chemapp-lang", "en");
    processNmrSpectrum.mockReset();
    getNmrSpectrumView.mockReset();
    analyzeSpectrum.mockReset();
    resetNmrSpectrum.mockReset();
    saveManualResult.mockReset();
    downloadMarkdownReport.mockReset();
  });

  it("normalizes numeric form values and omits blank or invalid values", () => {
    expect(optionalFiniteNumber("7.26")).toBe(7.26);
    expect(optionalFiniteNumber(-12.5)).toBe(-12.5);
    expect(optionalFiniteNumber("")).toBeUndefined();
    expect(optionalFiniteNumber("   ")).toBeUndefined();
    expect(optionalFiniteNumber("not-a-number")).toBeUndefined();
  });

  it("binds empty-result preview, apply, and reset to the spectrum's result revision", async () => {
    const processed = { ...spectrum, result_revision: 7 };
    processNmrSpectrum.mockResolvedValue(processed);
    resetNmrSpectrum.mockResolvedValue(processed);
    renderWorkbench(null, {}, processed);
    fireEvent.click(screen.getByRole("button", { name: "Review & apply" }));
    fireEvent.click(await screen.findByRole("button", { name: "Apply processing" }));
    await waitFor(() => expect(processNmrSpectrum).toHaveBeenCalledTimes(2));
    for (const [, payload] of processNmrSpectrum.mock.calls) {
      expect(payload).toMatchObject({ expected_revision: 4, expected_result_revision: 7 });
    }
    await waitFor(() => expect(screen.getByRole("button", { name: "Restore original spectrum" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Restore original spectrum" }));
    fireEvent.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Restore original spectrum" }));
    await waitFor(() => expect(resetNmrSpectrum).toHaveBeenCalledExactlyOnceWith("nmr-1", 4, 7));
  });

  it.each(["apply", "reset"] as const)("preserves the shown result when %s rejects an unseen manual save", async (operation) => {
    const { ApiError } = await import("../services/api");
    const conflict = new ApiError("The analysis result changed", 409);
    processNmrSpectrum.mockResolvedValueOnce(spectrum).mockRejectedValueOnce(conflict);
    resetNmrSpectrum.mockRejectedValue(conflict);
    const onSpectrumChanged = vi.fn();
    const onResultChanged = vi.fn();
    const onError = vi.fn();
    renderWorkbench(defaultResult, { onSpectrumChanged, onResultChanged, onError });
    if (operation === "apply") {
      fireEvent.click(screen.getByRole("button", { name: "Review & apply" }));
      fireEvent.click(await screen.findByRole("button", { name: "Apply processing" }));
    } else {
      fireEvent.click(screen.getByRole("button", { name: "Restore original spectrum" }));
      fireEvent.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Restore original spectrum" }));
    }
    await waitFor(() => expect(onError).toHaveBeenLastCalledWith(expect.stringMatching(/changed|reload/i)));
    expect(onSpectrumChanged).not.toHaveBeenCalled();
    expect(onResultChanged).not.toHaveBeenCalled();
  });

  it("sends automatic phase/reference parameters without mutually exclusive manual values", async () => {
    processNmrSpectrum.mockResolvedValue({
      ...spectrum,
      x_data: [9, 8, 7],
      quality_before: {},
      quality_after: {},
      warnings: [],
    });
    renderWorkbench();

    fireEvent.change(screen.getByLabelText("Zero-order phase (deg)"), {
      target: { value: "12.5" },
    });
    fireEvent.change(screen.getByLabelText("First-order phase (deg)"), {
      target: { value: "-30" },
    });
    fireEvent.change(screen.getByLabelText("Phase pivot ppm"), {
      target: { value: "4.7" },
    });
    fireEvent.change(screen.getByLabelText("Current reference ppm"), {
      target: { value: "7.28" },
    });
    fireEvent.click(screen.getByLabelText("Automatic phase correction"));
    fireEvent.click(screen.getByLabelText("Automatic solvent referencing"));
    fireEvent.click(screen.getByRole("button", { name: "Review & apply" }));

    await waitFor(() => expect(processNmrSpectrum).toHaveBeenCalledTimes(1));
    const [id, payload] = processNmrSpectrum.mock.calls[0];
    expect(id).toBe("nmr-1");
    expect(payload).toMatchObject({
      auto_phase: true,
      auto_phase_first_order: true,
      auto_phase_max_first_deg: 720,
      auto_reference: true,
      reference_window_ppm: 0.12,
      reference_min_snr: 5,
      preview_only: true,
      replay_from_original: true,
      expected_revision: 4,
      expected_result_revision: 3,
    });
    expect(payload.phase_zero_deg).toBeUndefined();
    expect(payload.phase_first_deg).toBeUndefined();
    expect(payload.phase_pivot_ppm).toBeUndefined();
    expect(payload.reference_current_ppm).toBeUndefined();
    expect(payload.reference_target_ppm).toBeUndefined();
    expect(payload.reference_solvent).toBeUndefined();
  });

  it("shows preview QC, then discards the preview without changing the current view", async () => {
    processNmrSpectrum.mockResolvedValue({
      ...spectrum,
      x_data: [9, 8, 7],
      quality_before: {
        noise_sigma: 2,
        baseline_rms: 5,
        quality_flags: ["baseline_drift"],
      },
      quality_after: {
        noise_sigma: 1,
        baseline_rms: 0.5,
        quality_flags: [],
      },
      warnings: [],
    });
    renderWorkbench();

    expect(screen.getByTestId("spectrum-viewer")).toHaveAttribute("data-first-x", "8");
    expect(screen.getByTestId("spectrum-viewer")).toHaveAttribute("data-has-result", "true");
    fireEvent.click(screen.getByRole("button", { name: "Review & apply" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Preview" })).toHaveAttribute(
        "aria-pressed",
        "true",
      );
    });
    expect(screen.getByTestId("spectrum-viewer")).toHaveAttribute("data-first-x", "9");
    expect(screen.getByTestId("spectrum-viewer")).toHaveAttribute("data-has-result", "false");
    expect(screen.getByRole("table", { name: "Processing quality comparison" }))
      .toHaveTextContent("2.000");
    expect(screen.getByRole("table", { name: "Processing quality comparison" }))
      .toHaveTextContent("1.000");

    fireEvent.click(screen.getByRole("button", { name: "Cancel preview" }));
    expect(screen.queryByRole("button", { name: "Preview" })).not.toBeInTheDocument();
    expect(screen.queryByText("The processing preview is ready. Inspect the spectrum before committing it."))
      .not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Current" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByTestId("spectrum-viewer")).toHaveAttribute("data-first-x", "8");
    expect(screen.getByTestId("spectrum-viewer")).toHaveAttribute("data-has-result", "true");
  });

  it("loads the immutable original view on demand and hides current analysis overlays", async () => {
    getNmrSpectrumView.mockResolvedValue({
      ...spectrum,
      x_data: [10, 9, 8],
      view: "original",
      quality_metrics: {
        noise_sigma: 3,
        quality_flags: ["baseline_drift"],
      },
    });
    renderWorkbench();

    fireEvent.click(screen.getByRole("button", { name: "Original" }));

    await waitFor(() => expect(getNmrSpectrumView).toHaveBeenCalledWith("nmr-1", "original", expect.any(AbortSignal)));
    expect(screen.getByRole("button", { name: "Original" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByTestId("spectrum-viewer")).toHaveAttribute("data-first-x", "10");
    expect(screen.getByTestId("spectrum-viewer")).toHaveAttribute("data-has-result", "false");
    expect(screen.getByText(/immutable original spectrum is shown/i)).toBeInTheDocument();
    expect(screen.getByText("Baseline drift was detected")).toBeInTheDocument();
  });

  it("blocks tab switching while processing changes are unsaved", () => {
    renderWorkbench();

    fireEvent.change(screen.getByLabelText("Zero-order phase (deg)"), {
      target: { value: "1.0" },
    });
    fireEvent.click(
      screen.getByRole("tab", { name: "Integrals / peak review" }),
    );

    expect(screen.getByRole("button", { name: "Cancel preview" })).toBeInTheDocument();
    expect(
      screen.getByRole("tab", { name: "Integrals / peak review" }),
    ).toHaveAttribute("aria-selected", "false");
  });

  it("requires explicit discard before analysis and sends the frozen result revision", async () => {
    analyzeSpectrum.mockResolvedValue({ ...defaultResult, result_revision: 4 });
    renderWorkbench();

    fireEvent.change(screen.getByLabelText("Zero-order phase (deg)"), {
      target: { value: "2.0" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Analyze / repick peaks" }));

    expect(analyzeSpectrum).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Discard changes" }));
    await waitFor(() => expect(analyzeSpectrum).toHaveBeenCalledWith(
      "nmr-1",
      { auto_reference: false },
      expect.any(AbortSignal),
      3,
      false,
    ));
  });

  it("keeps processing edits dirty when preview application is rejected", async () => {
    const onError = vi.fn();
    processNmrSpectrum
      .mockResolvedValueOnce({
        ...spectrum,
        x_data: [9, 8, 7],
        quality_before: {},
        quality_after: {},
        warnings: [],
      })
      .mockRejectedValueOnce(new Error("revision rejected"));
    renderWorkbench(defaultResult, { onError });

    fireEvent.change(screen.getByLabelText("Zero-order phase (deg)"), {
      target: { value: "4.0" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Review & apply" }));
    fireEvent.click(await screen.findByRole("button", { name: "Apply processing" }));

    await waitFor(() => expect(onError).toHaveBeenLastCalledWith("revision rejected"));
    fireEvent.click(screen.getByRole("tab", { name: "Integrals / peak review" }));
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Discard unsaved changes?");
    expect(screen.getByRole("tab", { name: "Integrals / peak review" }))
      .toHaveAttribute("aria-selected", "false");
  });

  it("does not clear a processing draft when discard confirmation is cancelled", () => {
    renderWorkbench();

    fireEvent.change(screen.getByLabelText("Zero-order phase (deg)"), {
      target: { value: "5.0" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Analyze / repick peaks" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel preview" }));
    fireEvent.click(screen.getByRole("button", { name: "Analyze / repick peaks" }));

    expect(analyzeSpectrum).not.toHaveBeenCalled();
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Discard unsaved changes?");
  });

  it("chains draft discard and reviewed-result overwrite confirmations", async () => {
    analyzeSpectrum.mockResolvedValue({ ...defaultResult, result_revision: 4 });
    renderWorkbench({
      ...defaultResult,
      metrics: { manual_confirmed: true },
    });

    fireEvent.click(screen.getByRole("tab", { name: "Integrals / peak review" }));
    fireEvent.click(screen.getByRole("button", { name: "Edit manual draft" }));
    fireEvent.click(screen.getByRole("button", { name: "Analyze / repick peaks" }));
    fireEvent.click(screen.getByRole("button", { name: "Discard changes" }));

    expect(analyzeSpectrum).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Overwrite and reanalyze" }));
    await waitFor(() => expect(analyzeSpectrum).toHaveBeenCalledWith(
      "nmr-1",
      { auto_reference: false },
      expect.any(AbortSignal),
      3,
      true,
    ));
  });

  it("tracks multiplet text as dirty and binds overwrite to its original revision", async () => {
    analyzeSpectrum.mockResolvedValue({ ...defaultResult, result_revision: 4 });
    renderWorkbench({
      ...defaultResult,
      metrics: { manual_confirmed: true },
    });

    fireEvent.click(screen.getByRole("tab", { name: "Multiplets" }));
    fireEvent.change(screen.getByLabelText("Manual multiplet ranges"), {
      target: { value: "4.10..3.96" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Rebuild multiplets/ }));

    expect(analyzeSpectrum).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Overwrite and reanalyze" }));
    await waitFor(() => expect(analyzeSpectrum).toHaveBeenCalledWith(
      "nmr-1",
      expect.objectContaining({
        multiplet_ranges: [{ start: 4.1, end: 3.96 }],
      }),
      expect.any(AbortSignal),
      3,
      true,
    ));
  });

  it("retains a multiplet draft after the rebuild fails", async () => {
    analyzeSpectrum.mockRejectedValue(new Error("rebuild rejected"));
    renderWorkbench();

    fireEvent.click(screen.getByRole("tab", { name: "Multiplets" }));
    fireEvent.change(screen.getByLabelText("Manual multiplet ranges"), {
      target: { value: "4.10..3.96" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Rebuild multiplets/ }));
    await waitFor(() => expect(analyzeSpectrum).toHaveBeenCalledWith(
      "nmr-1",
      expect.objectContaining({ multiplet_ranges: [{ start: 4.1, end: 3.96 }] }),
      expect.any(AbortSignal),
      3,
      false,
    ));

    fireEvent.click(screen.getByRole("button", { name: "Analyze / repick peaks" }));
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Discard unsaved changes?");
    expect(screen.getByLabelText("Manual multiplet ranges")).toHaveValue("4.10..3.96");
  });

  it("resets revision-bound multiplet drafts when a new result revision arrives", async () => {
    const onDirtyChange = vi.fn();
    const view = renderWorkbench(defaultResult, { onDirtyChange });

    fireEvent.click(screen.getByRole("tab", { name: "Multiplets" }));
    fireEvent.change(screen.getByLabelText("Manual multiplet ranges"), {
      target: { value: "4.10..3.96" },
    });
    await waitFor(() => expect(onDirtyChange).toHaveBeenLastCalledWith(true));

    view.updateSource(spectrum, { ...defaultResult, result_revision: 4 });

    await waitFor(() => expect(screen.getByLabelText("Manual multiplet ranges")).toHaveValue(""));
    await waitFor(() => expect(onDirtyChange).toHaveBeenLastCalledWith(false));
  });
});

describe("NMR workbench keyboard tabs", () => {
  beforeEach(() => {
    localStorage.setItem("chemapp-lang", "en");
    vi.resetAllMocks();
  });

  it("moves focus with arrows, Home, and End while leaving activation and the Tab entry point unchanged", async () => {
    const user = userEvent.setup();
    renderWorkbench();
    const [process, review, multiplets, report] = screen.getAllByRole("tab");

    await user.click(screen.getByRole("button", { name: "Current" }));
    await user.tab();
    expect(process).toHaveFocus();
    expect(process).toHaveAttribute("tabindex", "0");
    for (const tab of [review, multiplets, report]) {
      expect(tab).toHaveAttribute("tabindex", "-1");
    }

    await user.keyboard("{ArrowRight}");
    expect(review).toHaveFocus();
    await user.keyboard("{ArrowRight}");
    expect(multiplets).toHaveFocus();
    await user.keyboard("{ArrowLeft}");
    expect(review).toHaveFocus();
    await user.keyboard("{End}");
    expect(report).toHaveFocus();
    await user.keyboard("{ArrowRight}");
    expect(process).toHaveFocus();
    await user.keyboard("{ArrowLeft}");
    expect(report).toHaveFocus();
    await user.keyboard("{Home}");
    expect(process).toHaveFocus();
    await user.keyboard("{ArrowRight}");
    expect(review).toHaveFocus();

    expect(process).toHaveAttribute("aria-selected", "true");
    expect(review).toHaveAttribute("aria-selected", "false");
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    await user.tab();
    expect(screen.getByRole("tabpanel", { name: "Process" })).toHaveFocus();
    await user.tab({ shift: true });
    expect(process).toHaveFocus();
  });

  it.each(["{Enter}", " "])("activates the focused tab with %s and connects each panel to its tab", async (key) => {
    const user = userEvent.setup();
    renderWorkbench();
    const tabs = screen.getAllByRole("tab");
    const [process, review] = tabs;
    for (const tab of tabs) {
      const panel = document.getElementById(tab.getAttribute("aria-controls") || "");
      expect(panel).toHaveAttribute("role", "tabpanel");
      expect(panel).toHaveAttribute("aria-labelledby", tab.id);
      expect(panel).toHaveProperty("hidden", tab !== process);
    }

    await user.click(process);
    await user.keyboard("{ArrowRight}");
    await user.keyboard(key);

    expect(review).toHaveFocus();
    expect(review).toHaveAttribute("aria-selected", "true");
    expect(review).toHaveAttribute("tabindex", "0");
    expect(process).toHaveAttribute("tabindex", "-1");
    const panel = screen.getByRole("tabpanel", { name: "Integrals / peak review" });
    expect(panel.id).toBe(review.getAttribute("aria-controls"));
    expect(panel).toHaveAttribute("aria-labelledby", review.id);
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(screen.getAllByRole("tabpanel", { hidden: true })).toHaveLength(4);
    expect(within(panel).getByRole("button", { name: "Edit manual draft" })).toBeInTheDocument();
    await user.tab();
    expect(panel).toHaveFocus();
    await user.tab({ shift: true });
    expect(review).toHaveFocus();
  });

  it("skips result-dependent tabs when there is no analysis", async () => {
    const user = userEvent.setup();
    renderWorkbench(null);
    const [process, ...disabledTabs] = screen.getAllByRole("tab");
    await user.click(process);
    for (const key of ["{ArrowRight}", "{ArrowLeft}", "{End}", "{Home}"]) {
      await user.keyboard(key);
      expect(process).toHaveFocus();
      expect(process).toHaveAttribute("aria-selected", "true");
    }
    for (const tab of disabledTabs) {
      expect(tab).toBeDisabled();
      expect(tab).toHaveAttribute("tabindex", "-1");
      expect(tab).toHaveAttribute("aria-selected", "false");
    }
    await user.tab();
    expect(screen.getByRole("tabpanel", { name: "Process" })).toHaveFocus();
  });

  it("keeps an enabled tab and panel available after the selected result is removed", async () => {
    const user = userEvent.setup();
    const view = renderWorkbench();
    await user.click(screen.getByRole("tab", { name: "Integrals / peak review" }));

    view.updateSource(spectrum, null);

    const process = screen.getByRole("tab", { name: "Process" });
    expect(process).toHaveAttribute("aria-selected", "true");
    expect(process).toHaveAttribute("tabindex", "0");
    await user.click(screen.getByRole("button", { name: "Current" }));
    await user.tab();
    expect(process).toHaveFocus();
    expect(screen.getByRole("tabpanel", { name: "Process" })).toBeInTheDocument();
  });

  it("preserves a dirty draft through keyboard navigation and cancellation until discard is confirmed", async () => {
    const user = userEvent.setup();
    const onDirtyChange = vi.fn();
    renderWorkbench(defaultResult, { onDirtyChange });
    const process = screen.getByRole("tab", { name: "Process" });
    const review = screen.getByRole("tab", { name: "Integrals / peak review" });
    const reference = screen.getByLabelText("Current reference ppm");
    await user.type(reference, "7.28");
    await user.click(process);
    await user.keyboard("{End}{Home}{ArrowRight}");

    expect(review).toHaveFocus();
    expect(reference).toHaveValue("7.28");
    expect(onDirtyChange).toHaveBeenLastCalledWith(true);
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    await user.keyboard("{Enter}");
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Discard unsaved changes?");
    expect(process).toHaveAttribute("aria-selected", "true");
    expect(reference).toHaveValue("7.28");

    await user.click(screen.getByRole("button", { name: "Cancel preview" }));
    expect(reference).toHaveValue("7.28");
    expect(onDirtyChange).toHaveBeenLastCalledWith(true);
    await user.click(process);
    await user.keyboard("{ArrowRight} ");
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Discard unsaved changes?");
    expect(reference).toHaveValue("7.28");
    await user.click(screen.getByRole("button", { name: "Discard changes" }));

    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(review).toHaveAttribute("aria-selected", "true");
    expect(review).toHaveAttribute("tabindex", "0");
    expect(onDirtyChange).toHaveBeenLastCalledWith(false);
    await user.click(process);
    expect(screen.getByLabelText("Current reference ppm")).toHaveValue("");
    expect(analyzeSpectrum).not.toHaveBeenCalled();
    expect(processNmrSpectrum).not.toHaveBeenCalled();
  });

  it.each(["review", "multiplets"] as const)("keeps a dirty %s draft until keyboard activation is confirmed", async (tab) => {
    const user = userEvent.setup();
    const onDirtyChange = vi.fn();
    renderWorkbench(defaultResult, { onDirtyChange });
    const source = screen.getByRole("tab", { name: tab === "review" ? "Integrals / peak review" : "Multiplets" });
    const process = screen.getByRole("tab", { name: "Process" });
    await user.click(source);
    if (tab === "review") {
      await user.click(screen.getByRole("button", { name: "Edit manual draft" }));
    } else {
      await user.type(screen.getByLabelText("Manual multiplet ranges"), "4.1..3.9");
    }
    await user.click(source);
    await user.keyboard("{Home}");

    expect(process).toHaveFocus();
    expect(source).toHaveAttribute("aria-selected", "true");
    expect(onDirtyChange).toHaveBeenLastCalledWith(true);
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    await user.keyboard(" ");
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Discard unsaved changes?");
    await user.click(screen.getByRole("button", { name: "Cancel preview" }));
    expect(source).toHaveAttribute("aria-selected", "true");
    expect(onDirtyChange).toHaveBeenLastCalledWith(true);
    if (tab === "multiplets") {
      expect(screen.getByLabelText("Manual multiplet ranges")).toHaveValue("4.1..3.9");
    }

    await user.click(source);
    await user.keyboard("{Home}{Enter}");
    await user.click(screen.getByRole("button", { name: "Discard changes" }));
    expect(process).toHaveAttribute("aria-selected", "true");
    expect(onDirtyChange).toHaveBeenLastCalledWith(false);
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(analyzeSpectrum).not.toHaveBeenCalled();
    expect(saveManualResult).not.toHaveBeenCalled();
    await user.click(source);
    if (tab === "multiplets") {
      expect(screen.getByLabelText("Manual multiplet ranges")).toHaveValue("");
    }
    // The discarded editor must not leave a stale dirty flag on a later visit.
    await user.keyboard("{Home}{Enter}");
    expect(process).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  });

  it("does not consume modified navigation keys or vertical arrows", () => {
    renderWorkbench();
    const process = screen.getByRole("tab", { name: "Process" });
    process.focus();
    for (const modifier of ["altKey", "ctrlKey", "metaKey"]) {
      for (const key of ["ArrowLeft", "ArrowRight", "Home", "End"]) {
        expect(fireEvent.keyDown(process, { key, [modifier]: true })).toBe(true);
        expect(process).toHaveFocus();
      }
    }
    for (const key of ["ArrowUp", "ArrowDown"]) {
      expect(fireEvent.keyDown(process, { key })).toBe(true);
      expect(process).toHaveFocus();
    }
  });

  it("locks keyboard focus and activation as soon as a mutation starts, before disabled state renders", async () => {
    const { pending, response } = preparePendingOperation("analysis");
    const user = userEvent.setup();
    renderWorkbench();
    const [process, review] = screen.getAllByRole("tab");
    const analyze = screen.getByRole("button", { name: "Analyze / repick peaks" });
    process.focus();

    act(() => {
      analyze.click();
      expect(process).toBeEnabled();
      process.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true, cancelable: true }));
      review.click();
    });

    expect(process).toHaveFocus();
    expect(process).toHaveAttribute("aria-selected", "true");
    expect(review).toHaveAttribute("aria-selected", "false");
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    for (const tab of screen.getAllByRole("tab")) expect(tab).toBeDisabled();
    for (const key of ["{ArrowRight}", "{ArrowLeft}", "{Home}", "{End}", "{Enter}", " "]) {
      await user.keyboard(key);
      expect(process).toHaveFocus();
      expect(process).toHaveAttribute("aria-selected", "true");
    }
    expect(analyzeSpectrum).toHaveBeenCalledTimes(1);

    await act(async () => {
      pending.resolve(response);
      await pending.promise;
    });
    expect(process).toBeEnabled();
    await user.keyboard("{ArrowRight}{Enter}");
    expect(review).toHaveFocus();
    expect(review).toHaveAttribute("aria-selected", "true");
  });
});

const lifecycleOperations = ["preview", "apply", "analysis", "reset", "original", "rebuild", "save", "export"] as const;
type LifecycleOperation = typeof lifecycleOperations[number];
const resultWithMultiplets: AnalysisResult = {
  ...defaultResult,
  multiplets: [{
    center_ppm: 4,
    range_ppm: [4.1, 3.9],
    component_positions: [4],
    n_peaks: 1,
    n_components: 1,
    estimated_j_hz: null,
    intensity_max: 1,
  }],
};

function preparePendingOperation(operation: LifecycleOperation) {
  // These fields satisfy both spectrum and result response contracts, allowing
  // the same deferred lifecycle matrix to exercise every request path.
  const response = { ...spectrum, ...resultWithMultiplets, spectrum_revision: 5, result_revision: 4 };
  const pending = deferred<typeof response>();
  processNmrSpectrum.mockReturnValue(pending.promise);
  getNmrSpectrumView.mockReturnValue(pending.promise);
  analyzeSpectrum.mockReturnValue(pending.promise);
  resetNmrSpectrum.mockReturnValue(pending.promise);
  saveManualResult.mockReturnValue(pending.promise);
  downloadMarkdownReport.mockImplementation(() => pending.promise.then(() => undefined));
  if (operation === "apply") processNmrSpectrum.mockResolvedValueOnce(spectrum);
  return { pending, response };
}

async function startOperation(operation: LifecycleOperation) {
  switch (operation) {
    case "preview":
    case "apply":
      fireEvent.click(screen.getByRole("button", { name: "Review & apply" }));
      if (operation === "apply") {
        fireEvent.click(await screen.findByRole("button", { name: "Apply processing" }));
      }
      break;
    case "analysis":
      fireEvent.click(screen.getByRole("button", { name: "Analyze / repick peaks" }));
      break;
    case "reset":
      fireEvent.click(screen.getByRole("button", { name: "Restore original spectrum" }));
      fireEvent.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Restore original spectrum" }));
      break;
    case "original":
      fireEvent.click(screen.getByRole("button", { name: "Original" }));
      break;
    case "rebuild":
    case "save":
      fireEvent.click(screen.getByRole("tab", { name: "Multiplets" }));
      if (operation === "rebuild") {
        fireEvent.change(screen.getByLabelText("Manual multiplet ranges"), { target: { value: "4.1..3.9" } });
        fireEvent.click(screen.getByRole("button", { name: /Rebuild multiplets/ }));
      } else {
        fireEvent.click(screen.getByRole("button", { name: "Save reviewed multiplets" }));
      }
      break;
    case "export":
      fireEvent.click(screen.getByRole("button", { name: "Export report" }));
      break;
  }
}

describe("NMR request ownership", () => {
  beforeEach(() => {
    localStorage.setItem("chemapp-lang", "en");
    processNmrSpectrum.mockReset();
    getNmrSpectrumView.mockReset();
    analyzeSpectrum.mockReset();
    resetNmrSpectrum.mockReset();
    saveManualResult.mockReset();
    downloadMarkdownReport.mockReset();
  });

  describe.each(lifecycleOperations)("%s", (operation) => {
    it.each(["success", "failure"] as const)("ignores late %s callbacks after unmount", async (outcome) => {
      const { pending, response } = preparePendingOperation(operation);
      const onSpectrumChanged = vi.fn();
      const onResultChanged = vi.fn();
      const onError = vi.fn();
      const view = renderWorkbench(resultWithMultiplets, { onSpectrumChanged, onResultChanged, onError });
      await startOperation(operation);
      onError.mockClear();
      view.unmount();

      if (operation === "analysis" || operation === "rebuild") {
        expect(analyzeSpectrum.mock.calls[0][2]?.aborted).toBe(true);
      }
      if (operation === "original") {
        expect(getNmrSpectrumView.mock.calls[0][2]?.aborted).toBe(true);
      }
      await act(async () => {
        if (outcome === "success") pending.resolve(response);
        else pending.reject(new Error("late request failed"));
        await pending.promise.catch(() => undefined);
      });

      expect(onSpectrumChanged).not.toHaveBeenCalled();
      expect(onResultChanged).not.toHaveBeenCalled();
      expect(onError).not.toHaveBeenCalled();
    });

    it("publishes a still-current success and releases its controls", async () => {
      const { pending, response } = preparePendingOperation(operation);
      const onSpectrumChanged = vi.fn();
      const onResultChanged = vi.fn();
      const onError = vi.fn();
      renderWorkbench(resultWithMultiplets, { onSpectrumChanged, onResultChanged, onError });
      await startOperation(operation);
      onError.mockClear();

      await act(async () => {
        pending.resolve(response);
        await pending.promise;
      });

      if (operation === "apply" || operation === "reset") {
        expect(onSpectrumChanged).toHaveBeenCalledExactlyOnceWith(response);
        expect(onResultChanged).toHaveBeenCalledExactlyOnceWith(null);
      } else if (operation === "analysis" || operation === "rebuild" || operation === "save") {
        expect(onResultChanged).toHaveBeenCalledExactlyOnceWith(response);
        expect(onSpectrumChanged).not.toHaveBeenCalled();
      } else if (operation === "preview" || operation === "original") {
        expect(screen.getByRole("button", { name: operation === "preview" ? "Preview" : "Original" }))
          .toHaveAttribute("aria-pressed", "true");
      }
      expect(onError).not.toHaveBeenCalled();
      expect(screen.getByRole("button", { name: "Analyze / repick peaks" })).toBeEnabled();
      expect(screen.getByRole("button", { name: "Export report" })).toBeEnabled();
    });

    it("reports a current failure and allows retry", async () => {
      const { pending } = preparePendingOperation(operation);
      const onError = vi.fn();
      renderWorkbench(resultWithMultiplets, { onError });
      await startOperation(operation);
      onError.mockClear();

      await act(async () => {
        pending.reject(new Error("current request failed"));
        await pending.promise.catch(() => undefined);
      });

      expect(onError).toHaveBeenCalledExactlyOnceWith("current request failed");
      expect(screen.getByRole("button", { name: "Analyze / repick peaks" })).toBeEnabled();
      expect(screen.getByRole("button", { name: "Export report" })).toBeEnabled();
    });
  });

  it.each(["selection", "spectrum revision", "result revision"] as const)(
    "does not let a prior %s response or finally replace newer work",
    async (change) => {
      const oldRequest = deferred<AnalysisResult>();
      const newRequest = deferred<AnalysisResult>();
      analyzeSpectrum.mockReturnValueOnce(oldRequest.promise).mockReturnValueOnce(newRequest.promise);
      const onResultChanged = vi.fn();
      const onError = vi.fn();
      const view = renderWorkbench(defaultResult, { onResultChanged, onError });
      await startOperation("analysis");
      const oldSignal = analyzeSpectrum.mock.calls[0][2];
      const nextSpectrum = change === "selection"
        ? { ...spectrum, id: "nmr-2", x_data: [11, 10, 9] }
        : change === "spectrum revision" ? { ...spectrum, spectrum_revision: 5 } : spectrum;
      const nextResult = change === "result revision" ? { ...defaultResult, result_revision: 5 } : defaultResult;
      view.updateSource(nextSpectrum, nextResult);
      expect(oldSignal?.aborted).toBe(true);
      await startOperation("analysis");
      onError.mockClear();
      const newSignal = analyzeSpectrum.mock.calls[1][2];

      await act(async () => {
        oldRequest.resolve({ ...defaultResult, summary: "stale" });
        await oldRequest.promise;
      });

      expect(onResultChanged).not.toHaveBeenCalled();
      expect(onError).not.toHaveBeenCalled();
      expect(screen.getByRole("button", { name: "Working…" })).toBeDisabled();
      expect(newSignal?.aborted).toBe(false);
      const updated = { ...nextResult, result_revision: 6, summary: "current" };
      await act(async () => {
        newRequest.resolve(updated);
        await newRequest.promise;
      });
      expect(onResultChanged).toHaveBeenCalledExactlyOnceWith(updated);
      expect(screen.getByRole("button", { name: "Analyze / repick peaks" })).toBeEnabled();
    },
  );

  it("ignores an old error without clearing the replacement request's busy state", async () => {
    const oldRequest = deferred<AnalysisResult>();
    const newRequest = deferred<AnalysisResult>();
    analyzeSpectrum.mockReturnValueOnce(oldRequest.promise).mockReturnValueOnce(newRequest.promise);
    const onError = vi.fn();
    const view = renderWorkbench(defaultResult, { onError });
    await startOperation("analysis");
    view.updateSource(spectrum, { ...defaultResult, result_revision: 4 });
    await startOperation("analysis");
    onError.mockClear();
    await act(async () => {
      oldRequest.reject(new Error("stale failure"));
      await oldRequest.promise.catch(() => undefined);
    });
    expect(onError).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Working…" })).toBeDisabled();
    await act(async () => {
      newRequest.resolve(defaultResult);
      await newRequest.promise;
    });
  });

  it("invalidates a pending preview when inputs change without releasing a newer preview", async () => {
    const oldRequest = deferred<SpectrumData>();
    const newRequest = deferred<SpectrumData>();
    processNmrSpectrum.mockReturnValueOnce(oldRequest.promise).mockReturnValueOnce(newRequest.promise);
    renderWorkbench();
    await startOperation("preview");
    fireEvent.change(screen.getByLabelText("Zero-order phase (deg)"), { target: { value: "15" } });
    expect(screen.getByRole("button", { name: "Review & apply" })).toBeEnabled();
    await startOperation("preview");
    await act(async () => {
      oldRequest.resolve({ ...spectrum, x_data: [20, 19, 18] });
      await oldRequest.promise;
    });
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(screen.getByTestId("spectrum-viewer")).toHaveAttribute("data-first-x", "8");
    expect(screen.getByRole("button", { name: "Review & apply" })).toBeDisabled();
    expect(processNmrSpectrum.mock.calls[1][1]).toMatchObject({ phase_zero_deg: 15, expected_revision: 4 });
    await act(async () => {
      newRequest.resolve({ ...spectrum, x_data: [10, 9, 8] });
      await newRequest.promise;
    });
    expect(screen.getByTestId("spectrum-viewer")).toHaveAttribute("data-first-x", "10");
    expect(screen.getByRole("button", { name: "Apply processing" })).toBeInTheDocument();
  });

  it("requires a fresh preview after editing previously reviewed inputs", async () => {
    processNmrSpectrum.mockResolvedValue(spectrum);
    renderWorkbench();
    await startOperation("preview");
    await screen.findByRole("button", { name: "Apply processing" });
    fireEvent.change(screen.getByLabelText("Zero-order phase (deg)"), { target: { value: "15" } });
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Preview" })).not.toBeInTheDocument();
    expect(processNmrSpectrum).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Review & apply" }));
    await screen.findByRole("button", { name: "Apply processing" });
    expect(processNmrSpectrum.mock.calls[1][1]).toMatchObject({ phase_zero_deg: 15, preview_only: true });
  });

  it.each(["spectrum", "result"] as const)("withdraws preview approval when the %s revision changes", async (source) => {
    processNmrSpectrum.mockResolvedValue(spectrum);
    const view = renderWorkbench();
    await startOperation("preview");
    await screen.findByRole("button", { name: "Apply processing" });
    view.updateSource(
      source === "spectrum" ? { ...spectrum, spectrum_revision: 5 } : spectrum,
      source === "result" ? { ...defaultResult, result_revision: 4 } : defaultResult,
    );
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Preview" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Current" })).toHaveAttribute("aria-pressed", "true");
    expect(processNmrSpectrum).toHaveBeenCalledTimes(1);
  });

  it("withdraws stale navigation confirmation without discarding unrelated processing edits", () => {
    const view = renderWorkbench();
    fireEvent.change(screen.getByLabelText("Zero-order phase (deg)"), { target: { value: "15" } });
    fireEvent.click(screen.getByRole("tab", { name: "Multiplets" }));
    expect(screen.getByRole("alertdialog")).toBeInTheDocument();
    view.updateSource(spectrum, { ...defaultResult, result_revision: 4 });
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Process" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByLabelText("Zero-order phase (deg)")).toHaveValue(15);
  });

  it.each(["success", "failure"] as const)("keeps Current selected after canceling an original read that ends in %s", async (outcome) => {
    const { pending, response } = preparePendingOperation("original");
    const onError = vi.fn();
    renderWorkbench(defaultResult, { onError });
    await startOperation("original");
    fireEvent.click(screen.getByRole("button", { name: "Current" }));
    expect(getNmrSpectrumView.mock.calls[0][2]?.aborted).toBe(true);
    onError.mockClear();
    await act(async () => {
      if (outcome === "success") pending.resolve({ ...response, x_data: [20, 19, 18] });
      else pending.reject(new Error("canceled read failed"));
      await pending.promise.catch(() => undefined);
    });
    expect(onError).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Current" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("spectrum-viewer")).toHaveAttribute("data-first-x", "8");
    expect(screen.getByRole("button", { name: "Analyze / repick peaks" })).toBeEnabled();
  });

  it.each(["analysis", "export"] as const)("rejects duplicate %s actions before disabled state renders", async (operation) => {
    const { pending, response } = preparePendingOperation(operation);
    renderWorkbench();
    const button = screen.getByRole("button", { name: operation === "analysis" ? "Analyze / repick peaks" : "Export report" });
    act(() => {
      button.click();
      button.click();
    });
    expect(operation === "analysis" ? analyzeSpectrum : downloadMarkdownReport).toHaveBeenCalledTimes(1);
    await act(async () => {
      pending.resolve(response);
      await pending.promise;
    });
  });

  it.each(["apply", "reset", "rebuild", "save"] as const)("rejects duplicate %s mutations before disabled state renders", async (operation) => {
    const { pending, response } = preparePendingOperation(operation);
    renderWorkbench(resultWithMultiplets);
    let button: HTMLElement;
    if (operation === "apply") {
      fireEvent.click(screen.getByRole("button", { name: "Review & apply" }));
      button = await screen.findByRole("button", { name: "Apply processing" });
    } else if (operation === "reset") {
      fireEvent.click(screen.getByRole("button", { name: "Restore original spectrum" }));
      button = within(screen.getByRole("alertdialog")).getByRole("button", { name: "Restore original spectrum" });
    } else {
      fireEvent.click(screen.getByRole("tab", { name: "Multiplets" }));
      if (operation === "rebuild") {
        fireEvent.change(screen.getByLabelText("Manual multiplet ranges"), { target: { value: "4.1..3.9" } });
      }
      button = screen.getByRole("button", { name: operation === "rebuild" ? /Rebuild multiplets/ : "Save reviewed multiplets" });
    }
    act(() => {
      button.click();
      button.click();
    });
    if (operation === "apply") {
      expect(processNmrSpectrum).toHaveBeenCalledTimes(2);
      expect(processNmrSpectrum.mock.calls[1][1]).toMatchObject({ preview_only: false, expected_revision: 4, expected_result_revision: 3 });
    } else if (operation === "reset") {
      expect(resetNmrSpectrum).toHaveBeenCalledExactlyOnceWith("nmr-1", 4, 3);
    } else if (operation === "rebuild") {
      expect(analyzeSpectrum).toHaveBeenCalledExactlyOnceWith(
        "nmr-1", expect.objectContaining({ multiplet_ranges: [{ start: 4.1, end: 3.9 }] }), expect.any(AbortSignal), 3, false,
      );
    } else {
      expect(saveManualResult).toHaveBeenCalledExactlyOnceWith(
        "nmr-1", expect.objectContaining({ expected_revision: 3, multiplets: resultWithMultiplets.multiplets }),
      );
    }
    expect(screen.getByRole("tab", { name: "Process" })).toBeDisabled();
    await act(async () => {
      pending.resolve(response);
      await pending.promise;
    });
  });
});
