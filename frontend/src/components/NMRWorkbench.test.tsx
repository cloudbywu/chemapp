import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LangProvider } from "../i18n/LangContext";
import type { AnalysisResult, SpectrumData } from "../types/spectrum";
import NMRWorkbench, { optionalFiniteNumber } from "./NMRWorkbench";

const processNmrSpectrum = vi.fn();
const getNmrSpectrumView = vi.fn();
const analyzeSpectrum = vi.fn();

vi.mock("../services/api", () => ({
  analyzeSpectrum: (...args: unknown[]) => analyzeSpectrum(...args),
  ApiError: class ApiError extends Error {
    status: number;

    constructor(message: string, status: number) {
      super(message);
      this.status = status;
    }
  },
  downloadMarkdownReport: vi.fn(),
  getNmrSpectrumView: (...args: unknown[]) => getNmrSpectrumView(...args),
  processNmrSpectrum: (...args: unknown[]) => processNmrSpectrum(...args),
  resetNmrSpectrum: vi.fn(),
  saveManualResult: vi.fn(),
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
  result: AnalysisResult = defaultResult,
  callbacks: {
    onSpectrumChanged?: (next: SpectrumData) => void;
    onResultChanged?: (next: AnalysisResult | null) => void;
    onError?: (message: string) => void;
    onDirtyChange?: (dirty: boolean) => void;
  } = {},
) {
  const onSpectrumChanged = callbacks.onSpectrumChanged ?? vi.fn<(next: SpectrumData) => void>();
  const onResultChanged = callbacks.onResultChanged ?? vi.fn<(next: AnalysisResult | null) => void>();
  const onError = callbacks.onError ?? vi.fn<(message: string) => void>();
  const onDirtyChange = callbacks.onDirtyChange ?? vi.fn<(dirty: boolean) => void>();
  return render(
    <LangProvider>
      <NMRWorkbench
        spectrum={spectrum}
        result={result}
        spectra={[]}
        onSpectrumChanged={onSpectrumChanged}
        onResultChanged={onResultChanged}
        onError={onError}
        onDirtyChange={onDirtyChange}
      />
    </LangProvider>,
  );
}

describe("NMR processing payload numbers", () => {
  beforeEach(() => {
    localStorage.setItem("chemapp-lang", "en");
    processNmrSpectrum.mockReset();
    getNmrSpectrumView.mockReset();
    analyzeSpectrum.mockReset();
  });

  it("normalizes numeric form values and omits blank or invalid values", () => {
    expect(optionalFiniteNumber("7.26")).toBe(7.26);
    expect(optionalFiniteNumber(-12.5)).toBe(-12.5);
    expect(optionalFiniteNumber("")).toBeUndefined();
    expect(optionalFiniteNumber("   ")).toBeUndefined();
    expect(optionalFiniteNumber("not-a-number")).toBeUndefined();
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

    await waitFor(() => expect(getNmrSpectrumView).toHaveBeenCalledWith("nmr-1", "original"));
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
      undefined,
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
      undefined,
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
      undefined,
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
      undefined,
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

    view.rerender(
      <LangProvider>
        <NMRWorkbench
          spectrum={spectrum}
          result={{ ...defaultResult, result_revision: 4 }}
          spectra={[]}
          onSpectrumChanged={vi.fn()}
          onResultChanged={vi.fn()}
          onError={vi.fn()}
          onDirtyChange={onDirtyChange}
        />
      </LangProvider>,
    );

    await waitFor(() => expect(screen.getByLabelText("Manual multiplet ranges")).toHaveValue(""));
    await waitFor(() => expect(onDirtyChange).toHaveBeenLastCalledWith(false));
  });
});
