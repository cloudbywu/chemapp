import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type * as api from "./services/api";
import type { AnalysisResult, SpectrumData, SpectrumListItem } from "./types/spectrum";
import { LangProvider } from "./i18n/LangContext";

const listSpectra = vi.fn<typeof api.listSpectra>();
const getSpectrum = vi.fn<typeof api.getSpectrum>();
const getResult = vi.fn<typeof api.getResult>();
const analyzeSpectrum = vi.fn<typeof api.analyzeSpectrum>();
const deleteSpectrum = vi.fn<typeof api.deleteSpectrum>();
const listExamples = vi.fn<typeof api.listExamples>();
const loadExample = vi.fn<typeof api.loadExample>();
const uploadFile = vi.fn<typeof api.uploadFile>();

beforeEach(() => {
  listExamples.mockReset().mockResolvedValue([]);
  loadExample.mockReset();
  uploadFile.mockReset();
});

vi.mock("./services/api", () => ({
  listSpectra: (...args: Parameters<typeof api.listSpectra>) => listSpectra(...args),
  getSpectrum: (...args: Parameters<typeof api.getSpectrum>) => getSpectrum(...args),
  getResult: (...args: Parameters<typeof api.getResult>) => getResult(...args),
  analyzeSpectrum: (...args: Parameters<typeof api.analyzeSpectrum>) => analyzeSpectrum(...args),
  deleteSpectrum: (...args: Parameters<typeof api.deleteSpectrum>) => deleteSpectrum(...args),
  downloadMarkdownReport: vi.fn(),
  downloadSpectrumCsv: vi.fn(),
  listExamples: (...args: Parameters<typeof api.listExamples>) => listExamples(...args),
  loadExample: (...args: Parameters<typeof api.loadExample>) => loadExample(...args),
  uploadFile: (...args: Parameters<typeof api.uploadFile>) => uploadFile(...args),
  ApiError: class ApiError extends Error {
    status?: number;
  },
}));

vi.mock("./components/AIChatSidebar", () => ({
  default: ({ onSpectrumMutated }: { onSpectrumMutated: (id: string) => Promise<void> }) => (
    <button type="button" onClick={() => void onSpectrumMutated("A")}>Finish AI mutation A</button>
  ),
}));
vi.mock("./components/CompareView", () => ({ default: () => null }));
vi.mock("./components/InferencePanel", () => ({ default: () => null }));
vi.mock("./components/SettingsPanel", () => ({ default: () => null }));
vi.mock("./components/ModelAssetsPanel", () => ({ default: () => <p>Official model downloads</p> }));
vi.mock("./components/MLTrainingPanel", () => ({ default: () => null }));
vi.mock("./components/AnalysisControls", () => ({ default: () => null }));
vi.mock("./components/QualityPanel", () => ({ default: () => null }));
vi.mock("./components/ManualReviewPanel", () => ({
  default: ({ onDirtyChange }: { onDirtyChange?: (dirty: boolean) => void }) => (
    <div>
      <button type="button" onClick={() => onDirtyChange?.(true)}>Dirty manual review</button>
      <button type="button" onClick={() => onDirtyChange?.(false)}>Clean manual review</button>
    </div>
  ),
}));
vi.mock("./components/SpectrumViewer", () => ({ default: () => null }));
vi.mock("./components/AnalysisPanel", () => ({
  default: ({ result, onDirtyChange }: { result: AnalysisResult; onDirtyChange?: (dirty: boolean) => void }) => (
    <div>
      <p>{result.summary}</p>
      <button type="button" onClick={() => onDirtyChange?.(true)}>Dirty HPLC events</button>
      <button type="button" onClick={() => onDirtyChange?.(false)}>Clean HPLC events</button>
    </div>
  ),
}));
vi.mock("./components/MLPredictionPanel", () => ({ default: () => null }));
vi.mock("./components/NMRWorkbench", () => ({
  default: ({
    spectrum,
    onDirtyChange,
  }: {
    spectrum: { id: string };
    onDirtyChange?: (dirty: boolean) => void;
  }) => (
    <div>
      Workbench {spectrum.id}
      <button type="button" onClick={() => onDirtyChange?.(true)}>Dirty workbench</button>
      <button type="button" onClick={() => onDirtyChange?.(false)}>Clean workbench</button>
    </div>
  ),
}));
vi.mock("./components/SpectrumGoldReviewPanel", () => ({
  default: ({ onDirtyChange }: { onDirtyChange?: (dirty: boolean) => void }) => (
    <div>
      <button type="button" onClick={() => onDirtyChange?.(true)}>Dirty gold review</button>
      <button type="button" onClick={() => onDirtyChange?.(false)}>Clean gold review</button>
    </div>
  ),
  SpectrumReviewQueuePanel: () => null,
}));

describe("spectrum selection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem("chemapp-lang", "en");
    listSpectra.mockResolvedValue([{
      id: "spectrum-1",
      technique: "NMR",
      points: 1024,
      name: "Sample A",
      has_result: true,
      summary: "",
      spectrum_revision: 4,
      result_revision: 2,
    }]);
    getSpectrum.mockResolvedValue({
      id: "spectrum-1",
      technique: "NMR",
      x_data: [1, 0],
      y_data: [0, 1],
      x_label: "ppm",
      y_label: "Intensity",
      x_unit: "ppm",
      y_unit: "",
      parameters: {},
      metadata: {},
      peaks: [],
      source_file: "sample.jdf",
    });
    getResult.mockResolvedValue({
      technique: "NMR",
      result_revision: 2,
      peaks: [],
      metrics: {},
      summary: "Existing result",
    });
  });

  it("provides actionable empty-workspace shortcuts without starting analysis", async () => {
    listExamples.mockResolvedValue([{ technique: "NMR", label: "Demo spectrum", filename: "demo.txt", path: "demo.txt", size: 1 }]);
    const view = render(<LangProvider><App /></LangProvider>);
    await screen.findByRole("button", { name: /Demo spectrum/ });
    expect(screen.getByRole("heading", { name: "Start with a spectrum" })).toBeInTheDocument();
    const input = view.container.querySelector("input[type=file]")!;
    const chooser = vi.spyOn(input as HTMLInputElement, "click");
    fireEvent.click(screen.getByRole("button", { name: /Import data/ }));
    expect(chooser).toHaveBeenCalledTimes(1);
    const examples = view.container.querySelector(".example-loader") as HTMLDetailsElement;
    examples.open = false;
    fireEvent.click(screen.getByRole("button", { name: "Browse examples" }));
    expect(examples.open).toBe(true);
    expect(screen.getByRole("button", { name: /Demo spectrum/ })).toHaveFocus();
    expect(analyzeSpectrum).not.toHaveBeenCalled();
  });

  it("supports keyboard navigation without changing workspace until activated", async () => {
    const user = userEvent.setup();
    render(<LangProvider><App /></LangProvider>);
    const spectra = await screen.findByRole("button", { name: "Spectra" });
    const settings = screen.getByRole("button", { name: "Settings" });
    spectra.focus();
    await user.keyboard("{End}");
    expect(settings).toHaveFocus();
    expect(spectra).toHaveAttribute("aria-current", "page");
    await user.keyboard("{Enter}");
    expect(settings).toHaveAttribute("aria-current", "page");
    await user.keyboard("{ArrowRight}");
    expect(spectra).toHaveFocus();
    expect(settings).toHaveAttribute("aria-current", "page");
    await user.keyboard("{Home}");
    expect(spectra).toHaveFocus();
  });

  it("returns to the spectra workspace when a new or already loaded spectrum is selected", async () => {
    render(<LangProvider><App /></LangProvider>);
    const sample = await screen.findByRole("button", { name: "Select spectrum Sample A" });
    const settings = screen.getByRole("button", { name: "Settings" });
    fireEvent.click(settings);
    fireEvent.click(sample);
    expect(await screen.findByText("Workbench spectrum-1")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Spectra" })).toHaveAttribute("aria-current", "page");
    fireEvent.click(settings);
    fireEvent.click(sample);
    expect(await screen.findByText("Workbench spectrum-1")).toBeInTheDocument();
    expect(getSpectrum).toHaveBeenCalledTimes(1);
    expect(analyzeSpectrum).not.toHaveBeenCalled();
  });

  it("loads the existing result with GET and never starts analysis on selection", async () => {
    render(<LangProvider><App /></LangProvider>);

    fireEvent.click(await screen.findByRole("button", { name: "Select spectrum Sample A" }));
    expect(await screen.findByText("Workbench spectrum-1")).toBeInTheDocument();

    await waitFor(() => expect(getResult).toHaveBeenCalledWith("spectrum-1", expect.any(AbortSignal)));
    expect(analyzeSpectrum).not.toHaveBeenCalled();
  });

  it("supports Enter and Space on the native language button", async () => {
    const user = userEvent.setup();
    render(<LangProvider><App /></LangProvider>);

    const toChinese = await screen.findByRole("button", { name: "切换到中文" });
    toChinese.focus();
    await user.keyboard("{Enter}");
    expect(document.documentElement).toHaveAttribute("lang", "zh-CN");

    const toEnglish = screen.getByRole("button", { name: "Switch to English" });
    toEnglish.focus();
    await user.keyboard(" ");
    expect(document.documentElement).toHaveAttribute("lang", "en");
  });

  it("does not render the newly selected workbench until selection resolves", async () => {
    let resolveSpectrum!: (value: SpectrumData | PromiseLike<SpectrumData>) => void;
    getSpectrum.mockReturnValue(
      new Promise<SpectrumData>((resolve) => {
        resolveSpectrum = resolve;
      }),
    );
    render(<LangProvider><App /></LangProvider>);

    fireEvent.click(await screen.findByRole("button", { name: "Select spectrum Sample A" }));
    expect(screen.queryByText("Workbench spectrum-1")).not.toBeInTheDocument();

    resolveSpectrum({
      id: "spectrum-1",
      technique: "NMR",
      x_data: [1, 0],
      y_data: [0, 1],
      x_label: "ppm",
      y_label: "Intensity",
      x_unit: "ppm",
      y_unit: "",
      parameters: {},
      metadata: {},
      peaks: [],
      source_file: "sample.jdf",
    });
    expect(await screen.findByText("Workbench spectrum-1")).toBeInTheDocument();
  });

  it("allows retrying the same spectrum after a failed load", async () => {
    getSpectrum.mockRejectedValueOnce(new Error("temporary failure"));
    render(<LangProvider><App /></LangProvider>);

    const select = await screen.findByRole("button", { name: "Select spectrum Sample A" });
    fireEvent.click(select);
    expect(await screen.findByRole("alert")).toHaveTextContent("temporary failure");

    fireEvent.click(select);
    expect(await screen.findByText("Workbench spectrum-1")).toBeInTheDocument();
    expect(getSpectrum).toHaveBeenCalledTimes(2);
  });

  it("keeps navigation protected until every dirty source is clean", async () => {
    render(<LangProvider><App /></LangProvider>);
    fireEvent.click(await screen.findByRole("button", { name: "Select spectrum Sample A" }));
    expect(await screen.findByText("Workbench spectrum-1")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Dirty workbench" }));
    fireEvent.click(screen.getByRole("button", { name: "Dirty gold review" }));
    fireEvent.click(screen.getByRole("button", { name: "Clean workbench" }));
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));

    expect(screen.getByRole("alertdialog")).toHaveTextContent("Discard unsaved changes?");
  });

  it.each(["upload", "example"] as const)("protects edits made while a %s is pending", async (operation) => {
    const pending = deferred<SpectrumListItem[]>();
    listExamples.mockResolvedValue([{
      technique: "NMR", label: "Example spectrum", path: "example.txt", filename: "example.txt", size: 1,
    }]);
    uploadFile.mockReturnValue(pending.promise);
    loadExample.mockReturnValue(pending.promise);
    const view = render(<LangProvider><App /></LangProvider>);
    fireEvent.click(await screen.findByRole("button", { name: "Select spectrum Sample A" }));
    expect(await screen.findByText("Workbench spectrum-1")).toBeInTheDocument();

    if (operation === "upload") {
      fireEvent.change(view.container.querySelector("input[type=file]")!, {
        target: { files: [new File(["x"], "sample.txt")] },
      });
    } else fireEvent.click(await screen.findByRole("button", { name: /Example spectrum/ }));
    fireEvent.click(screen.getByRole("button", { name: "Dirty workbench" }));
    fireEvent.click(screen.getByRole("button", { name: "Dirty gold review" }));

    await act(async () => {
      pending.resolve([{
        id: "spectrum-2", technique: "NMR", points: 2, name: "Sample B", has_result: false, summary: "",
      }]);
      await pending.promise;
    });
    expect(screen.getByText("Workbench spectrum-1")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Select spectrum Sample B" })).toBeInTheDocument();
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Discard unsaved changes?");
    expect(getSpectrum).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "Keep editing" }));
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Discard unsaved changes?");
  });

  it("requires discarding non-NMR editor drafts before top-level analysis", async () => {
    listSpectra.mockResolvedValue([{
      id: "hplc-1",
      technique: "HPLC",
      points: 32,
      name: "HPLC sample",
      has_result: true,
      summary: "",
      spectrum_revision: 5,
      result_revision: 2,
    }]);
    getSpectrum.mockResolvedValue({
      id: "hplc-1",
      spectrum_revision: 5,
      technique: "HPLC",
      x_data: [0, 1],
      y_data: [0, 1],
      x_label: "Time",
      y_label: "Intensity",
      x_unit: "min",
      y_unit: "",
      parameters: {},
      metadata: {},
      peaks: [],
      source_file: "sample.csv",
    });
    getResult.mockResolvedValue({
      technique: "HPLC",
      result_revision: 2,
      peaks: [],
      metrics: {},
      summary: "Existing result",
    });
    analyzeSpectrum.mockResolvedValue({
      technique: "HPLC",
      result_revision: 3,
      peaks: [],
      metrics: {},
      summary: "Updated result",
    });
    render(<LangProvider><App /></LangProvider>);

    fireEvent.click(await screen.findByRole("button", { name: "Select spectrum HPLC sample" }));
    fireEvent.click(await screen.findByRole("button", { name: "Dirty manual review" }));
    fireEvent.click(screen.getByRole("button", { name: "Dirty HPLC events" }));
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));

    expect(analyzeSpectrum).not.toHaveBeenCalled();
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Discard unsaved changes?");
    fireEvent.click(screen.getByRole("button", { name: "Discard changes" }));
    await waitFor(() => expect(analyzeSpectrum).toHaveBeenCalledWith(
      "hplc-1",
      {},
      expect.any(AbortSignal),
      2,
      false,
    ));
  });

  it("deletes with the revisions captured when the confirmation was opened", async () => {
    deleteSpectrum.mockResolvedValue(undefined);
    render(<LangProvider><App /></LangProvider>);

    fireEvent.click(await screen.findByRole("button", { name: "Delete spectrum Sample A" }));
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Sample A");

    // A background refresh after the decision must not replace the captured revisions.
    listSpectra.mockResolvedValue([{
      id: "spectrum-1",
      technique: "NMR",
      points: 1024,
      name: "Sample A",
      has_result: true,
      summary: "",
      spectrum_revision: 9,
      result_revision: 8,
    }]);
    fireEvent.click(screen.getByRole("button", { name: "Delete spectrum" }));

    await waitFor(() => expect(deleteSpectrum).toHaveBeenCalledWith("spectrum-1", 4, 2));
    expect(getSpectrum).not.toHaveBeenCalled();
  });

  it("ignores a slower stale selection response", async () => {
    listSpectra.mockResolvedValue([
      {
        id: "spectrum-1", technique: "NMR", points: 10, name: "Sample A",
        has_result: true, summary: "",
      },
      {
        id: "spectrum-2", technique: "NMR", points: 10, name: "Sample B",
        has_result: true, summary: "",
      },
    ]);
    let resolveFirst!: (value: SpectrumData | PromiseLike<SpectrumData>) => void;
    let resolveSecond!: (value: SpectrumData | PromiseLike<SpectrumData>) => void;
    getSpectrum.mockImplementation((id: string) => new Promise<SpectrumData>((resolve) => {
      if (id === "spectrum-1") resolveFirst = resolve;
      else resolveSecond = resolve;
    }));
    getResult.mockImplementation((id: string) => Promise.resolve({
      technique: "NMR", result_revision: 2, peaks: [], metrics: {}, summary: id,
    }));
    render(<LangProvider><App /></LangProvider>);

    fireEvent.click(await screen.findByRole("button", { name: "Select spectrum Sample A" }));
    fireEvent.click(screen.getByRole("button", { name: "Select spectrum Sample B" }));
    resolveSecond({
      id: "spectrum-2", technique: "NMR", x_data: [1], y_data: [1],
      x_label: "ppm", y_label: "Intensity", x_unit: "ppm", y_unit: "",
      parameters: {}, metadata: {}, peaks: [], source_file: "b.jdf",
    });
    expect(await screen.findByText("Workbench spectrum-2")).toBeInTheDocument();

    resolveFirst({
      id: "spectrum-1", technique: "NMR", x_data: [1], y_data: [1],
      x_label: "ppm", y_label: "Intensity", x_unit: "ppm", y_unit: "",
      parameters: {}, metadata: {}, peaks: [], source_file: "a.jdf",
    });
    await waitFor(() => expect(screen.queryByText("Workbench spectrum-1")).not.toBeInTheDocument());
  });
});


function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function analysisResult(summary: string): AnalysisResult {
  return { technique: "HPLC", result_revision: 3, peaks: [], metrics: {}, summary };
}

describe("analysis response ownership", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    analyzeSpectrum.mockReset();
    localStorage.setItem("chemapp-lang", "en");
    listSpectra.mockResolvedValue(["A", "B"].map((id) => ({
      id, technique: "HPLC", points: 2, name: `Sample ${id}`, has_result: true,
      summary: "", spectrum_revision: 1, result_revision: 2,
    })));
    getSpectrum.mockImplementation((id) => Promise.resolve({
      id, technique: "HPLC", spectrum_revision: 1, x_data: [0, 1], y_data: [0, 1],
      x_label: "Time", y_label: "Intensity", x_unit: "min", y_unit: "",
      parameters: {}, metadata: {}, peaks: [], source_file: `${id}.csv`,
    }));
    getResult.mockImplementation((id) => Promise.resolve({
      ...analysisResult(`Saved result ${id}`), result_revision: 2,
    }));
  });

  async function selectSample(id: string) {
    fireEvent.click(await screen.findByRole("button", { name: `Select spectrum Sample ${id}` }));
    await screen.findByText(`Saved result ${id}`);
  }

  it("applies an analysis response while its selection is still current", async () => {
    const pending = deferred<AnalysisResult>();
    analyzeSpectrum.mockReturnValue(pending.promise);
    render(<LangProvider><App /></LangProvider>);
    await selectSample("A");
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    expect(screen.getByRole("button", { name: "Analyzing..." })).toBeDisabled();

    await act(async () => {
      pending.resolve(analysisResult("New result A"));
      await pending.promise;
    });
    expect(screen.getByText("New result A")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run Analysis" })).toBeEnabled();
  });

  it("preserves the new spectrum's result and unsaved edits after a stale success", async () => {
    const pending = deferred<AnalysisResult>();
    analyzeSpectrum.mockReturnValue(pending.promise);
    render(<LangProvider><App /></LangProvider>);
    await selectSample("A");
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    await selectSample("B");
    fireEvent.click(screen.getByRole("button", { name: "Dirty manual review" }));
    fireEvent.click(screen.getByRole("button", { name: "Dirty HPLC events" }));

    // A completed server request can still resolve after client cancellation.
    await act(async () => {
      pending.resolve(analysisResult("Stale result A"));
      await pending.promise;
    });
    expect(screen.getByText("Saved result B")).toBeInTheDocument();
    expect(screen.queryByText("Stale result A")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run Analysis" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Discard unsaved changes?");
  });

  it("ignores errors from an earlier selection", async () => {
    const pending = deferred<AnalysisResult>();
    analyzeSpectrum.mockReturnValue(pending.promise);
    render(<LangProvider><App /></LangProvider>);
    await selectSample("A");
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    await selectSample("B");

    await act(async () => {
      pending.reject(new Error("Old analysis failed"));
      await pending.promise.catch(() => undefined);
    });
    expect(screen.queryByText("Old analysis failed")).not.toBeInTheDocument();
    expect(screen.getByText("Saved result B")).toBeInTheDocument();
  });

  it("does not reuse a response after leaving and returning to the same spectrum", async () => {
    const pending = deferred<AnalysisResult>();
    analyzeSpectrum.mockReturnValue(pending.promise);
    render(<LangProvider><App /></LangProvider>);
    await selectSample("A");
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    await selectSample("B");
    await selectSample("A");

    await act(async () => {
      pending.resolve(analysisResult("Old visit result A"));
      await pending.promise;
    });
    expect(screen.getByText("Saved result A")).toBeInTheDocument();
    expect(screen.queryByText("Old visit result A")).not.toBeInTheDocument();
  });

  it("keeps a newer analysis busy when the older request settles", async () => {
    const first = deferred<AnalysisResult>();
    const second = deferred<AnalysisResult>();
    analyzeSpectrum.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    render(<LangProvider><App /></LangProvider>);
    await selectSample("A");
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    await selectSample("B");
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    expect(analyzeSpectrum).toHaveBeenCalledTimes(2);

    await act(async () => {
      first.resolve(analysisResult("Stale result A"));
      await first.promise;
    });
    expect(screen.getByRole("button", { name: "Analyzing..." })).toBeDisabled();
    expect(screen.getByText("Saved result B")).toBeInTheDocument();
    await act(async () => {
      second.resolve(analysisResult("New result B"));
      await second.promise;
    });
    expect(screen.getByText("New result B")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run Analysis" })).toBeEnabled();
  });

  it("aborts pending analysis on selection changes and unmount", async () => {
    const first = deferred<AnalysisResult>();
    const second = deferred<AnalysisResult>();
    analyzeSpectrum.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const view = render(<LangProvider><App /></LangProvider>);
    await selectSample("A");
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    const firstSignal = analyzeSpectrum.mock.calls[0][2];
    expect(firstSignal).toBeInstanceOf(AbortSignal);
    expect(firstSignal?.aborted).toBe(false);
    await selectSample("B");
    expect(firstSignal?.aborted).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    const secondSignal = analyzeSpectrum.mock.calls[1][2];
    view.unmount();
    expect(secondSignal?.aborted).toBe(true);
  });

  it("reports a current failure and allows retrying analysis", async () => {
    analyzeSpectrum.mockRejectedValueOnce(new Error("Current analysis failed"));
    analyzeSpectrum.mockResolvedValueOnce(analysisResult("Retried result A"));
    render(<LangProvider><App /></LangProvider>);
    await selectSample("A");
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    expect(await screen.findByText("Current analysis failed")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    expect(await screen.findByText("Retried result A")).toBeInTheDocument();
    expect(screen.queryByText("Current analysis failed")).not.toBeInTheDocument();
  });

  it("invalidates analysis when its selected spectrum is deleted", async () => {
    const pending = deferred<AnalysisResult>();
    analyzeSpectrum.mockReturnValue(pending.promise);
    deleteSpectrum.mockResolvedValue(undefined);
    render(<LangProvider><App /></LangProvider>);
    await selectSample("A");
    fireEvent.click(screen.getByRole("button", { name: "Run Analysis" }));
    const signal = analyzeSpectrum.mock.calls[0][2];
    fireEvent.click(screen.getByRole("button", { name: "Delete spectrum Sample A" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete spectrum" }));
    await waitFor(() => expect(screen.queryByText("Saved result A")).not.toBeInTheDocument());
    expect(signal?.aborted).toBe(true);

    await selectSample("B");
    await act(async () => {
      pending.resolve(analysisResult("Deleted result A"));
      await pending.promise;
    });
    expect(screen.getByText("Saved result B")).toBeInTheDocument();
    expect(screen.queryByText("Deleted result A")).not.toBeInTheDocument();
  });

  it("keeps the newer selection when an earlier spectrum's deletion completes", async () => {
    const pendingDelete = deferred<void>();
    deleteSpectrum.mockReturnValue(pendingDelete.promise);
    render(<LangProvider><App /></LangProvider>);
    await selectSample("A");
    fireEvent.click(screen.getByRole("button", { name: "Delete spectrum Sample A" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete spectrum" }));
    await selectSample("B");

    await act(async () => {
      pendingDelete.resolve(undefined);
      await pendingDelete.promise;
    });
    expect(screen.getByText("Saved result B")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run Analysis" })).toBeEnabled();
  });


  it("does not restore a deleted spectrum when its deferred load finally resolves", async () => {
    const pendingLoad = deferred<SpectrumData>();
    getSpectrum.mockReturnValueOnce(pendingLoad.promise);
    deleteSpectrum.mockResolvedValue(undefined);
    render(<LangProvider><App /></LangProvider>);
    fireEvent.click(await screen.findByRole("button", { name: "Select spectrum Sample A" }));
    const signal = getSpectrum.mock.calls[0][1];
    fireEvent.click(screen.getByRole("button", { name: "Delete spectrum Sample A" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete spectrum" }));
    await waitFor(() => expect(signal?.aborted).toBe(true));

    await act(async () => {
      pendingLoad.resolve({
        id: "A", technique: "HPLC", x_data: [0], y_data: [1],
        x_label: "Time", y_label: "Intensity", x_unit: "min", y_unit: "",
        parameters: {}, metadata: {}, peaks: [], source_file: "A.csv",
      });
      await pendingLoad.promise;
    });
    expect(screen.queryByText("Saved result A")).not.toBeInTheDocument();
    expect(screen.queryByText("Loading...")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Start with a spectrum" })).toBeInTheDocument();
  });


  it("does not navigate back when an AI refresh finishes after selecting another spectrum", async () => {
    const refresh = deferred<Awaited<ReturnType<typeof api.listSpectra>>>();
    render(<LangProvider><App /></LangProvider>);
    await selectSample("A");
    listSpectra.mockReturnValueOnce(refresh.promise);
    fireEvent.click(screen.getByRole("button", { name: "Finish AI mutation A" }));
    await selectSample("B");
    await act(async () => {
      refresh.resolve([]);
      await refresh.promise;
    });
    expect(screen.getByText("Saved result B")).toBeInTheDocument();
    expect(getSpectrum).toHaveBeenCalledTimes(2);
  });

  it("preserves edits made while an AI mutation refresh was pending", async () => {
    const refresh = deferred<Awaited<ReturnType<typeof api.listSpectra>>>();
    render(<LangProvider><App /></LangProvider>);
    await selectSample("A");
    listSpectra.mockReturnValueOnce(refresh.promise);
    fireEvent.click(screen.getByRole("button", { name: "Finish AI mutation A" }));
    fireEvent.click(screen.getByRole("button", { name: "Dirty manual review" }));
    await act(async () => {
      refresh.resolve([]);
      await refresh.promise;
    });
    expect(screen.getByText("Saved result A")).toBeInTheDocument();
    expect(getSpectrum).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Discard unsaved changes?");
  });


  it("keeps a newly uploaded spectrum when an older initial list arrives late", async () => {
    const initialList = deferred<SpectrumListItem[]>();
    const uploaded = { id: "new", name: "New upload", technique: "NMR", points: 1, has_result: false, summary: "" };
    const existing = { id: "existing", name: "Previously stored", technique: "NMR", points: 1, has_result: false, summary: "" };
    listSpectra.mockReturnValueOnce(initialList.promise).mockResolvedValue([existing, uploaded]);
    uploadFile.mockResolvedValue([uploaded]);
    render(<LangProvider><App /></LangProvider>);
    fireEvent.change(screen.getByLabelText("Choose instrument data files to upload", { selector: "input" }), {
      target: { files: [new File(["data"], "new.jdf")] },
    });
    await screen.findByRole("button", { name: "Select spectrum New upload" });
    await act(async () => {
      initialList.resolve([]);
      await initialList.promise;
    });
    expect(screen.getByRole("button", { name: "Select spectrum New upload" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Select spectrum Previously stored" })).toBeInTheDocument();
    expect(listSpectra).toHaveBeenCalledTimes(2);
  });

  it("keeps the newest list when overlapping refreshes settle out of order", async () => {
    const first = deferred<SpectrumListItem[]>();
    const second = deferred<SpectrumListItem[]>();
    render(<LangProvider><App /></LangProvider>);
    await screen.findByRole("button", { name: "Select spectrum Sample A" });
    listSpectra.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    fireEvent.click(screen.getByRole("button", { name: "Finish AI mutation A" }));
    fireEvent.click(screen.getByRole("button", { name: "Finish AI mutation A" }));
    await act(async () => {
      second.resolve([{ id: "new", name: "Latest list", technique: "NMR", points: 1, has_result: false, summary: "" }]);
      await second.promise;
    });
    await act(async () => {
      first.resolve([]);
      await first.promise;
    });
    expect(screen.getByRole("button", { name: "Select spectrum Latest list" })).toBeInTheDocument();
  });

});
