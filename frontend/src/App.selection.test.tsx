import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type * as api from "./services/api";
import type { SpectrumData } from "./types/spectrum";
import { LangProvider } from "./i18n/LangContext";

const listSpectra = vi.fn<typeof api.listSpectra>();
const getSpectrum = vi.fn<typeof api.getSpectrum>();
const getResult = vi.fn<typeof api.getResult>();
const analyzeSpectrum = vi.fn<typeof api.analyzeSpectrum>();
const deleteSpectrum = vi.fn<typeof api.deleteSpectrum>();

vi.mock("./services/api", () => ({
  listSpectra: (...args: Parameters<typeof api.listSpectra>) => listSpectra(...args),
  getSpectrum: (...args: Parameters<typeof api.getSpectrum>) => getSpectrum(...args),
  getResult: (...args: Parameters<typeof api.getResult>) => getResult(...args),
  analyzeSpectrum: (...args: Parameters<typeof api.analyzeSpectrum>) => analyzeSpectrum(...args),
  deleteSpectrum: (...args: Parameters<typeof api.deleteSpectrum>) => deleteSpectrum(...args),
  downloadMarkdownReport: vi.fn(),
  downloadSpectrumCsv: vi.fn(),
  listExamples: vi.fn().mockResolvedValue([]),
  loadExample: vi.fn(),
  uploadFile: vi.fn(),
  ApiError: class ApiError extends Error {
    status?: number;
  },
}));

vi.mock("./components/AIChatSidebar", () => ({ default: () => null }));
vi.mock("./components/CompareView", () => ({ default: () => null }));
vi.mock("./components/InferencePanel", () => ({ default: () => null }));
vi.mock("./components/SettingsPanel", () => ({ default: () => null }));
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
  default: ({ onDirtyChange }: { onDirtyChange?: (dirty: boolean) => void }) => (
    <div>
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
      undefined,
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
