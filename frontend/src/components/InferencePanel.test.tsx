import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LangProvider } from "../i18n/LangContext";
import {
  analyzeBatch, compareHplcBatch, downloadBatchCsvZip, downloadDocxReport,
  downloadHtmlReport, downloadHplcComparisonCsv, downloadMarkdownReport,
  getBatchQuality, getBatchWorkbench, runInference,
} from "../services/api";
import type { InferenceResponse, SpectrumListItem } from "../types/spectrum";
import InferencePanel from "./InferencePanel";

vi.mock("../services/api", () => ({
  analyzeBatch: vi.fn(), compareHplcBatch: vi.fn(), downloadBatchCsvZip: vi.fn(),
  downloadDocxReport: vi.fn(), downloadHtmlReport: vi.fn(), downloadHplcComparisonCsv: vi.fn(),
  downloadMarkdownReport: vi.fn(), getBatchQuality: vi.fn(), getBatchWorkbench: vi.fn(), runInference: vi.fn(),
}));

const spectra: SpectrumListItem[] = ["a", "b", "c"].map((id) => ({
  id, name: id, technique: "HPLC", points: 10, has_result: true,
  summary: "", spectrum_revision: 2, result_revision: 3,
}));
const inference = (marker: string): InferenceResponse => ({
  sample_name: "", techniques: [], generated_at: "", report_markdown: "",
  inference: {
    technique_results: {}, cross_validations: [], consistency_score: 0.8,
    confidence: 0.8, anomalies: [], conclusions: [], overall_assessment: marker,
  },
});
const quality = (marker: string): Awaited<ReturnType<typeof getBatchQuality>> => ({
  counts: { good: 1 },
  items: [{ id: "a", name: marker, technique: "HPLC", status: "good", score: 0.8,
    warnings: [], info: [], n_peaks: 1, manual_confirmed: false }],
});
const workbench = (marker: string): Awaited<ReturnType<typeof getBatchWorkbench>> => ({
  summary: { count: 1, manual_confirmed: 0, total_points: 10, technique_counts: { HPLC: 1 }, quality_counts: {} },
  items: [{ id: "a", name: marker, technique: "HPLC", points: 10, n_peaks: 0,
    quality: null, manual_confirmed: false, ai_modified: false, summary: "No quality yet", export_ready: true }],
});
const hplc = (marker: string): Awaited<ReturnType<typeof compareHplcBatch>> => ({
  reference_id: "a", channel: "", rt_tolerance: 0.08, rows: [],
  drift: [{ id: "a", name: marker, channel: "", mean_rt_shift: 0,
    max_abs_rt_shift: 0, matched_peaks: 1, total_area: 20 }],
});
const batchResult: Awaited<ReturnType<typeof analyzeBatch>> = {
  results: [{ id: "a", name: "a", technique: "HPLC", n_peaks: 1, summary: "Analyzed", metrics: {}, result_revision: 4 }],
  errors: [{ id: "b", error: "Needs review" }],
};
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}
const view = (items = spectra, onDataChanged?: () => void) => (
  <LangProvider><InferencePanel spectra={items} onDataChanged={onDataChanged} /></LangProvider>
);
const toggle = (id: string) => fireEvent.click(screen.getByRole("checkbox", { name: `HPLC ${id}` }));
const selectPair = () => { toggle("a"); toggle("b"); };
const click = (label: string) => fireEvent.click(screen.getByRole("button", { name: label }));
const markers = (marker: string) => screen.queryAllByText(marker, { exact: true });

beforeEach(() => {
  vi.resetAllMocks();
  localStorage.setItem("chemapp-lang", "en");
});

function readRegressions<T>(label: string, api: (ids: string[]) => Promise<T>, response: (marker: string) => T) {
  describe(`${label} ownership`, () => {
    it("hides completed results immediately after selection changes", async () => {
      vi.mocked(api).mockReturnValue(Promise.resolve(response("Old result")));
      render(view());
      selectPair();
      click(label);
      await screen.findAllByText("Old result");
      toggle("b");
      expect(markers("Old result")).toHaveLength(0);
    });

    it("ignores an older success without releasing the newer request", async () => {
      const oldRequest = deferred<T>();
      const newRequest = deferred<T>();
      vi.mocked(api).mockReturnValueOnce(oldRequest.promise).mockReturnValueOnce(newRequest.promise);
      render(view());
      selectPair();
      click(label);
      toggle("b");
      toggle("c");
      click(label);
      expect(vi.mocked(api).mock.calls.map(([ids]) => ids)).toEqual([["a", "b"], ["a", "c"]]);
      await act(async () => { oldRequest.resolve(response("Old result")); await oldRequest.promise; });
      expect(markers("Old result")).toHaveLength(0);
      expect(screen.getByRole("status")).toHaveTextContent("Loading...");
      expect(screen.getByRole("button", { name: label === "Run Inference" ? "Running..." : `${label} · Loading...` })).toBeDisabled();
      await act(async () => { newRequest.resolve(response("Current result")); await newRequest.promise; });
      expect(markers("Current result").length).toBeGreaterThan(0);
      expect(screen.getByRole("button", { name: label })).toBeEnabled();
    });

    it("ignores a late failure after leaving and returning to the same selection", async () => {
      const oldRequest = deferred<T>();
      vi.mocked(api).mockReturnValueOnce(oldRequest.promise).mockReturnValueOnce(Promise.resolve(response("Current result")));
      render(view());
      selectPair();
      click(label);
      toggle("b");
      toggle("b");
      click(label);
      await screen.findAllByText("Current result");
      await act(async () => { oldRequest.reject(new Error("Old failure")); await oldRequest.promise.catch(() => undefined); });
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(markers("Current result").length).toBeGreaterThan(0);
    });

    it.each(["spectrum_revision", "result_revision"] as const)("invalidates pending reads when %s changes", async (revision) => {
      const pending = deferred<T>();
      vi.mocked(api).mockReturnValue(pending.promise);
      const mounted = render(view());
      selectPair();
      click(label);
      mounted.rerender(view(spectra.map((item) => item.id === "a" ? { ...item, [revision]: 9 } : item)));
      expect(screen.getByRole("button", { name: label })).toBeEnabled();
      await act(async () => { pending.resolve(response("Old revision")); await pending.promise; });
      expect(markers("Old revision")).toHaveLength(0);
    });

    it("owns repeated gestures synchronously and announces loading", async () => {
      const pending = deferred<T>();
      vi.mocked(api).mockReturnValue(pending.promise);
      render(view());
      selectPair();
      const button = screen.getByRole("button", { name: label });
      act(() => { fireEvent.click(button); fireEvent.click(button); });
      expect(api).toHaveBeenCalledTimes(1);
      expect(button).toHaveAttribute("aria-busy", "true");
      await act(async () => { pending.resolve(response("Current result")); await pending.promise; });
      expect(button).toHaveAttribute("aria-busy", "false");
    });

    it("announces current failures and permits retry", async () => {
      vi.mocked(api).mockRejectedValue(new Error("Current failure"));
      render(view());
      selectPair();
      click(label);
      expect(await screen.findByRole("alert")).toHaveTextContent("Current failure");
      expect(screen.getByRole("button", { name: label })).toBeEnabled();
    });

    it("cannot publish into a replacement panel after unmount", async () => {
      const pending = deferred<T>();
      vi.mocked(api).mockReturnValue(pending.promise);
      const mounted = render(view());
      selectPair();
      click(label);
      mounted.unmount();
      render(view());
      await act(async () => { pending.resolve(response("Unmounted result")); await pending.promise; });
      expect(markers("Unmounted result")).toHaveLength(0);
      expect(screen.queryByRole("status")).not.toBeInTheDocument();
    });
  });
}
readRegressions("Run Inference", runInference, inference);
readRegressions("Quality overview", getBatchQuality, quality);
readRegressions("Batch workbench", getBatchWorkbench, workbench);
readRegressions("Match HPLC peaks", compareHplcBatch, hplc);

describe("report request ownership", () => {
  it.each([
    ["Markdown", downloadMarkdownReport], ["HTML", downloadHtmlReport],
    ["Word", downloadDocxReport], ["Export data bundle", downloadBatchCsvZip],
    ["HPLC CSV", downloadHplcComparisonCsv],
  ] as const)("deduplicates %s and ignores old errors without clearing newer busy state", async (label, api) => {
    const oldRequest = deferred<void>();
    const newRequest = deferred<void>();
    vi.mocked(api).mockReturnValueOnce(oldRequest.promise).mockReturnValueOnce(newRequest.promise);
    render(view());
    selectPair();
    const button = screen.getByRole("button", { name: label });
    act(() => { fireEvent.click(button); fireEvent.click(button); });
    expect(api).toHaveBeenCalledTimes(1);
    expect(button).toHaveAttribute("aria-busy", "true");
    toggle("b");
    toggle("c");
    click(label);
    await act(async () => { oldRequest.reject(new Error("Old report failure")); await oldRequest.promise.catch(() => undefined); });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: `${label} · Loading...` })).toBeDisabled();
    await act(async () => { newRequest.resolve(); await newRequest.promise; });
    expect(screen.getByRole("button", { name: label })).toBeEnabled();
  });

  it("does not replace a newer operation's error with an older independent error", async () => {
    const oldReport = deferred<void>();
    vi.mocked(downloadMarkdownReport).mockReturnValue(oldReport.promise);
    vi.mocked(getBatchQuality).mockRejectedValue(new Error("Current quality error"));
    render(view());
    selectPair();
    click("Markdown");
    click("Quality overview");
    expect(await screen.findByRole("alert")).toHaveTextContent("Current quality error");
    await act(async () => { oldReport.reject(new Error("Older report error")); await oldReport.promise.catch(() => undefined); });
    expect(screen.getByRole("alert")).toHaveTextContent("Current quality error");
  });
});

describe("selection lifecycle", () => {
  it("prunes deleted IDs, clears their results, and does not reselect reappearing records", async () => {
    vi.mocked(runInference).mockResolvedValue(inference("Removed selection"));
    const mounted = render(view());
    selectPair();
    click("Run Inference");
    await screen.findByText("Removed selection");
    mounted.rerender(view(spectra.filter((item) => item.id === "c")));
    expect(screen.getByText("0 selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run Inference" })).toBeDisabled();
    expect(markers("Removed selection")).toHaveLength(0);
    mounted.rerender(view());
    expect(screen.getByRole("checkbox", { name: "HPLC a" })).not.toBeChecked();
    toggle("c");
    click("Run Inference");
    await screen.findByText("Removed selection");
    expect(runInference).toHaveBeenLastCalledWith(["c"]);
  });

  it("preserves selection order and results when unrelated records or list order change", async () => {
    vi.mocked(runInference).mockResolvedValue(inference("Stable result"));
    const mounted = render(view());
    toggle("b");
    toggle("a");
    click("Run Inference");
    await screen.findByText("Stable result");
    expect(runInference).toHaveBeenCalledWith(["b", "a"]);
    mounted.rerender(view([...spectra].reverse().map((item) => item.id === "c" ? { ...item, result_revision: 8 } : item)));
    expect(screen.getByText("Stable result")).toBeInTheDocument();
  });
});

describe("batch mutation ownership", () => {
  it("captures expected revisions, suppresses duplicate gestures, and announces current partial success", async () => {
    const pending = deferred<Awaited<ReturnType<typeof analyzeBatch>>>();
    const onDataChanged = vi.fn();
    vi.mocked(analyzeBatch).mockReturnValue(pending.promise);
    render(view(spectra, onDataChanged));
    selectPair();
    const button = screen.getByRole("button", { name: "Batch analyze" });
    act(() => { fireEvent.click(button); fireEvent.click(button); });
    expect(analyzeBatch).toHaveBeenCalledExactlyOnceWith(["a", "b"], { a: 3, b: 3 });
    expect(button).toHaveAttribute("aria-busy", "true");
    await act(async () => { pending.resolve(batchResult); await pending.promise; });
    expect(onDataChanged).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("status")).toHaveTextContent("Analyzed 1 spectra; 1 failed");
    expect(screen.getByRole("button", { name: "Batch analyze" })).toBeEnabled();
  });

  it("invalidates in-flight reads before mutating their source data", async () => {
    const read = deferred<InferenceResponse>();
    const batch = deferred<Awaited<ReturnType<typeof analyzeBatch>>>();
    vi.mocked(runInference).mockReturnValue(read.promise);
    vi.mocked(analyzeBatch).mockReturnValue(batch.promise);
    render(view());
    selectPair();
    click("Run Inference");
    click("Batch analyze");
    await act(async () => { read.resolve(inference("Pre-mutation result")); await read.promise; });
    expect(markers("Pre-mutation result")).toHaveLength(0);
    expect(screen.getByRole("button", { name: "Run Inference" })).toBeDisabled();
    await act(async () => { batch.resolve(batchResult); await batch.promise; });
    expect(screen.getByRole("button", { name: "Run Inference" })).toBeEnabled();
  });

  it("keeps a mutation locked across selections and refreshes shared data without publishing its old summary", async () => {
    const pending = deferred<Awaited<ReturnType<typeof analyzeBatch>>>();
    const firstCallback = vi.fn();
    const latestCallback = vi.fn();
    vi.mocked(analyzeBatch).mockReturnValue(pending.promise);
    const mounted = render(view(spectra, firstCallback));
    selectPair();
    click("Batch analyze");
    toggle("b");
    toggle("c");
    mounted.rerender(view(spectra, latestCallback));
    expect(screen.getByRole("button", { name: "Batch analysis running…" })).toBeDisabled();
    await act(async () => { pending.resolve(batchResult); await pending.promise; });
    expect(firstCallback).not.toHaveBeenCalled();
    expect(latestCallback).toHaveBeenCalledTimes(1);
    expect(screen.queryByText("Analyzed 1 spectra; 1 failed")).not.toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "HPLC c" })).toBeChecked();
    expect(analyzeBatch).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Batch analyze" })).toBeEnabled();
  });

  it("does not report stale batch errors to a different selection", async () => {
    const pending = deferred<Awaited<ReturnType<typeof analyzeBatch>>>();
    vi.mocked(analyzeBatch).mockReturnValue(pending.promise);
    render(view());
    selectPair();
    click("Batch analyze");
    toggle("b");
    await act(async () => { pending.reject(new Error("Old batch error")); await pending.promise.catch(() => undefined); });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Batch analyze" })).toBeEnabled();
  });

  it("refreshes shared data after the panel unmounts", async () => {
    const pending = deferred<Awaited<ReturnType<typeof analyzeBatch>>>();
    const onDataChanged = vi.fn();
    vi.mocked(analyzeBatch).mockReturnValue(pending.promise);
    const mounted = render(view(spectra, onDataChanged));
    selectPair();
    click("Batch analyze");
    mounted.unmount();
    await act(async () => { pending.resolve(batchResult); await pending.promise; });
    expect(onDataChanged).toHaveBeenCalledTimes(1);
  });
});
