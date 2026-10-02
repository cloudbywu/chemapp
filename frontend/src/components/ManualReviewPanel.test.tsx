import type { ComponentProps } from "react";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LangProvider } from "../i18n/LangContext";
import { ApiError, integrateRanges, listResultVersions, restoreResultVersion, saveManualResult } from "../services/api";
import type { AnalysisResult, SpectrumData } from "../types/spectrum";
import ManualReviewPanel from "./ManualReviewPanel";

vi.mock("../services/api", () => ({
  ApiError: class ApiError extends Error { status = 0; },
  integrateRanges: vi.fn(), listResultVersions: vi.fn(),
  restoreResultVersion: vi.fn(), saveManualResult: vi.fn(),
}));

const spectrum: SpectrumData = {
  id: "uv-1", technique: "UV-Vis", x_data: [200, 300], y_data: [0, 1],
  x_label: "Wavelength", y_label: "Absorbance", x_unit: "nm", y_unit: "",
  parameters: {}, metadata: {}, peaks: [], source_file: "uv.csv",
};
const result: AnalysisResult = { technique: "UV-Vis", peaks: [], metrics: {}, summary: "", result_revision: 1 };

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

describe("manual peak input validation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem("chemapp-lang", "en");
    vi.mocked(listResultVersions).mockResolvedValue({ versions: [] });
  });

  it.each([
    ["", ""], ["  ", "1"], ["250", ""], ["NaN", "1"], ["250", "Infinity"],
  ])("rejects position %j and intensity %j without creating a peak", async (position, intensity) => {
    const onDirtyChange = vi.fn();
    render(<LangProvider><ManualReviewPanel spectrum={spectrum} result={result} onSaved={vi.fn()} onDirtyChange={onDirtyChange} /></LangProvider>);
    fireEvent.change(screen.getByRole("textbox", { name: "Position" }), { target: { value: position } });
    fireEvent.change(screen.getByRole("textbox", { name: "Intensity" }), { target: { value: intensity } });
    fireEvent.click(screen.getByRole("button", { name: "Add peak" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Enter a valid peak position and intensity");
    expect(screen.getAllByRole("row")).toHaveLength(1);
    expect(onDirtyChange).not.toHaveBeenCalledWith(true);
  });

  it("accepts explicit zeros rather than confusing zero with a missing input", async () => {
    const onDirtyChange = vi.fn();
    render(<LangProvider><ManualReviewPanel spectrum={spectrum} result={result} onSaved={vi.fn()} onDirtyChange={onDirtyChange} /></LangProvider>);
    fireEvent.change(screen.getByRole("textbox", { name: "Position" }), { target: { value: "0" } });
    fireEvent.change(screen.getByRole("textbox", { name: "Intensity" }), { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: "Add peak" }));
    expect(await screen.findByRole("cell", { name: "0.0000" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "0.00e+0" })).toBeInTheDocument();
    expect(onDirtyChange).toHaveBeenCalledWith(true);
  });

  it("does not publish a saved result after the editor unmounts", async () => {
    const pending = deferred<AnalysisResult>();
    vi.mocked(saveManualResult).mockReturnValue(pending.promise);
    const onSaved = vi.fn();
    const mounted = render(<LangProvider><ManualReviewPanel spectrum={spectrum} result={result} onSaved={onSaved} /></LangProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Save reviewed version" }));
    expect(saveManualResult).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Saving..." })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Add peak" })).toBeDisabled();
    for (const input of screen.getAllByRole("textbox")) expect(input).toBeDisabled();
    mounted.unmount();
    await act(async () => { pending.resolve({ ...result, result_revision: 2 }); await pending.promise; });
    expect(onSaved).not.toHaveBeenCalled();
    expect(listResultVersions).toHaveBeenCalledTimes(1);
  });

  it("does not publish a restored result after the editor unmounts", async () => {
    const pending = deferred<AnalysisResult>();
    vi.mocked(restoreResultVersion).mockReturnValue(pending.promise);
    vi.mocked(listResultVersions).mockResolvedValue({ versions: [{
      version: 1, note: "Reviewed", created_at: "2026-10-01", n_peaks: 0,
      manual_confirmed: true, summary: "",
    }] });
    const onSaved = vi.fn();
    const mounted = render(<LangProvider><ManualReviewPanel spectrum={spectrum} result={result} onSaved={onSaved} /></LangProvider>);
    fireEvent.click(await screen.findByRole("button", { name: "Restore v1" }));
    fireEvent.click(screen.getByRole("button", { name: "Restore version" }));
    expect(restoreResultVersion).toHaveBeenCalledWith("uv-1", 1, 1);
    expect(screen.getByRole("button", { name: "Add peak" })).toBeDisabled();
    for (const input of screen.getAllByRole("textbox")) expect(input).toBeDisabled();
    mounted.unmount();
    await act(async () => { pending.resolve({ ...result, result_revision: 2 }); await pending.promise; });
    expect(onSaved).not.toHaveBeenCalled();
  });

  it("releases the save lock on failure so the user can retry", async () => {
    vi.mocked(saveManualResult).mockRejectedValueOnce(new Error("Temporary failure")).mockResolvedValueOnce({ ...result, result_revision: 2 });
    const onSaved = vi.fn();
    render(<LangProvider><ManualReviewPanel spectrum={spectrum} result={result} onSaved={onSaved} /></LangProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Save reviewed version" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Temporary failure");
    expect(screen.getByRole("button", { name: "Add peak" })).toBeEnabled();
    for (const input of screen.getAllByRole("textbox")) expect(input).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Save reviewed version" }));
    expect(await screen.findByText("Reviewed version saved")).toBeInTheDocument();
    expect(onSaved).toHaveBeenCalledWith(expect.objectContaining({ result_revision: 2 }));
  });
});

type IntegrationResponse = Awaited<ReturnType<typeof integrateRanges>>;
const integrationResponse = (...areas: number[]): IntegrationResponse => ({
  integrals: areas.map((area, index) => ({ start: index, end: index + 1, center: index + 0.5, area })),
});
const nmrResult: AnalysisResult = {
  ...result, technique: "NMR",
  integrals: [1, 2].map((center) => ({
    center_ppm: center, start_ppm: center - 0.1, end_ppm: center + 0.1,
    raw_area: center * 11, relative_area: 1, intensity: 1,
  })),
};
const hplcResult: AnalysisResult = {
  ...result, technique: "HPLC", peaks: [1, 2].map((position) => ({
    position, intensity: 1, area: position * 11, width: 0.1,
    assignment: "A", multiplicity: "", coupling_constant: null,
  })),
};
function renderEditor(technique: "NMR" | "HPLC", extra: Partial<ComponentProps<typeof ManualReviewPanel>> = {}) {
  return <LangProvider><ManualReviewPanel
    spectrum={{ ...spectrum, technique, parameters: { channels: [{ name: "A" }, { name: "B" }] } }}
    result={technique === "NMR" ? nmrResult : hplcResult}
    onSaved={vi.fn()}
    {...extra}
  /></LangProvider>;
}
const rangeTable = (technique: "NMR" | "HPLC") => {
  const section = screen.getByRole("heading", { name: technique === "NMR" ? "NMR integration ranges" : "HPLC peak reintegration" }).closest(".table-section");
  if (!section) throw new Error("Range table not found");
  return within(section as HTMLElement);
};
const rangeRow = (technique: "NMR" | "HPLC", index = 0) => within(rangeTable(technique).getAllByRole("row")[index + 1]);

describe("manual recalculation ownership", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    localStorage.setItem("chemapp-lang", "en");
    vi.mocked(listResultVersions).mockResolvedValue({ versions: [] });
  });

  it("keeps the center input mounted and focused as its value changes", () => {
    render(renderEditor("NMR"));
    const center = rangeRow("NMR").getAllByRole("textbox")[0];
    center.focus();
    fireEvent.change(center, { target: { value: "1.25" } });
    expect(rangeRow("NMR").getAllByRole("textbox")[0]).toBe(center);
    expect(center).toHaveFocus();
    expect(center).toHaveValue("1.25");
  });

  it.each(["NMR", "HPLC"] as const)("announces %s recalculation failures and allows retry", async (technique) => {
    vi.mocked(integrateRanges).mockRejectedValueOnce(new Error("Integration offline")).mockResolvedValueOnce(integrationResponse(0, 0));
    render(renderEditor(technique));
    const button = technique === "NMR" ? screen.getByRole("button", { name: "Recalculate integrals" }) : rangeRow("HPLC").getByRole("button", { name: "Recalculate" });
    fireEvent.click(button);
    expect(await screen.findByRole("status")).toHaveTextContent("Integration offline");
    fireEvent.click(button);
    expect(await screen.findAllByRole("cell", { name: "0.00" })).not.toHaveLength(0);
    expect(screen.queryByText("Integration offline")).not.toBeInTheDocument();
  });

  it("applies a batch response only to NMR ranges that have not been edited", async () => {
    const pending = deferred<IntegrationResponse>();
    vi.mocked(integrateRanges).mockReturnValue(pending.promise);
    render(renderEditor("NMR"));
    fireEvent.click(screen.getByRole("button", { name: "Recalculate integrals" }));
    fireEvent.change(rangeRow("NMR").getAllByRole("textbox")[1], { target: { value: "0.8" } });
    await act(async () => { pending.resolve(integrationResponse(111, 222)); await pending.promise; });
    expect(rangeRow("NMR").getByRole("cell", { name: "11.00" })).toBeInTheDocument();
    expect(rangeRow("NMR", 1).getByRole("cell", { name: "222.00" })).toBeInTheDocument();
    expect(rangeRow("NMR").getAllByRole("textbox")[1]).toHaveValue("0.8");
  });

  it("keeps the newer NMR calculation when requests resolve out of order", async () => {
    const old = deferred<IntegrationResponse>();
    const newer = deferred<IntegrationResponse>();
    vi.mocked(integrateRanges).mockReturnValueOnce(old.promise).mockReturnValueOnce(newer.promise);
    render(renderEditor("NMR"));
    const button = screen.getByRole("button", { name: "Recalculate integrals" });
    fireEvent.click(button);
    fireEvent.click(button);
    await act(async () => { newer.resolve(integrationResponse(0, 42)); await newer.promise; });
    await act(async () => { old.resolve(integrationResponse(111, 222)); await old.promise; });
    expect(rangeRow("NMR").getByRole("cell", { name: "0.00" })).toBeInTheDocument();
    expect(rangeRow("NMR", 1).getByRole("cell", { name: "42.00" })).toBeInTheDocument();
  });

  it.each(["start", "end", "channel"])("invalidates HPLC responses when their %s changes", async (field) => {
    const pending = deferred<IntegrationResponse>();
    vi.mocked(integrateRanges).mockReturnValue(pending.promise);
    render(renderEditor("HPLC"));
    const row = rangeRow("HPLC");
    fireEvent.click(row.getByRole("button", { name: "Recalculate" }));
    if (field === "channel") fireEvent.change(row.getByRole("combobox"), { target: { value: "B" } });
    else fireEvent.change(row.getAllByRole("textbox")[field === "start" ? 0 : 1], { target: { value: field === "start" ? "0.8" : "1.2" } });
    await act(async () => { pending.resolve(integrationResponse(999)); await pending.promise; });
    expect(screen.getByRole("cell", { name: "11.00" })).toBeInTheDocument();
    expect(screen.queryByRole("cell", { name: "999.00" })).not.toBeInTheDocument();
  });

  it("lets independent HPLC rows complete without cancelling each other", async () => {
    const first = deferred<IntegrationResponse>();
    const second = deferred<IntegrationResponse>();
    vi.mocked(integrateRanges).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    render(renderEditor("HPLC"));
    fireEvent.click(rangeRow("HPLC").getByRole("button", { name: "Recalculate" }));
    fireEvent.click(rangeRow("HPLC", 1).getByRole("button", { name: "Recalculate" }));
    await act(async () => { second.resolve(integrationResponse(222)); await second.promise; });
    await act(async () => { first.resolve(integrationResponse(111)); await first.promise; });
    expect(screen.getByRole("cell", { name: "111.00" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "222.00" })).toBeInTheDocument();
  });

  it("ignores an obsolete plot-picked result after the same range is picked again", async () => {
    const old = deferred<IntegrationResponse>();
    const newer = deferred<IntegrationResponse>();
    vi.mocked(integrateRanges).mockReturnValueOnce(old.promise).mockReturnValueOnce(newer.promise);
    const mounted = render(renderEditor("NMR", { pickedPlotRange: { mode: "nmr", index: 0, start: 0.8, end: 1.2, nonce: 1 } }));
    mounted.rerender(renderEditor("NMR", { pickedPlotRange: { mode: "nmr", index: 0, start: 0.7, end: 1.3, nonce: 2 } }));
    await act(async () => { newer.resolve(integrationResponse(42)); await newer.promise; });
    await act(async () => { old.resolve(integrationResponse(99)); await old.promise; });
    expect(rangeRow("NMR").getByRole("cell", { name: "42.00" })).toBeInTheDocument();
    expect(rangeRow("NMR").getAllByRole("textbox")[1]).toHaveValue("0.7");
  });

  it("uses the peak's assigned HPLC channel for a first plot-picked window", async () => {
    vi.mocked(integrateRanges).mockResolvedValue(integrationResponse(42));
    render(renderEditor("HPLC", {
      result: { ...hplcResult, peaks: hplcResult.peaks.map((peak) => ({ ...peak, assignment: "B" })) },
      pickedPlotRange: { mode: "hplc", index: 0, start: 0.8, end: 1.2, nonce: 1 },
    }));
    expect(integrateRanges).toHaveBeenCalledWith("uv-1", [{ start: 0.8, end: 1.2, center: 1, channel: "B" }]);
    expect(await screen.findByRole("cell", { name: "42.00" })).toBeInTheDocument();
    expect(rangeRow("HPLC").getByRole("combobox")).toHaveValue("B");
  });

  it("keeps custom windows with their existing peaks when insertion sorts the rows", () => {
    render(renderEditor("HPLC", { result: { ...hplcResult, peaks: [...hplcResult.peaks].reverse() } }));
    fireEvent.change(rangeRow("HPLC").getAllByRole("textbox")[0], { target: { value: "1.7" } });
    fireEvent.change(rangeRow("HPLC").getByRole("combobox"), { target: { value: "B" } });
    fireEvent.change(rangeRow("HPLC", 1).getAllByRole("textbox")[0], { target: { value: "0.7" } });
    fireEvent.change(screen.getByRole("textbox", { name: "Position" }), { target: { value: "0.5" } });
    fireEvent.change(screen.getByRole("textbox", { name: "Intensity" }), { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: "Add peak" }));
    expect(rangeRow("HPLC").getByRole("cell", { name: "0.500" })).toBeInTheDocument();
    expect(rangeRow("HPLC").getAllByRole("textbox")[0]).toHaveValue("0.45");
    expect(rangeRow("HPLC", 1).getByRole("cell", { name: "1.000" })).toBeInTheDocument();
    expect(rangeRow("HPLC", 1).getAllByRole("textbox")[0]).toHaveValue("0.7");
    expect(rangeRow("HPLC", 2).getByRole("cell", { name: "2.000" })).toBeInTheDocument();
    expect(rangeRow("HPLC", 2).getAllByRole("textbox")[0]).toHaveValue("1.7");
    expect(rangeRow("HPLC", 2).getByRole("combobox")).toHaveValue("B");
  });

  it("does not copy a custom window onto an inserted peak at the same position", () => {
    render(renderEditor("HPLC"));
    fireEvent.change(rangeRow("HPLC").getAllByRole("textbox")[0], { target: { value: "0.7" } });
    fireEvent.change(screen.getByRole("textbox", { name: "Position" }), { target: { value: "1" } });
    fireEvent.change(screen.getByRole("textbox", { name: "Intensity" }), { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: "Add peak" }));
    expect(rangeRow("HPLC").getAllByRole("textbox")[0]).toHaveValue("0.7");
    expect(rangeRow("HPLC", 1).getAllByRole("textbox")[0]).toHaveValue("0.95");
  });

  it("removes a deleted peak's window and shifts only the surviving peak's window", async () => {
    vi.mocked(integrateRanges).mockResolvedValue(integrationResponse(42));
    render(renderEditor("HPLC"));
    fireEvent.change(rangeRow("HPLC").getAllByRole("textbox")[0], { target: { value: "0.7" } });
    fireEvent.change(rangeRow("HPLC", 1).getAllByRole("textbox")[0], { target: { value: "1.7" } });
    fireEvent.change(rangeRow("HPLC", 1).getByRole("combobox"), { target: { value: "B" } });
    fireEvent.click(screen.getByRole("button", { name: "Delete peak at 1" }));
    expect(rangeRow("HPLC").getByRole("cell", { name: "2.000" })).toBeInTheDocument();
    expect(rangeRow("HPLC").getAllByRole("textbox")[0]).toHaveValue("1.7");
    expect(rangeRow("HPLC").getByRole("combobox")).toHaveValue("B");
    fireEvent.click(rangeRow("HPLC").getByRole("button", { name: "Recalculate" }));
    expect(integrateRanges).toHaveBeenCalledWith("uv-1", [{ start: 1.7, end: 2.1, center: 2, channel: "B", baseline: "linear" }]);
    expect(await screen.findByRole("cell", { name: "42.00" })).toBeInTheDocument();
  });

  it("does not show a stale HPLC failure after the window changes", async () => {
    const pending = deferred<IntegrationResponse>();
    vi.mocked(integrateRanges).mockReturnValue(pending.promise);
    render(renderEditor("HPLC"));
    fireEvent.click(rangeRow("HPLC").getByRole("button", { name: "Recalculate" }));
    fireEvent.change(rangeRow("HPLC").getAllByRole("textbox")[0], { target: { value: "0.8" } });
    await act(async () => { pending.reject(new Error("Obsolete failure")); await pending.promise.catch(() => undefined); });
    expect(screen.queryByText("Obsolete failure")).not.toBeInTheDocument();
  });

  it("invalidates recalculation before saving, even when that save later fails", async () => {
    const pending = deferred<IntegrationResponse>();
    vi.mocked(integrateRanges).mockReturnValue(pending.promise);
    vi.mocked(saveManualResult).mockRejectedValue(new Error("Save offline"));
    render(renderEditor("NMR"));
    fireEvent.click(screen.getByRole("button", { name: "Recalculate integrals" }));
    fireEvent.click(screen.getByRole("button", { name: "Save reviewed version" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Save offline");
    await act(async () => { pending.resolve(integrationResponse(111, 222)); await pending.promise; });
    expect(rangeRow("NMR").getByRole("cell", { name: "11.00" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Save offline");
  });

  it.each(["NMR", "HPLC"] as const)("ignores %s recalculation completion after unmount", async (technique) => {
    const pending = deferred<IntegrationResponse>();
    vi.mocked(integrateRanges).mockReturnValue(pending.promise);
    const onDirtyChange = vi.fn();
    const mounted = render(renderEditor(technique, { onDirtyChange }));
    fireEvent.click(technique === "NMR" ? screen.getByRole("button", { name: "Recalculate integrals" }) : rangeRow("HPLC").getByRole("button", { name: "Recalculate" }));
    mounted.unmount();
    onDirtyChange.mockClear();
    await act(async () => { pending.resolve(integrationResponse(111, 222)); await pending.promise; });
    expect(onDirtyChange).not.toHaveBeenCalled();
  });
});

describe("historical result source provenance", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    localStorage.setItem("chemapp-lang", "en");
  });

  const version = (number: number) => ({
    version: number, note: `Review ${number}`, created_at: "2026-10-01", n_peaks: 0,
    manual_confirmed: true, summary: "",
  });

  it.each(["en", "zh"])("lists blocked history with clear reasons in %s and preserves compatible versions", async (language) => {
    localStorage.setItem("chemapp-lang", language);
    vi.mocked(listResultVersions).mockResolvedValue({ versions: [
      { ...version(1), spectrum_revision: null, restorable: false },
      { ...version(2), spectrum_revision: 1, restorable: false },
      version(3),
      { ...version(4), spectrum_revision: 2, restorable: true },
    ] });
    render(renderEditor("NMR"));
    const restore = language === "en" ? "Restore" : "恢复";
    expect(await screen.findByRole("button", { name: `${restore} v1` })).toBeDisabled();
    expect(screen.getByRole("button", { name: `${restore} v2` })).toBeDisabled();
    expect(screen.getByRole("button", { name: `${restore} v3` })).toBeEnabled();
    expect(screen.getByRole("button", { name: `${restore} v4` })).toBeEnabled();
    expect(screen.getByText(language === "en"
      ? "This historical version has no verifiable source spectrum revision. It remains available to view but cannot be restored."
      : "此历史版本缺少可验证的原始谱图修订信息，仅供查看，无法恢复。"
    )).toBeInTheDocument();
    expect(screen.getByText(language === "en"
      ? "This historical version belongs to a different spectrum revision and cannot be restored. Reanalyze the current spectrum."
      : "此历史版本来自不同的谱图修订，无法恢复。请重新分析当前谱图。"
    )).toBeInTheDocument();
    expect(restoreResultVersion).not.toHaveBeenCalled();
  });

  it("refreshes provenance after a rejected restore rather than reporting an unrelated result conflict", async () => {
    vi.mocked(listResultVersions)
      .mockResolvedValueOnce({ versions: [{ ...version(1), spectrum_revision: 1, restorable: true }] })
      .mockResolvedValueOnce({ versions: [{ ...version(1), spectrum_revision: 1, restorable: false }] });
    vi.mocked(restoreResultVersion).mockRejectedValue(Object.assign(new ApiError("Source changed"), { status: 409 }));
    render(renderEditor("NMR"));
    fireEvent.click(await screen.findByRole("button", { name: "Restore v1" }));
    fireEvent.click(screen.getByRole("button", { name: "Restore version" }));
    expect(await screen.findByRole("status")).toHaveTextContent("This historical version belongs to a different spectrum revision");
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Restore v1" })).toBeDisabled();
    expect(screen.queryByText("The result was updated by another operation. Reload before editing it again.")).not.toBeInTheDocument();
  });
});
