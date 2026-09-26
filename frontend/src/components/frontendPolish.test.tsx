import type { ReactElement } from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { LangProvider } from "../i18n/LangContext";
import CompareView from "./CompareView";
import ErrorBoundary from "./ErrorBoundary";
import FluorescencePanel from "./FluorescencePanel";
import InferencePanel from "./InferencePanel";
import SettingsPanel from "./SettingsPanel";
import { compareHplcBatch, compareSpectra, getBatchQuality } from "../services/api";

vi.mock("../services/api", () => ({
  analyzeBatch: vi.fn(),
  compareHplcBatch: vi.fn(),
  compareSpectra: vi.fn(),
  downloadBatchCsvZip: vi.fn(),
  downloadDocxReport: vi.fn(),
  downloadHtmlReport: vi.fn(),
  downloadHplcComparisonCsv: vi.fn(),
  downloadMarkdownReport: vi.fn(),
  getBatchQuality: vi.fn(),
  getBatchWorkbench: vi.fn(),
  runInference: vi.fn(),
}));

const renderWithLang = (ui: ReactElement) => render(<LangProvider>{ui}</LangProvider>);

describe("frontend polish i18n coverage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem("chemapp-lang", "en");
  });

  it("renders the ErrorBoundary fallback through i18n (en)", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const Thrower = () => {
      throw new Error("boom");
    };
    renderWithLang(
      <ErrorBoundary>
        <Thrower />
      </ErrorBoundary>,
    );
    expect(await screen.findByRole("heading", { name: "Something went wrong" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry this view" })).toBeInTheDocument();
    errorSpy.mockRestore();
  });

  it("renders the ErrorBoundary fallback through i18n (zh)", async () => {
    localStorage.setItem("chemapp-lang", "zh");
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const Thrower = () => {
      throw new Error("boom");
    };
    renderWithLang(
      <ErrorBoundary>
        <Thrower />
      </ErrorBoundary>,
    );
    expect(await screen.findByRole("heading", { name: "出现错误" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "重试此视图" })).toBeInTheDocument();
    errorSpy.mockRestore();
  });

  it("renders the custom preset label from i18n, not the hardcoded preset entry (zh)", () => {
    localStorage.setItem("chemapp-lang", "zh");
    renderWithLang(<SettingsPanel />);
    const select = screen.getByLabelText("模型选择");
    expect(within(select).getByRole("option", { name: "— 其他（自定义） —" })).toBeInTheDocument();
    expect(screen.queryByText("其他 (自定义)")).not.toBeInTheDocument();
  });

  it("renders the custom preset label from i18n (en)", () => {
    renderWithLang(<SettingsPanel />);
    const select = screen.getByLabelText("Model preset");
    expect(within(select).getByRole("option", { name: "— Other (custom) —" })).toBeInTheDocument();
  });

  it("renders CompareView points and the stokes-only path through i18n (zh)", async () => {
    localStorage.setItem("chemapp-lang", "zh");
    vi.mocked(compareSpectra).mockResolvedValue({
      id1: "f1",
      id2: "f2",
      technique1: "Fluorescence",
      technique2: "Fluorescence",
      shared: { points1: 256, points2: 128 },
      stokes: {
        stokes_shift_nm: 12.5,
        stokes_shift_cm1: 800,
        excitation_peak_nm: 350,
        emission_peak_nm: 362.5,
      },
    });
    renderWithLang(
      <CompareView
        spectra={[
          { id: "f1", technique: "Fluorescence", points: 10, name: "Ex", has_result: true, summary: "" },
          { id: "f2", technique: "Fluorescence", points: 20, name: "Em", has_result: true, summary: "" },
        ]}
      />,
    );
    const [first, second] = screen.getAllByRole("combobox");
    fireEvent.change(first, { target: { value: "f1" } });
    fireEvent.change(second, { target: { value: "f2" } });
    fireEvent.click(screen.getByRole("button", { name: "对比" }));

    expect(await screen.findByText("点数: 256 vs 128")).toBeInTheDocument();
    expect(screen.getByText(/Δλ = 12\.5 nm/)).toBeInTheDocument();
    // The stokes-only lightweight path must not fabricate an analysis result
    // (the fake object used to render a stray "类型" metric).
    expect(screen.queryByText("类型")).not.toBeInTheDocument();
  });

  it("renders the FluorescencePanel stokes-only path without a result", () => {
    renderWithLang(
      <FluorescencePanel
        stokes={{
          stokes_shift_nm: 12.5,
          stokes_shift_cm1: 800,
          excitation_peak_nm: 350,
          emission_peak_nm: 362.5,
        }}
      />,
    );
    expect(screen.getByRole("heading", { name: "Paired Stokes Shift" })).toBeInTheDocument();
    expect(screen.getByText(/Δλ = 12\.5 nm/)).toBeInTheDocument();
  });

  it("renders nothing for the FluorescencePanel stokes-only path without stokes", () => {
    const { container } = renderWithLang(<FluorescencePanel />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders InferencePanel export buttons and table headers through i18n (en)", async () => {
    renderWithLang(
      <InferencePanel
        spectra={[
          { id: "h1", technique: "HPLC", points: 10, name: "H1", has_result: true, summary: "" },
          { id: "h2", technique: "HPLC", points: 20, name: "H2", has_result: true, summary: "" },
        ]}
      />,
    );

    expect(screen.getByRole("button", { name: "Markdown" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "HTML" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Word" })).toBeInTheDocument();

    for (const checkbox of screen.getAllByRole("checkbox")) {
      fireEvent.click(checkbox);
    }
    expect(screen.getByRole("button", { name: "HPLC CSV" })).toBeInTheDocument();

    vi.mocked(getBatchQuality).mockResolvedValue({ items: [], counts: { good: 1, review: 2, poor: 3 } });
    fireEvent.click(screen.getByRole("button", { name: "Quality overview" }));
    expect(await screen.findByText("Good 1 · Needs review 2 · Poor 3")).toBeInTheDocument();

    vi.mocked(compareHplcBatch).mockResolvedValue({
      reference_id: "h1",
      channel: "ch1",
      rt_tolerance: 0.08,
      rows: [
        {
          peak_index: 0,
          reference_rt: 1.5,
          reference_area: 100,
          reference_type: "analyte",
          name: "p1",
          samples: [
            { id: "h2", name: "H2", matched: true, rt_shift: 0.01, area: 90, area_percent: 90 },
          ],
        },
      ],
      drift: [
        {
          id: "h2",
          name: "H2",
          channel: "ch1",
          mean_rt_shift: 0.01,
          max_abs_rt_shift: 0.02,
          matched_peaks: 1,
          total_area: 90,
        },
      ],
    });
    fireEvent.click(screen.getByRole("button", { name: "Match HPLC peaks" }));

    expect(await screen.findByRole("columnheader", { name: "Total area" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Ref tR" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Type" })).toBeInTheDocument();
  });
});
