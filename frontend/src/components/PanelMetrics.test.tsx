import type { ReactElement } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { LangProvider } from "../i18n/LangContext";
import type { AnalysisResult } from "../types/spectrum";
import CompareView from "./CompareView";
import ElectrochemPanel from "./ElectrochemPanel";
import FluorescencePanel from "./FluorescencePanel";
import HPLCPanel from "./HPLCPanel";
import MLTrainingPanel from "./MLTrainingPanel";
import NMRPanel from "./NMRPanel";
import XRDSPanel from "./XRDSPanel";
import { compareSpectra, getMLStatus } from "../services/api";

vi.mock("../services/api", () => ({
  analyzeSpectrum: vi.fn(),
  ApiError: class ApiError extends Error {
    status = 0;
  },
  compareSpectra: vi.fn(),
  getMLStatus: vi.fn(),
}));

const PLACEHOLDER = "—";

const baseResult = (technique: string, metrics: Record<string, unknown>): AnalysisResult => ({
  technique,
  peaks: [],
  metrics,
  summary: "summary",
});

const renderWithLang = (ui: ReactElement) => render(<LangProvider>{ui}</LangProvider>);

const metricValueByLabel = (label: string): string => {
  const metric = screen.getByText(label).closest(".metric");
  expect(metric).not.toBeNull();
  return (metric as HTMLElement).querySelector(".metric-value")?.textContent ?? "";
};

describe("panel metric guards", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem("chemapp-lang", "en");
  });

  it("renders placeholders for malformed NMR metrics instead of crashing", () => {
    renderWithLang(<NMRPanel result={baseResult("NMR", { n_peaks: "many", n_multiplets: null })} />);
    expect(metricValueByLabel("Peaks")).toBe(PLACEHOLDER);
    expect(metricValueByLabel("Multiplets")).toBe(PLACEHOLDER);
    expect(metricValueByLabel("Noise Level")).toBe(PLACEHOLDER);
    expect(metricValueByLabel("Total Integral")).toBe(PLACEHOLDER);
  });

  it("renders valid NMR metrics unchanged", () => {
    const result: AnalysisResult = {
      ...baseResult("NMR", { n_peaks: 5, n_multiplets: 2 }),
      noise_level: 12.34,
      total_integral: 5678,
    };
    renderWithLang(<NMRPanel result={result} />);
    expect(metricValueByLabel("Peaks")).toBe("5");
    expect(metricValueByLabel("Multiplets")).toBe("2");
    expect(metricValueByLabel("Noise Level")).toBe((12.34).toExponential(2));
    expect(metricValueByLabel("Total Integral")).toBe((5678).toExponential(2));
  });

  it("renders placeholders for malformed HPLC metrics", () => {
    renderWithLang(<HPLCPanel result={baseResult("HPLC", {
      n_peaks: "x",
      n_channels: {},
      time_range_min: "bad",
      integration_source: 5,
    })} />);
    expect(metricValueByLabel("Total peaks")).toBe(PLACEHOLDER);
    expect(metricValueByLabel("Channels")).toBe(PLACEHOLDER);
    expect(metricValueByLabel("Time range")).toBe(PLACEHOLDER);
    expect(screen.getByText(/ChemApp computed/)).toBeInTheDocument();
  });

  it("renders valid HPLC metrics unchanged", () => {
    renderWithLang(<HPLCPanel result={baseResult("HPLC", {
      n_peaks: 3,
      n_channels: 2,
      time_range_min: [0, 12],
      integration_source: "instrument_record",
    })} />);
    expect(metricValueByLabel("Total peaks")).toBe("3");
    expect(metricValueByLabel("Channels")).toBe("2");
    expect(metricValueByLabel("Time range")).toBe("0 – 12 min");
    expect(screen.getByText(/raw instrument record/)).toBeInTheDocument();
  });

  it("renders placeholders for malformed XRD metrics", () => {
    renderWithLang(<XRDSPanel result={baseResult("XRD", {
      n_peaks: {},
      wavelength_a: "N/A",
      two_theta_range: [10],
      max_intensity: null,
    })} />);
    expect(metricValueByLabel("Peaks")).toBe(PLACEHOLDER);
    expect(metricValueByLabel("Wavelength")).toBe(PLACEHOLDER);
    expect(metricValueByLabel("2θ Range")).toBe(PLACEHOLDER);
    expect(metricValueByLabel("Max Intensity")).toBe(PLACEHOLDER);
  });

  it("renders valid XRD metrics unchanged", () => {
    renderWithLang(<XRDSPanel result={baseResult("XRD", {
      n_peaks: 4,
      wavelength_a: 1.5406,
      two_theta_range: [10, 80],
      max_intensity: 999,
    })} />);
    expect(metricValueByLabel("Peaks")).toBe("4");
    expect(metricValueByLabel("Wavelength")).toBe("1.5406 Å");
    expect(metricValueByLabel("2θ Range")).toBe("10° – 80°");
    expect(metricValueByLabel("Max Intensity")).toBe((999).toExponential(2));
  });

  it("renders placeholders for malformed CV metrics", () => {
    renderWithLang(<ElectrochemPanel result={baseResult("ElectroChem", {
      sub_type: "CV",
      scan_rate_v_s: "fast",
      ep_anodic_v: "high",
    })} />);
    expect(screen.getByRole("heading", { name: `CV Analysis (${PLACEHOLDER} V/s)` })).toBeInTheDocument();
    expect(metricValueByLabel("Ep, a (V)")).toBe(PLACEHOLDER);
  });

  it("renders valid CV metrics unchanged", () => {
    renderWithLang(<ElectrochemPanel result={baseResult("ElectroChem", {
      sub_type: "CV",
      scan_rate_v_s: 0.1,
      ep_anodic_v: 0.5,
      delta_ep_v: 0.07,
      ip_anodic_a: 0.0003,
      ip_ratio: 1.07,
    })} />);
    expect(screen.getByRole("heading", { name: "CV Analysis (0.1 V/s)" })).toBeInTheDocument();
    expect(metricValueByLabel("Ep, a (V)")).toBe("0.5000");
    expect(metricValueByLabel("ΔEp (mV)")).toBe("70.0");
    expect(metricValueByLabel("ip, a (μA)")).toBe("300.00");
    expect(metricValueByLabel("ip, a / ip, c")).toBe("1.070");
  });

  it("renders placeholders for malformed EIS metrics", () => {
    renderWithLang(<ElectrochemPanel result={baseResult("ElectroChem", {
      sub_type: "EIS",
      rs_ohm: "low",
      rct_ohm: null,
      z_range_real: "wide",
    })} />);
    expect(screen.getByRole("heading", { name: "EIS Analysis" })).toBeInTheDocument();
    expect(metricValueByLabel("Rs (Ω)")).toBe(PLACEHOLDER);
    expect(metricValueByLabel("Rct (Ω)")).toBe(PLACEHOLDER);
    expect(metricValueByLabel("Z' range (Ω)")).toBe(PLACEHOLDER);
  });

  it("renders a placeholder for a malformed fluorescence sub type", () => {
    renderWithLang(<FluorescencePanel result={baseResult("Fluorescence", { sub_type: 42 })} />);
    expect(metricValueByLabel("Type")).toBe(PLACEHOLDER);
  });

  it("renders a valid fluorescence sub type unchanged", () => {
    renderWithLang(<FluorescencePanel result={baseResult("Fluorescence", { sub_type: "emission" })} />);
    expect(metricValueByLabel("Type")).toBe("emission");
  });

  it("falls back to untrained/idle for malformed ML status fields", async () => {
    vi.mocked(getMLStatus).mockResolvedValue({
      model_loaded: "yes",
      training: "corrupted",
      n_classes: 3,
      device: "cpu",
    });
    renderWithLang(<MLTrainingPanel />);
    await waitFor(() => expect(metricValueByLabel("Model status")).toBe("Not trained"));
    expect(metricValueByLabel("Training status")).toBe("idle");
    expect(metricValueByLabel("Device")).toBe("cpu");
  });

  it("renders valid ML status fields unchanged", async () => {
    vi.mocked(getMLStatus).mockResolvedValue({
      model_loaded: true,
      training: { status: "running" },
      n_classes: 3,
      device: "cuda",
    });
    renderWithLang(<MLTrainingPanel />);
    await waitFor(() => expect(metricValueByLabel("Model status")).toBe("Trained"));
    expect(metricValueByLabel("Training status")).toBe("running");
  });

  it("renders placeholders for malformed comparison point counts", async () => {
    vi.mocked(compareSpectra).mockResolvedValue({
      id1: "s1",
      id2: "s2",
      technique1: "NMR",
      technique2: "HPLC",
      shared: { points1: "lots", points2: 128 },
    });
    renderWithLang(<CompareView spectra={[
      { id: "s1", technique: "NMR", points: 10, name: "Sample A", has_result: true, summary: "" },
      { id: "s2", technique: "HPLC", points: 20, name: "Sample B", has_result: true, summary: "" },
    ]} />);
    const [first, second] = screen.getAllByRole("combobox");
    fireEvent.change(first, { target: { value: "s1" } });
    fireEvent.change(second, { target: { value: "s2" } });
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    expect(await screen.findByText(`Points: ${PLACEHOLDER} vs 128`)).toBeInTheDocument();
  });

  it("renders valid comparison point counts unchanged", async () => {
    vi.mocked(compareSpectra).mockResolvedValue({
      id1: "s1",
      id2: "s2",
      technique1: "NMR",
      technique2: "HPLC",
      shared: { points1: 256, points2: 128 },
    });
    renderWithLang(<CompareView spectra={[
      { id: "s1", technique: "NMR", points: 10, name: "Sample A", has_result: true, summary: "" },
      { id: "s2", technique: "HPLC", points: 20, name: "Sample B", has_result: true, summary: "" },
    ]} />);
    const [first, second] = screen.getAllByRole("combobox");
    fireEvent.change(first, { target: { value: "s1" } });
    fireEvent.change(second, { target: { value: "s2" } });
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    expect(await screen.findByText("Points: 256 vs 128")).toBeInTheDocument();
  });
});
