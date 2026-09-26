import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { LangProvider } from "../i18n/LangContext";
import type { AnalysisOptions } from "../types/spectrum";
import AnalysisControls from "./AnalysisControls";

const renderControls = (value: AnalysisOptions, onChange = vi.fn()) => {
  render(
    <LangProvider>
      <AnalysisControls technique="NMR" value={value} onChange={onChange} />
    </LangProvider>,
  );
  return onChange;
};

describe("AnalysisControls numeric inputs", () => {
  beforeEach(() => {
    localStorage.setItem("chemapp-lang", "en");
  });

  it("removes the override when a numeric field is cleared", () => {
    const onChange = renderControls({ solvent_tolerance_ppm: 0.04 });
    fireEvent.change(screen.getByLabelText("Solvent tolerance (ppm)"), { target: { value: "" } });
    expect(onChange).toHaveBeenCalledWith({});
  });

  it("writes parsed finite numbers while typing", () => {
    const onChange = renderControls({});
    fireEvent.change(screen.getByLabelText("Solvent tolerance (ppm)"), { target: { value: "0.08" } });
    expect(onChange).toHaveBeenCalledWith({ solvent_tolerance_ppm: 0.08 });
  });

  it("keeps other overrides when one field changes", () => {
    const onChange = renderControls({ noise_factor: 3, solvent_tolerance_ppm: 0.04 });
    fireEvent.change(screen.getByLabelText("Solvent tolerance (ppm)"), { target: { value: "0.08" } });
    expect(onChange).toHaveBeenCalledWith({ noise_factor: 3, solvent_tolerance_ppm: 0.08 });
  });
});
