import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { LangProvider } from "../i18n/LangContext";
import type { AnalysisResult } from "../types/spectrum";
import HPLCPanel from "./HPLCPanel";

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
}));

const result = (revision: number): AnalysisResult => ({
  technique: "HPLC",
  result_revision: revision,
  peaks: [],
  metrics: {
    n_peaks: 0,
    n_channels: 1,
    time_range_min: [0, 10],
    channel_peaks: {
      DAD1: {
        wavelength_nm: 254,
        color: "#06b6d4",
        peaks: [],
      },
    },
    integration_events: [
      { channel: "DAD1", start: 1, end: 2, mode: "force_bb" },
    ],
  },
  summary: "HPLC result",
});

describe("HPLC integration revision guard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem("chemapp-lang", "en");
  });

  it("keeps the draft bound to its original revision after a concurrent prop update", async () => {
    analyzeSpectrum.mockRejectedValue(new Error("revision conflict"));
    const onDirtyChange = vi.fn();
    const onResultChanged = vi.fn();
    const view = render(
      <LangProvider>
        <HPLCPanel
          result={result(2)}
          spectrumId="hplc-1"
          onResultChanged={onResultChanged}
          onDirtyChange={onDirtyChange}
        />
      </LangProvider>,
    );

    fireEvent.change(screen.getAllByRole("spinbutton")[0], {
      target: { value: "1.25" },
    });
    await waitFor(() => expect(onDirtyChange).toHaveBeenCalledWith(true));

    view.rerender(
      <LangProvider>
        <HPLCPanel
          result={result(3)}
          spectrumId="hplc-1"
          onResultChanged={onResultChanged}
          onDirtyChange={onDirtyChange}
        />
      </LangProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Apply events and reintegrate" }));

    await waitFor(() => expect(analyzeSpectrum).toHaveBeenCalledWith(
      "hplc-1",
      expect.objectContaining({
        integration_events: [
          { channel: "DAD1", start: 1.25, end: 2, mode: "force_bb" },
        ],
      }),
      undefined,
      2,
      false,
    ));
    expect(onResultChanged).not.toHaveBeenCalled();
    expect(onDirtyChange).toHaveBeenLastCalledWith(true);
  });

  it("synchronizes a clean editor to a newer result revision", async () => {
    analyzeSpectrum.mockResolvedValue(result(4));
    const onDirtyChange = vi.fn();
    const onResultChanged = vi.fn();
    const view = render(
      <LangProvider>
        <HPLCPanel
          result={result(2)}
          spectrumId="hplc-1"
          onResultChanged={onResultChanged}
          onDirtyChange={onDirtyChange}
        />
      </LangProvider>,
    );
    const next = result(3);
    next.metrics.integration_events = [
      { channel: "DAD1", start: 3.5, end: 4, mode: "force_vv" },
    ];

    view.rerender(
      <LangProvider>
        <HPLCPanel
          result={next}
          spectrumId="hplc-1"
          onResultChanged={onResultChanged}
          onDirtyChange={onDirtyChange}
        />
      </LangProvider>,
    );

    await waitFor(() => expect(screen.getAllByRole("spinbutton")[0]).toHaveValue(3.5));
    expect(onDirtyChange).toHaveBeenLastCalledWith(false);
    fireEvent.change(screen.getAllByRole("spinbutton")[0], {
      target: { value: "3.75" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Apply events and reintegrate" }));

    await waitFor(() => expect(analyzeSpectrum).toHaveBeenCalledWith(
      "hplc-1",
      expect.objectContaining({
        integration_events: [
          { channel: "DAD1", start: 3.75, end: 4, mode: "force_vv" },
        ],
      }),
      undefined,
      3,
      false,
    ));
  });

  it("clears only its own dirty source after a successful commit", async () => {
    analyzeSpectrum.mockResolvedValue({
      ...result(3),
      metrics: {
        ...result(3).metrics,
        integration_events: [
          { channel: "DAD1", start: 1.5, end: 2, mode: "force_bb" },
        ],
      },
    });
    const onDirtyChange = vi.fn();
    const onResultChanged = vi.fn();
    render(
      <LangProvider>
        <HPLCPanel
          result={result(2)}
          spectrumId="hplc-1"
          onResultChanged={onResultChanged}
          onDirtyChange={onDirtyChange}
        />
      </LangProvider>,
    );

    fireEvent.change(screen.getAllByRole("spinbutton")[0], {
      target: { value: "1.5" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Apply events and reintegrate" }));

    await waitFor(() => expect(onResultChanged).toHaveBeenCalled());
    await waitFor(() => expect(onDirtyChange).toHaveBeenLastCalledWith(false));
  });

  it("does not snap an event start to 0 when its input is cleared", () => {
    const onDirtyChange = vi.fn();
    render(
      <LangProvider>
        <HPLCPanel
          result={result(2)}
          spectrumId="hplc-1"
          onResultChanged={vi.fn()}
          onDirtyChange={onDirtyChange}
        />
      </LangProvider>,
    );
    const startInput = screen.getAllByRole("spinbutton")[0];
    fireEvent.change(startInput, { target: { value: "" } });
    expect(onDirtyChange).not.toHaveBeenCalledWith(true);
    expect(screen.getByRole("button", { name: "Apply events and reintegrate" })).toBeDisabled();
  });
});
