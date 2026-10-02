import { StrictMode } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LangProvider } from "../i18n/LangContext";
import type { SpectrumData } from "../types/spectrum";
import SpectrumViewer from "./SpectrumViewer";

const plotly = vi.hoisted(() => ({
  react: vi.fn<(element: HTMLElement, traces: unknown, layout: unknown, config: unknown) => Promise<void>>(),
  relayout: vi.fn<(element: HTMLElement, layout: unknown) => Promise<void>>(),
  purge: vi.fn<(element: HTMLElement) => void>(),
}));
vi.mock("../plotlyCustom", () => ({ default: plotly }));

const spectrum: SpectrumData = {
  id: "first", technique: "UV-Vis", x_data: [200, 250, 300], y_data: [0, 1, 0],
  x_label: "Wavelength", x_unit: "nm", y_label: "Absorbance", y_unit: "AU",
  parameters: {}, metadata: { name: "First sample" }, peaks: [], source_file: "first.csv",
};
const nextSpectrum: SpectrumData = { ...spectrum, id: "second", y_data: [0, 2, 0] };
const view = (data = spectrum) => <LangProvider><SpectrumViewer spectrum={data} /></LangProvider>;
const resetButton = () => screen.getByRole("button", { name: "Restore full spectrum" });

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

describe("spectrum plot lifecycle", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    localStorage.setItem("chemapp-lang", "en");
    plotly.react.mockResolvedValue();
    plotly.relayout.mockResolvedValue();
    vi.spyOn(console, "error").mockImplementation(() => undefined);
  });

  it("announces loading and disables reset until the plot is ready", async () => {
    const pending = deferred<void>();
    plotly.react.mockReturnValue(pending.promise);
    render(view());
    expect(screen.getByRole("status")).toHaveTextContent("Drawing spectrum");
    expect(resetButton()).toBeDisabled();
    await waitFor(() => expect(plotly.react).toHaveBeenCalledOnce());
    const element = plotly.react.mock.calls[0][0];
    expect(element.isConnected).toBe(true);
    await act(async () => { pending.resolve(); await pending.promise; });
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(resetButton()).toBeEnabled();
    expect(screen.getByRole("img", { name: "Interactive UV-Vis spectrum" })).toHaveAttribute("aria-busy", "false");
    expect(plotly.react.mock.calls[0][1]).toEqual([expect.objectContaining({ x: [200, 250, 300], y: [0, 1, 0], type: "scatter" })]);
  });

  it("shows a recoverable render failure and retries in a fresh plot element", async () => {
    plotly.react.mockRejectedValueOnce(new Error("Renderer unavailable"));
    render(view());
    expect(await screen.findByRole("alert")).toHaveTextContent("The spectrum could not be displayed");
    expect(resetButton()).toBeDisabled();
    const failedElement = plotly.react.mock.calls[0][0];
    expect(plotly.purge).toHaveBeenCalledWith(failedElement);
    fireEvent.click(screen.getByRole("button", { name: "Retry plot" }));
    await waitFor(() => expect(resetButton()).toBeEnabled());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(plotly.react).toHaveBeenCalledTimes(2);
    expect(plotly.react.mock.calls[1][0]).not.toBe(failedElement);
    expect(failedElement.isConnected).toBe(false);
  });

  it("keeps a late render isolated from a newly selected spectrum", async () => {
    const oldRender = deferred<void>();
    const newRender = deferred<void>();
    plotly.react.mockReturnValueOnce(oldRender.promise).mockReturnValueOnce(newRender.promise);
    const mounted = render(view());
    await waitFor(() => expect(plotly.react).toHaveBeenCalledTimes(1));
    const oldElement = plotly.react.mock.calls[0][0];
    mounted.rerender(view(nextSpectrum));
    await waitFor(() => expect(plotly.react).toHaveBeenCalledTimes(2));
    const newElement = plotly.react.mock.calls[1][0];
    expect(oldElement.isConnected).toBe(false);
    expect(newElement.isConnected).toBe(true);
    await act(async () => { oldRender.resolve(); await oldRender.promise; });
    expect(plotly.purge).toHaveBeenCalledWith(oldElement);
    expect(plotly.purge.mock.calls.some(([element]) => element === newElement)).toBe(false);
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(resetButton()).toBeDisabled();
    await act(async () => { newRender.resolve(); await newRender.promise; });
    expect(resetButton()).toBeEnabled();
  });

  it("ignores late failures from the previous spectrum", async () => {
    const oldRender = deferred<void>();
    plotly.react.mockReturnValueOnce(oldRender.promise);
    const mounted = render(view());
    await waitFor(() => expect(plotly.react).toHaveBeenCalledOnce());
    mounted.rerender(view(nextSpectrum));
    await waitFor(() => expect(resetButton()).toBeEnabled());
    await act(async () => {
      oldRender.reject(new Error("Old render failed"));
      await oldRender.promise.catch(() => undefined);
    });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(resetButton()).toBeEnabled();
  });

  it("purges an unfinished plot after it settles following unmount", async () => {
    const pending = deferred<void>();
    plotly.react.mockReturnValue(pending.promise);
    const mounted = render(view());
    await waitFor(() => expect(plotly.react).toHaveBeenCalledOnce());
    const element = plotly.react.mock.calls[0][0];
    mounted.unmount();
    expect(element.isConnected).toBe(false);
    expect(plotly.purge).not.toHaveBeenCalled();
    await act(async () => { pending.resolve(); await pending.promise; });
    expect(plotly.purge).toHaveBeenCalledExactlyOnceWith(element);
  });

  it("purges a completed plot when unmounted", async () => {
    const mounted = render(view());
    await waitFor(() => expect(resetButton()).toBeEnabled());
    const element = plotly.react.mock.calls[0][0];
    mounted.unmount();
    expect(plotly.purge).toHaveBeenCalledExactlyOnceWith(element);
  });

  it("does not double-render the cancelled StrictMode initialization", async () => {
    render(<StrictMode>{view()}</StrictMode>);
    await waitFor(() => expect(resetButton()).toBeEnabled());
    expect(plotly.react).toHaveBeenCalledOnce();
    expect(plotly.react.mock.calls[0][0].isConnected).toBe(true);
  });

  it("deduplicates reset clicks and exposes reset failures as retryable", async () => {
    const pending = deferred<void>();
    plotly.relayout.mockReturnValue(pending.promise);
    render(view());
    await waitFor(() => expect(resetButton()).toBeEnabled());
    fireEvent.click(resetButton());
    fireEvent.click(resetButton());
    expect(plotly.relayout).toHaveBeenCalledOnce();
    expect(resetButton()).toBeDisabled();
    expect(plotly.relayout.mock.calls[0][1]).toEqual({ "xaxis.autorange": true, "yaxis.autorange": true });
    await act(async () => {
      pending.reject(new Error("Reset failed"));
      await pending.promise.catch(() => undefined);
    });
    expect(screen.getByRole("alert")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry plot" }));
    await waitFor(() => expect(resetButton()).toBeEnabled());
  });

  it("does not apply a stale reset failure to the newly selected spectrum", async () => {
    const pending = deferred<void>();
    plotly.relayout.mockReturnValue(pending.promise);
    const mounted = render(view());
    await waitFor(() => expect(resetButton()).toBeEnabled());
    fireEvent.click(resetButton());
    mounted.rerender(view(nextSpectrum));
    await waitFor(() => expect(resetButton()).toBeEnabled());
    await act(async () => {
      pending.reject(new Error("Old reset failed"));
      await pending.promise.catch(() => undefined);
    });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(resetButton()).toBeEnabled();
  });

  it("only accepts integration events belonging to the active plot", async () => {
    const callbacks = new Map<HTMLElement, (event: unknown) => void>();
    plotly.react.mockImplementation((element) => {
      Object.assign(element, {
        on: (_name: string, callback: (event: unknown) => void) => { callbacks.set(element, callback); },
        removeAllListeners: vi.fn(),
      });
      return Promise.resolve();
    });
    const selected = vi.fn();
    const integrationSelection = { label: "A", regions: [] };
    const selectedView = (data: SpectrumData) => (
      <LangProvider><SpectrumViewer spectrum={data} integrationSelection={integrationSelection} onIntegrationRangeSelected={selected} /></LangProvider>
    );
    const mounted = render(selectedView(spectrum));
    await waitFor(() => expect(resetButton()).toBeEnabled());
    const first = plotly.react.mock.calls[0][0];
    const firstCallback = callbacks.get(first)!;
    firstCallback({ range: { x: [250, 200] } });
    expect(selected).toHaveBeenLastCalledWith(200, 250);
    selected.mockClear();
    mounted.rerender(selectedView(nextSpectrum));
    await waitFor(() => expect(resetButton()).toBeEnabled());
    firstCallback({ range: { x: [200, 300] } });
    expect(selected).not.toHaveBeenCalled();
    callbacks.get(plotly.react.mock.calls[1][0])!({ range: { x: [240, 280] } });
    expect(selected).toHaveBeenCalledExactlyOnceWith(240, 280);
  });
});
