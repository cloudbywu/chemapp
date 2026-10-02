import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LangProvider } from "../i18n/LangContext";
import {
  cancelNmr2StructDownload,
  getNmr2StructWeights,
  startNmr2StructDownload,
  type ModelDownloadJob,
  type Nmr2StructWeights,
} from "../services/api";
import ModelAssetsPanel from "./ModelAssetsPanel";

vi.mock("../services/api", () => ({
  getNmr2StructWeights: vi.fn(),
  startNmr2StructDownload: vi.fn(),
  cancelNmr2StructDownload: vi.fn(),
}));

function inventory(): Nmr2StructWeights {
  return {
    source_repo: "https://github.com/MarklandGroup/NMR2Struct",
    revision: "a".repeat(40),
    storage_dir: "/home/chemist/.local/share/ChemApp/models/nmr2struct",
    assets: ["cnmr_only", "hnmr_only", "multitask"].map((id) => ({
      id: id as "cnmr_only" | "hnmr_only" | "multitask",
      filename: `${id}.pt`, size_bytes: 1073741824, sha256: "f".repeat(64),
      status: "missing", installed_bytes: 0, job: null,
    })),
  };
}
function job(status: ModelDownloadJob["status"] = "downloading"): ModelDownloadJob {
  return { id: "job-1", asset_id: "cnmr_only", status, downloaded_bytes: 536870912, total_bytes: 1073741824, error: null };
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((resolve_, reject_) => { resolve = resolve_; reject = reject_; });
  return { promise, resolve, reject };
}
const flush = () => act(async () => { await Promise.resolve(); });
const click = (name: string) => fireEvent.click(screen.getByRole("button", { name }));
const panel = () => render(<LangProvider><ModelAssetsPanel /></LangProvider>);

describe("official model weights", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.resetAllMocks();
    localStorage.setItem("chemapp-lang", "en");
    vi.mocked(getNmr2StructWeights).mockResolvedValue(inventory());
  });
  afterEach(() => vi.useRealTimers());

  it("shows verified source, sizes, checksums and local destination without starting a transfer", async () => {
    const data = inventory();
    data.assets[0].status = "ready";
    data.assets[0].installed_bytes = data.assets[0].size_bytes;
    vi.mocked(getNmr2StructWeights).mockResolvedValue(data);
    panel();
    await flush();
    expect(screen.getByRole("heading", { name: "Official NMR2Struct weights" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "MarklandGroup/NMR2Struct" })).toHaveAttribute("href", data.source_repo);
    expect(screen.getByText(data.storage_dir)).toBeInTheDocument();
    expect(screen.getAllByText("1.0 GiB")).toHaveLength(3);
    expect(screen.getAllByText("f".repeat(64))).toHaveLength(3);
    expect(screen.getByText("Installed and verified")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Download ¹³C only" })).not.toBeInTheDocument();
    expect(startNmr2StructDownload).not.toHaveBeenCalled();
    await act(() => vi.advanceTimersByTimeAsync(30000));
    expect(getNmr2StructWeights).toHaveBeenCalledTimes(1);
  });

  it("deduplicates repeated starts, locks other assets, polls progress, then refreshes availability", async () => {
    const start = deferred<ModelDownloadJob>();
    vi.mocked(startNmr2StructDownload).mockReturnValue(start.promise);
    panel();
    await flush();
    const button = screen.getByRole("button", { name: "Download ¹³C only" });
    act(() => { fireEvent.click(button); fireEvent.click(button); });
    expect(startNmr2StructDownload).toHaveBeenCalledExactlyOnceWith("cnmr_only", expect.any(AbortSignal));
    expect(screen.getByRole("button", { name: "Starting…" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Download ¹H only" })).toBeDisabled();
    const downloading = inventory();
    downloading.assets[0].job = job();
    vi.mocked(getNmr2StructWeights).mockResolvedValue(downloading);
    await act(async () => { start.resolve(job()); await start.promise; });
    expect(screen.getByRole("progressbar", { name: "¹³C only Downloading" })).toHaveAttribute("value", "50");
    const complete = inventory();
    complete.assets[0].status = "ready";
    complete.assets[0].job = job("completed");
    vi.mocked(getNmr2StructWeights).mockResolvedValue(complete);
    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(screen.getByText("Installed and verified")).toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Download ¹³C only" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download ¹H only" })).toBeEnabled();
    const calls = vi.mocked(getNmr2StructWeights).mock.calls.length;
    await act(() => vi.advanceTimersByTimeAsync(10000));
    expect(getNmr2StructWeights).toHaveBeenCalledTimes(calls);
  });

  it("cancels once, waits for the server to stop, and permits a fresh retry", async () => {
    const data = inventory();
    data.assets[0].job = job();
    vi.mocked(getNmr2StructWeights).mockResolvedValue(data);
    const cancel = deferred<ModelDownloadJob>();
    vi.mocked(cancelNmr2StructDownload).mockReturnValue(cancel.promise);
    panel();
    await flush();
    const button = screen.getByRole("button", { name: "Cancel download ¹³C only" });
    act(() => { fireEvent.click(button); fireEvent.click(button); });
    expect(cancelNmr2StructDownload).toHaveBeenCalledExactlyOnceWith("job-1", expect.any(AbortSignal));
    data.assets[0].job = job("cancelling");
    await act(async () => { cancel.resolve(job("cancelling")); await cancel.promise; });
    expect(screen.getByRole("button", { name: "Cancelling…" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Download ¹H only" })).toBeDisabled();
    const cancelled = inventory();
    cancelled.assets[0].job = job("cancelled");
    vi.mocked(getNmr2StructWeights).mockResolvedValue(cancelled);
    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(screen.getByText("Download cancelled")).toBeInTheDocument();
    vi.mocked(startNmr2StructDownload).mockResolvedValue({ ...job("queued"), id: "job-2" });
    click("Retry download ¹³C only");
    await flush();
    expect(startNmr2StructDownload).toHaveBeenCalledTimes(1);
  });

  it("retains action errors after successful inventory refresh and allows retry", async () => {
    vi.mocked(startNmr2StructDownload).mockRejectedValue(new Error("Administrator token required"));
    panel();
    await flush();
    click("Download ¹³C only");
    await flush();
    expect(screen.getByRole("alert")).toHaveTextContent("Administrator token required");
    expect(screen.getByRole("button", { name: "Download ¹³C only" })).toBeEnabled();
    expect(getNmr2StructWeights).toHaveBeenCalledTimes(2);
  });

  it("shows verification failures with a retry button and never claims installation", async () => {
    const data = inventory();
    data.assets[0].status = "invalid";
    data.assets[0].job = { ...job("error"), error: "SHA-256 verification failed" };
    vi.mocked(getNmr2StructWeights).mockResolvedValue(data);
    panel();
    await flush();
    expect(screen.getByRole("alert")).toHaveTextContent("SHA-256 verification failed");
    expect(screen.getByRole("button", { name: "Retry download ¹³C only" })).toBeEnabled();
    expect(screen.queryByText("Installed and verified")).not.toBeInTheDocument();
  });

  it("recovers an existing job after navigation without duplicate downloads", async () => {
    const data = inventory();
    data.assets[0].job = job();
    vi.mocked(getNmr2StructWeights).mockResolvedValue(data);
    const first = panel();
    await flush();
    const signal = vi.mocked(getNmr2StructWeights).mock.calls[0][0];
    first.unmount();
    expect(signal?.aborted).toBe(true);
    await act(() => vi.advanceTimersByTimeAsync(10000));
    expect(getNmr2StructWeights).toHaveBeenCalledTimes(1);
    panel();
    await flush();
    expect(screen.getByRole("button", { name: "Cancel download ¹³C only" })).toBeEnabled();
    expect(startNmr2StructDownload).not.toHaveBeenCalled();
    expect(cancelNmr2StructDownload).not.toHaveBeenCalled();
  });

  it("ignores a late read after manual refresh and keeps polls non-overlapping", async () => {
    const delayed = deferred<Nmr2StructWeights>();
    vi.mocked(getNmr2StructWeights).mockReturnValueOnce(delayed.promise);
    panel();
    await flush();
    await act(() => vi.advanceTimersByTimeAsync(20000));
    expect(getNmr2StructWeights).toHaveBeenCalledTimes(1);
    const data = inventory();
    data.assets[0].job = job();
    vi.mocked(getNmr2StructWeights).mockResolvedValue(data);
    click("Refresh model status");
    await flush();
    expect(vi.mocked(getNmr2StructWeights).mock.calls[0][0]?.aborted).toBe(true);
    await act(async () => { delayed.resolve(inventory()); await delayed.promise; });
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download ¹H only" })).toBeDisabled();
  });

  it("keeps observing after a connection interruption and blocks starts from stale inventory", async () => {
    const data = inventory();
    data.assets[0].job = job();
    vi.mocked(getNmr2StructWeights).mockResolvedValueOnce(data).mockRejectedValueOnce(new Error("Connection interrupted"));
    panel();
    await flush();
    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(screen.getByRole("alert")).toHaveTextContent("an existing download may still be running");
    expect(screen.getByRole("button", { name: "Download ¹H only" })).toBeDisabled();
    const cancelled = inventory();
    cancelled.assets[0].job = job("cancelled");
    vi.mocked(getNmr2StructWeights).mockResolvedValue(cancelled);
    await act(() => vi.advanceTimersByTimeAsync(2000));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry download ¹³C only" })).toBeEnabled();
  });

  it("aborts an in-flight start on unmount without sending cancellation", async () => {
    const start = deferred<ModelDownloadJob>();
    vi.mocked(startNmr2StructDownload).mockReturnValue(start.promise);
    const view = panel();
    await flush();
    click("Download ¹³C only");
    const signal = vi.mocked(startNmr2StructDownload).mock.calls[0][1];
    view.unmount();
    expect(signal?.aborted).toBe(true);
    await act(async () => { start.resolve(job()); await start.promise; });
    expect(cancelNmr2StructDownload).not.toHaveBeenCalled();
    expect(getNmr2StructWeights).toHaveBeenCalledTimes(1);
  });

  it("supports retrying initial inventory errors and Chinese labels", async () => {
    localStorage.setItem("chemapp-lang", "zh");
    vi.mocked(getNmr2StructWeights).mockRejectedValueOnce(new Error("Access required"));
    panel();
    await flush();
    expect(screen.getByRole("alert")).toHaveTextContent("Access required");
    click("刷新模型状态");
    await flush();
    expect(screen.getByRole("heading", { name: "NMR2Struct 官方模型权重" })).toBeInTheDocument();
    expect(within(screen.getByRole("article", { name: "仅 ¹³C" })).getByRole("button", { name: "下载 仅 ¹³C" })).toBeEnabled();
  });
});
