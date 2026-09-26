import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LangProvider } from "../i18n/LangContext";
import MLTrainingPanel from "./MLTrainingPanel";
import { getMLStatus } from "../services/api";

vi.mock("../services/api", () => ({
  getMLStatus: vi.fn(),
}));

const mockedGetMLStatus = vi.mocked(getMLStatus);

describe("MLTrainingPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem("chemapp-lang", "en");
    mockedGetMLStatus.mockResolvedValue({});
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders the legacy-pipeline notice through i18n (en)", async () => {
    render(
      <LangProvider>
        <MLTrainingPanel />
      </LangProvider>,
    );
    expect(
      await screen.findByText(/Legacy \/api\/ml\/train and \/api\/ml\/download are removed/),
    ).toBeInTheDocument();
  });

  it("renders the legacy-pipeline notice through i18n (zh)", async () => {
    localStorage.setItem("chemapp-lang", "zh");
    render(
      <LangProvider>
        <MLTrainingPanel />
      </LangProvider>,
    );
    expect(
      await screen.findByText(/旧版 \/api\/ml\/train 与 \/api\/ml\/download 已移除/),
    ).toBeInTheDocument();
  });

  it("pauses polling while the tab is hidden and refreshes immediately on return", async () => {
    vi.useFakeTimers();
    let hidden = false;
    Object.defineProperty(document, "hidden", { configurable: true, get: () => hidden });

    render(
      <LangProvider>
        <MLTrainingPanel />
      </LangProvider>,
    );
    // Flush the initial fetch.
    await act(async () => {
      await Promise.resolve();
    });
    expect(mockedGetMLStatus).toHaveBeenCalledTimes(1);

    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(mockedGetMLStatus).toHaveBeenCalledTimes(2);
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(mockedGetMLStatus).toHaveBeenCalledTimes(3);

    // Background tab: polling stops.
    hidden = true;
    act(() => {
      document.dispatchEvent(new Event("visibilitychange"));
    });
    act(() => {
      vi.advanceTimersByTime(30000);
    });
    expect(mockedGetMLStatus).toHaveBeenCalledTimes(3);

    // Back in the foreground: one immediate refresh, then polling resumes.
    hidden = false;
    act(() => {
      document.dispatchEvent(new Event("visibilitychange"));
    });
    expect(mockedGetMLStatus).toHaveBeenCalledTimes(4);
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(mockedGetMLStatus).toHaveBeenCalledTimes(5);

    Reflect.deleteProperty(document, "hidden");
  });
});
