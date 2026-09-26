import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LangProvider } from "../i18n/LangContext";
import SettingsPanel from "./SettingsPanel";

describe("SettingsPanel saved indicator", () => {
  beforeEach(() => {
    localStorage.setItem("chemapp-lang", "en");
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows then hides the saved indicator on its timer", () => {
    vi.useFakeTimers();
    render(
      <LangProvider>
        <SettingsPanel />
      </LangProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
    expect(screen.getByRole("button", { name: "Saved ✓" })).toBeInTheDocument();
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(screen.getByRole("button", { name: "Save settings" })).toBeInTheDocument();
  });

  it("clears the pending timer when unmounted", () => {
    vi.useFakeTimers();
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const view = render(
      <LangProvider>
        <SettingsPanel />
      </LangProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
    view.unmount();
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(errorSpy).not.toHaveBeenCalled();
    errorSpy.mockRestore();
  });
});
