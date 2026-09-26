import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LangProvider } from "../i18n/LangContext";
import {
  executeAIAction,
  getResult,
  previewAIAction,
  suggestAIActions,
} from "../services/api";
import AIChatSidebar from "./AIChatSidebar";
import { withExpectedRevision } from "./AIChatSidebar";

vi.mock("../services/api", () => ({
  API_BASE: "http://test",
  executeAIAction: vi.fn(),
  previewAIAction: vi.fn(),
  suggestAIActions: vi.fn(),
  generateExperimentAgentReport: vi.fn(),
  downloadExperimentAgentDocx: vi.fn(),
  downloadExperimentAgentMarkdown: vi.fn(),
  getResult: vi.fn(),
}));

describe("withExpectedRevision", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem("chemapp-lang", "en");
  });

  it("attaches the current result revision for spectrum actions", async () => {
    vi.mocked(getResult).mockResolvedValue({
      technique: "NMR",
      result_revision: 7,
      peaks: [],
      metrics: {},
      summary: "ok",
    });

    const action = await withExpectedRevision({
      name: "update_nmr_integral_range",
      args: { spectrum_id: "spectrum-1" },
    });

    expect(action.args.expected_revision).toBe(7);
  });

  it("leaves non-destructive actions unchanged", async () => {
    const action = await withExpectedRevision({
      name: "run_cross_inference",
      args: { ids: ["spectrum-1"] },
    });

    expect(action.args.expected_revision).toBeUndefined();
    expect(getResult).not.toHaveBeenCalled();
  });

  it("fails closed when a destructive action cannot bind a revision", async () => {
    vi.mocked(getResult).mockRejectedValue(new Error("offline"));

    await expect(withExpectedRevision({
      name: "delete_peak",
      args: { spectrum_id: "spectrum-1", index: 0 },
    })).rejects.toThrow("offline");
  });

  it("executes exactly the revision and token returned by preview, then refreshes", async () => {
    vi.mocked(getResult).mockResolvedValue({
      technique: "NMR",
      result_revision: 7,
      peaks: [],
      metrics: {},
      summary: "ok",
    });
    vi.mocked(suggestAIActions).mockResolvedValue({
      suggestions: [{
        title: "Delete peak",
        reason: "test",
        severity: "review",
        action: {
          name: "delete_peak",
          args: { spectrum_id: "spectrum-1", index: 0 },
        },
      }],
    });
    vi.mocked(previewAIAction).mockResolvedValue({
      action: "delete_peak",
      preview: { target: "peak 1" },
      expected_revision: 7,
      preview_token: "preview-token-7",
    });
    vi.mocked(executeAIAction).mockResolvedValue({
      action: "delete_peak",
      result: { ok: true, result_revision: 8 },
    });
    const onSpectrumMutated = vi.fn();

    render(
      <LangProvider>
        <AIChatSidebar
          spectra={[{
            id: "spectrum-1",
            technique: "NMR",
            points: 10,
            name: "Sample A",
            has_result: true,
            summary: "",
            result_revision: 7,
          }]}
          onSpectrumMutated={onSpectrumMutated}
        />
      </LangProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Expand AI assistant" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select spectrum Sample A" }));
    fireEvent.click(screen.getByRole("button", { name: "Suggestions" }));
    fireEvent.click(await screen.findByRole("button", { name: "Preview" }));

    await waitFor(() => expect(previewAIAction).toHaveBeenCalledWith({
      name: "delete_peak",
      args: { spectrum_id: "spectrum-1", index: 0, expected_revision: 7 },
      preview_token: undefined,
    }));

    fireEvent.click(await screen.findByRole("button", { name: "Execute delete_peak" }));
    await waitFor(() => expect(executeAIAction).toHaveBeenCalledWith({
      name: "delete_peak",
      args: { spectrum_id: "spectrum-1", index: 0, expected_revision: 7 },
      preview_token: "preview-token-7",
    }));
    expect(onSpectrumMutated).toHaveBeenCalledWith("spectrum-1");
  });

  it("blocks a destructive preview for the spectrum with unsaved edits", async () => {
    vi.mocked(suggestAIActions).mockResolvedValue({
      suggestions: [{
        title: "Delete peak",
        reason: "test",
        severity: "review",
        action: {
          name: "delete_peak",
          args: { spectrum_id: "spectrum-1", index: 0 },
        },
      }],
    });

    render(
      <LangProvider>
        <AIChatSidebar
          spectra={[{
            id: "spectrum-1",
            technique: "NMR",
            points: 10,
            name: "Sample A",
            has_result: true,
            summary: "",
          }]}
          protectedSpectrumId="spectrum-1"
        />
      </LangProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Expand AI assistant" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select spectrum Sample A" }));
    fireEvent.click(screen.getByRole("button", { name: "Suggestions" }));
    fireEvent.click(await screen.findByRole("button", { name: "Preview" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("unsaved edits");
    expect(previewAIAction).not.toHaveBeenCalled();
  });
});

describe("AIChatSidebar stream errors", () => {
  const spectrum = {
    id: "spectrum-1",
    technique: "NMR",
    points: 10,
    name: "Sample A",
    has_result: true,
    summary: "",
  };

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem("chemapp-lang", "en");
    localStorage.setItem("chemapp-ai-settings", JSON.stringify({ allowDataSharing: true }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const askQuestion = () => {
    fireEvent.click(screen.getByRole("button", { name: "Expand AI assistant" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select spectrum Sample A" }));
    fireEvent.change(screen.getByLabelText("Ask a question and press Enter..."), {
      target: { value: "What compound is this?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
  };

  it("detects an aborted stream by DOMException name, not message text", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new DOMException("cancelled by test", "AbortError"));
    vi.stubGlobal("fetch", fetchMock);
    render(
      <LangProvider>
        <AIChatSidebar spectra={[spectrum]} />
      </LangProvider>,
    );
    askQuestion();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    // Once loading finishes the Send button remounts; an abort must not surface an alert.
    await waitFor(() => expect(screen.getByRole("button", { name: "Send" })).toBeInTheDocument());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("still surfaces non-abort stream failures", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error("network down"));
    vi.stubGlobal("fetch", fetchMock);
    render(
      <LangProvider>
        <AIChatSidebar spectra={[spectrum]} />
      </LangProvider>,
    );
    askQuestion();
    expect(await screen.findByRole("alert")).toHaveTextContent("network down");
  });
});
