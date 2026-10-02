import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LangProvider } from "../i18n/LangContext";
import { translations, type Lang } from "../i18n/translations";
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

describe("AI drawer dismissal", () => {
  beforeEach(() => {
    localStorage.setItem("chemapp-lang", "en");
  });

  it("provides a close button inside the overlay and returns focus to its toggle", () => {
    render(<LangProvider><AIChatSidebar spectra={[]} /></LangProvider>);
    const toggle = screen.getByRole("button", { name: "Expand AI assistant" });
    fireEvent.click(toggle);
    const drawer = screen.getByRole("complementary", { name: "AI Assistant" });
    const close = within(drawer).getByRole("button", { name: "Collapse AI assistant" });
    expect(toggle).toHaveAttribute("aria-controls", drawer.id);
    expect(close).toHaveFocus();
    fireEvent.click(close);
    expect(screen.queryByRole("complementary", { name: "AI Assistant" })).not.toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(toggle).toHaveFocus();
  });

  it("dismisses from the keyboard without losing the conversation draft", () => {
    render(<LangProvider><AIChatSidebar spectra={[]} /></LangProvider>);
    const toggle = screen.getByRole("button", { name: "Expand AI assistant" });
    fireEvent.click(toggle);
    const input = screen.getByRole("textbox", { name: "Select spectra first" });
    fireEvent.change(input, { target: { value: "Draft question" } });
    fireEvent.keyDown(input, { key: "Escape" });
    expect(toggle).toHaveFocus();
    expect(screen.queryByRole("complementary", { name: "AI Assistant" })).not.toBeInTheDocument();
    fireEvent.click(toggle);
    expect(screen.getByRole("textbox", { name: "Select spectra first" })).toHaveValue("Draft question");
  });

  it("does not let a conversation confirmation's Escape close the drawer", () => {
    render(<LangProvider><AIChatSidebar spectra={[]} /></LangProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Expand AI assistant" }));
    fireEvent.click(screen.getByText("+", { selector: "button" }));
    fireEvent.click(screen.getAllByRole("button", { name: "Delete conversation New conversation" })[0]);
    fireEvent.keyDown(screen.getByRole("alertdialog"), { key: "Escape" });
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(screen.getByRole("complementary", { name: "AI Assistant" })).toBeInTheDocument();
  });
});

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

describe("AI undo protection", () => {
  const spectrum = {
    id: "spectrum-1",
    technique: "NMR",
    points: 10,
    name: "Sample A",
    has_result: true,
    summary: "",
    result_revision: 7,
  };
  const localizedCases = [
    {
      lang: "en",
      reason: "later_result_edit",
      expected: "The result was manually saved, restored, or reanalyzed after this AI action. Undo was blocked to protect those later changes; the current result is unchanged.",
    },
    {
      lang: "en",
      reason: "unverifiable_history",
      expected: "This AI action's history cannot be verified, so it cannot be safely undone. Undo was blocked to protect the current result; the current result is unchanged.",
    },
    {
      lang: "zh",
      reason: "later_result_edit",
      expected: "该 AI 动作之后，分析结果已被人工保存、恢复或重新分析。为保护后续修改，已阻止撤销；当前结果未改变。",
    },
    {
      lang: "zh",
      reason: "unverifiable_history",
      expected: "此 AI 动作的历史记录无法验证，不能安全撤销。为保护当前结果，已阻止撤销；当前结果未改变。",
    },
  ] as const;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getResult).mockResolvedValue({
      technique: "NMR",
      result_revision: 7,
      peaks: [],
      metrics: {},
      summary: "ok",
    });
    vi.mocked(previewAIAction).mockResolvedValue({
      action: "undo_last_ai_action",
      preview: { target: "previous AI action" },
      expected_revision: 7,
      preview_token: "undo-preview-7",
    });
  });

  const renderUndo = (lang: Lang, protectedSpectrumId: string | null = null) => {
    localStorage.setItem("chemapp-lang", lang);
    const t = translations[lang];
    const onSpectrumMutated = vi.fn();
    const rendered = render(
      <LangProvider>
        <AIChatSidebar
          spectra={[spectrum]}
          protectedSpectrumId={protectedSpectrumId}
          onSpectrumMutated={onSpectrumMutated}
        />
      </LangProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: t.ai.open }));
    fireEvent.click(screen.getByRole("checkbox", {
      name: t.list.selectAria.replace("{name}", spectrum.name),
    }));
    fireEvent.click(screen.getByRole("button", { name: t.ai.undo }));
    return { ...rendered, t, onSpectrumMutated };
  };

  describe.each(["ApiError", "Axios"] as const)("%s error detail", (shape) => {
    const blockedError = (reason: string) => {
      const detail = {
        code: "ai_undo_blocked",
        reason,
        message: "English server explanation",
        current_revision: 7,
      };
      return shape === "ApiError"
        ? Object.assign(new Error(detail.message), { name: "ApiError", status: 409, detail })
        : { response: { status: 409, data: { detail } } };
    };

    it.each(localizedCases)("localizes $reason in $lang without treating a blocked undo as committed", async ({ lang, reason, expected }) => {
      vi.mocked(executeAIAction).mockRejectedValue(blockedError(reason));
      const { t, onSpectrumMutated } = renderUndo(lang);
      const draft = screen.getByRole("textbox", { name: t.ai.placeholder });
      fireEvent.change(draft, { target: { value: "Keep my question" } });
      fireEvent.click(await screen.findByRole("button", { name: `${t.ai.execute} undo_last_ai_action` }));

      expect(await screen.findByRole("alert")).toHaveTextContent(expected);
      expect(executeAIAction).toHaveBeenCalledExactlyOnceWith({
        name: "undo_last_ai_action",
        args: { spectrum_id: "spectrum-1", expected_revision: 7 },
        preview_token: "undo-preview-7",
      });
      expect(onSpectrumMutated).not.toHaveBeenCalled();
      expect(screen.queryByText(new RegExp(t.ai.actionExecuted))).not.toBeInTheDocument();
      expect(screen.getByText(t.ai.actionConfirm)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: `${t.ai.execute} undo_last_ai_action` })).toBeEnabled();
      expect(draft).toHaveValue("Keep my question");
      fireEvent.click(screen.getByRole("button", { name: t.action.cancel }));
      expect(screen.queryByText(t.ai.actionConfirm)).not.toBeInTheDocument();
      expect(onSpectrumMutated).not.toHaveBeenCalled();
    });

    it.each(localizedCases)("localizes $reason in $lang if preview is blocked", async ({ lang, reason, expected }) => {
      vi.mocked(previewAIAction).mockRejectedValue(blockedError(reason));
      const { t, onSpectrumMutated } = renderUndo(lang);

      expect(await screen.findByRole("alert")).toHaveTextContent(expected);
      expect(screen.queryByText(t.ai.actionConfirm)).not.toBeInTheDocument();
      expect(executeAIAction).not.toHaveBeenCalled();
      expect(onSpectrumMutated).not.toHaveBeenCalled();
    });

    it.each([
      {
        lang: "en",
        sourceRevision: null,
        expected: "This historical version has no verifiable source spectrum revision. It remains available to view but cannot be restored.",
      },
      {
        lang: "en",
        sourceRevision: 1,
        expected: "This historical version belongs to a different spectrum revision and cannot be restored. Reanalyze the current spectrum.",
      },
      {
        lang: "zh",
        sourceRevision: null,
        expected: "此历史版本缺少可验证的原始谱图修订信息，仅供查看，无法恢复。",
      },
      {
        lang: "zh",
        sourceRevision: 1,
        expected: "此历史版本来自不同的谱图修订，无法恢复。请重新分析当前谱图。",
      },
    ] as const)("localizes blocked undo provenance (source $sourceRevision) in $lang", async ({ lang, sourceRevision, expected }) => {
      const detail = {
        code: "result_source_mismatch",
        source_spectrum_revision: sourceRevision,
        current_spectrum_revision: 2,
        message: "English source provenance explanation",
      };
      vi.mocked(executeAIAction).mockRejectedValue(shape === "ApiError"
        ? Object.assign(new Error(detail.message), { name: "ApiError", status: 409, detail })
        : { response: { status: 409, data: { detail } } });
      const { t, onSpectrumMutated } = renderUndo(lang);
      fireEvent.click(await screen.findByRole("button", { name: `${t.ai.execute} undo_last_ai_action` }));

      expect(await screen.findByRole("alert")).toHaveTextContent(expected);
      expect(onSpectrumMutated).not.toHaveBeenCalled();
      expect(screen.queryByText(new RegExp(t.ai.actionExecuted))).not.toBeInTheDocument();
      expect(screen.getByText(t.ai.actionConfirm)).toBeInTheDocument();
    });
  });

  it.each(["en", "zh"] as const)("still blocks unsaved drafts before undo preview in %s", async (lang) => {
    const { t, onSpectrumMutated } = renderUndo(lang, spectrum.id);

    expect(await screen.findByRole("alert")).toHaveTextContent(t.ai.unsavedActionBlocked);
    expect(getResult).not.toHaveBeenCalled();
    expect(previewAIAction).not.toHaveBeenCalled();
    expect(executeAIAction).not.toHaveBeenCalled();
    expect(onSpectrumMutated).not.toHaveBeenCalled();
  });

  it.each(["en", "zh"] as const)("still blocks drafts created between preview and execution in %s", async (lang) => {
    const { t, onSpectrumMutated, rerender } = renderUndo(lang);
    const execute = await screen.findByRole("button", { name: `${t.ai.execute} undo_last_ai_action` });
    rerender(
      <LangProvider>
        <AIChatSidebar
          spectra={[spectrum]}
          protectedSpectrumId={spectrum.id}
          onSpectrumMutated={onSpectrumMutated}
        />
      </LangProvider>,
    );
    fireEvent.click(execute);

    expect(await screen.findByRole("alert")).toHaveTextContent(t.ai.unsavedActionBlocked);
    expect(executeAIAction).not.toHaveBeenCalled();
    expect(onSpectrumMutated).not.toHaveBeenCalled();
    expect(screen.getByText(t.ai.actionConfirm)).toBeInTheDocument();
  });

  it("preserves ordinary action errors instead of labeling every conflict as a protected undo", async () => {
    vi.mocked(executeAIAction).mockRejectedValue(Object.assign(new Error("Result revision changed"), {
      status: 409,
      detail: { code: "revision_conflict", reason: "later_result_edit" },
    }));
    const { t, onSpectrumMutated } = renderUndo("en");
    fireEvent.click(await screen.findByRole("button", { name: `${t.ai.execute} undo_last_ai_action` }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Result revision changed");
    expect(onSpectrumMutated).not.toHaveBeenCalled();
    expect(screen.queryByText(new RegExp(t.ai.actionExecuted))).not.toBeInTheDocument();
  });

  it("still commits a safe undo, refreshes once, and dismisses its preview", async () => {
    vi.mocked(executeAIAction).mockResolvedValue({
      action: "undo_last_ai_action",
      result: { ok: true, result_revision: 8 },
    });
    const { t, onSpectrumMutated } = renderUndo("en");
    fireEvent.click(await screen.findByRole("button", { name: `${t.ai.execute} undo_last_ai_action` }));

    await waitFor(() => expect(onSpectrumMutated).toHaveBeenCalledExactlyOnceWith(spectrum.id));
    expect(await screen.findByText(new RegExp(t.ai.actionExecuted))).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByText(t.ai.actionConfirm)).not.toBeInTheDocument();
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
