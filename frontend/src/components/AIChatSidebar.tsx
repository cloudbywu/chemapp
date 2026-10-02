import { useEffect, useId, useRef, useState } from "react";
import type { SpectrumListItem } from "../types/spectrum";
import { loadAISettings } from "../services/aiSettings";
import { useLang } from "../i18n/LangContext";
import type { TranslationSchema } from "../i18n/translations";
import ConfirmDialog from "./ConfirmDialog";
import SafeMarkdown from "./SafeMarkdown";
import { chemAppAuthHeaders } from "../services/authTokens";
import { displayText } from "../utils/number";
import {
  downloadExperimentAgentDocx,
  downloadExperimentAgentMarkdown,
  executeAIAction,
  generateExperimentAgentReport,
  getResult,
  previewAIAction,
  suggestAIActions,
  API_BASE,
  type AIActionRequest,
  type ExperimentAgentReport,
} from "../services/api";

interface ChatMessage {
  role: "user" | "ai";
  text: string;
  reasoning: string;
  timestamp: number;
}

function extractActions(text: string): AIActionRequest[] {
  const actions: AIActionRequest[] = [];
  const re = /```chemapp-action\s*([\s\S]*?)```/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    try {
      // Model-generated action block; the truthiness gate below is the only
      // validation, so assert the expected shape at the parse boundary.
      const parsed = JSON.parse(match[1].trim()) as { name?: string; args?: Record<string, unknown> };
      if (parsed?.name && parsed?.args) actions.push({ name: parsed.name, args: parsed.args });
    } catch {
      // Ignore malformed action suggestions; the model text remains visible.
    }
  }
  return actions;
}

const DESTRUCTIVE_AI_ACTIONS = new Set([
  "update_nmr_integral_range",
  "delete_peak",
  "hplc_reintegrate_peak",
  "nmr_rebuild_multiplets",
  "apply_hplc_integration_events",
  "nmr_phase_correct",
  "xrd_rietveld_refine",
  "undo_last_ai_action",
]);

export function isDestructiveAIAction(action: AIActionRequest): boolean {
  return DESTRUCTIVE_AI_ACTIONS.has(action.name);
}

function actionSpectrumId(action: AIActionRequest): string | null {
  const raw = action.args?.spectrum_id;
  return raw === undefined || raw === null || displayText(raw).trim() === ""
    ? null
    : displayText(raw);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function undoBlockedMessage(error: unknown, messages: TranslationSchema): string | null {
  const apiError = asRecord(error);
  const response = asRecord(apiError?.response);
  const data = asRecord(response?.data);
  // ApiError preserves the server detail; also accept an unwrapped Axios error.
  const detail = asRecord(apiError?.detail ?? data?.detail);
  if (detail?.code === "result_source_mismatch") {
    return detail.source_spectrum_revision === null
      ? messages.manual.historicalSourceUnknown
      : messages.manual.historicalSourceChanged;
  }
  if (detail?.code !== "ai_undo_blocked") return null;
  if (detail.reason === "later_result_edit") return messages.ai.undoBlockedLaterEdit;
  if (detail.reason === "unverifiable_history") return messages.ai.undoBlockedHistory;
  return null;
}

export async function withExpectedRevision(
  action: AIActionRequest,
): Promise<AIActionRequest> {
  if (!isDestructiveAIAction(action)) return action;
  const spectrumId = actionSpectrumId(action);
  if (!spectrumId) throw new Error("A destructive AI action requires one spectrum");
  const result = await getResult(spectrumId);
  const revision = result?.result_revision;
  if (typeof revision !== "number") {
    throw new Error("The current result revision is unavailable");
  }
  return {
    ...action,
    preview_token: undefined,
    args: { ...action.args, expected_revision: revision },
  };
}

interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
}

interface Props {
  spectra: SpectrumListItem[];
  protectedSpectrumId?: string | null;
  onSpectrumMutated?: (spectrumId: string) => void | Promise<void>;
}

const TECH_COLORS: Record<string, string> = {
  NMR: "#3b82f6", "UV-Vis": "#10b981", Fluorescence: "#f59e0b",
  XRD: "#8b5cf6", HPLC: "#06b6d4", ElectroChem: "#ec4899",
};

let _convCounter = Date.now();
function newConvId() { return `conv_${++_convCounter}`; }
function nextTimestamp() { return ++_convCounter; }

export default function AIChatSidebar({
  spectra,
  protectedSpectrumId = null,
  onSpectrumMutated,
}: Props) {
  const { t } = useLang();
  const [open, setOpen] = useState(false);
  const drawerId = useId();
  const toggleRef = useRef<HTMLButtonElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streamedRef = useRef({ reasoning: "", answer: "" });

  const [conversations, setConversations] = useState<Conversation[]>(() => [
    { id: newConvId(), title: t.ai.newConversation, messages: [], createdAt: nextTimestamp() },
  ]);
  const [activeConvId, setActiveConvId] = useState(() => conversations[0].id);

  const activeConv = conversations.find((c) => c.id === activeConvId) || conversations[0];
  const messages = activeConv.messages;

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [suggestions, setSuggestions] = useState<Array<{ title: string; reason: string; action: AIActionRequest; severity: string }>>([]);
  const [preview, setPreview] = useState<{ action: AIActionRequest; data: Record<string, unknown> } | null>(null);
  const [agentOpen, setAgentOpen] = useState(false);
  const [handout, setHandout] = useState<File | null>(null);
  const [agentBusy, setAgentBusy] = useState(false);
  const [agentReport, setAgentReport] = useState<ExperimentAgentReport | null>(null);
  const [actionBusy, setActionBusy] = useState(false);
  const [deleteConvTarget, setDeleteConvTarget] = useState<string | null>(null);
  const [agentForm, setAgentForm] = useState({
    title: "",
    name: "",
    studentId: "",
    college: "",
    major: "",
    teacher: "",
    location: "",
    date: "",
    useLlm: false,
  });

  const selectedSpectra = spectra.filter((s) => selectedIds.has(s.id));
  const selectedCount = selectedSpectra.length;

  useEffect(() => {
    if (open) closeRef.current?.focus();
  }, [open]);

  const closeDrawer = () => {
    setOpen(false);
    toggleRef.current?.focus();
  };

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const newConversation = () => {
    const conv: Conversation = { id: newConvId(), title: t.ai.newConversation, messages: [], createdAt: nextTimestamp() };
    setConversations((prev) => [...prev, conv]);
    setActiveConvId(conv.id);
  };

  const deleteConversation = () => {
    const id = deleteConvTarget;
    if (!id) return;
    if (conversations.length <= 1) return;
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (activeConvId === id && next.length > 0) {
        setActiveConvId(next[next.length - 1].id);
      }
      return next;
    });
    setDeleteConvTarget(null);
  };

  // Build conversation history for API: only user/assistant messages, no reasoning
  const buildHistory = (): Array<{ role: string; content: string }> => {
    const history: Array<{ role: string; content: string }> = [];
    for (const m of messages) {
      if (m.role === "user") {
        history.push({ role: "user", content: m.text });
      } else if (m.role === "ai" && m.text.trim()) {
        history.push({ role: "assistant", content: m.text });
      }
    }
    // Don't include the last assistant message (being streamed)
    if (history.length > 0 && history[history.length - 1].role === "assistant" && loading) {
      history.pop();
    }
    return history;
  };

  const handleAsk = async () => {
    if (!question.trim() || selectedSpectra.length === 0) return;
    const settings = loadAISettings();
    if (!settings.allowDataSharing) {
      setError(t.ai.consentRequired);
      return;
    }
    const q = question.trim();
    setQuestion("");
    setError("");
    setLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;

    // Also set a 10-minute timeout
    const timeoutId = setTimeout(() => controller.abort(), 600000);

    const userMsg: ChatMessage = { role: "user", text: q, reasoning: "", timestamp: nextTimestamp() };
    const aiMsg: ChatMessage = { role: "ai", text: "", reasoning: "", timestamp: nextTimestamp() };

    setConversations((prev) =>
      prev.map((c) => (c.id === activeConvId ? { ...c, messages: [...c.messages, userMsg, aiMsg] } : c))
    );

    // Auto-title from first user message
    if (activeConv.messages.length === 0) {
      const title = q.length > 30 ? q.slice(0, 30) + "..." : q;
      setConversations((prev) =>
        prev.map((c) => (c.id === activeConvId ? { ...c, title } : c))
      );
    }

    const history = buildHistory();

    try {
      const response = await fetch(`${API_BASE}/api/ai/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...chemAppAuthHeaders(false) },
        body: JSON.stringify({
          ids: selectedSpectra.map((s) => s.id),
          question: q,
          api_key: settings.apiKey || undefined,
          base_url: settings.baseUrl || undefined,
          model: settings.model || undefined,
          history,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errText = await response.text();
        setConversations((prev) =>
          prev.map((c) => (c.id === activeConvId
            ? { ...c, messages: c.messages.map((m, j) => (j === c.messages.length - 1 ? { ...m, text: `Error: ${response.status} ${errText}` } : m)) }
            : c))
        );
        setLoading(false);
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) { setLoading(false); return; }

      const decoder = new TextDecoder();
      let buffer = "";
      streamedRef.current = { reasoning: "", answer: "" };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            // SSE payloads from our own backend: type/content are strings.
            const evt = JSON.parse(line.slice(6)) as { type?: string; content?: string };
            const previous = streamedRef.current;
            if (evt.type === "reasoning") {
              streamedRef.current = { ...previous, reasoning: `${previous.reasoning}${String(evt.content || "")}` };
            } else if (evt.type === "answer") {
              streamedRef.current = { ...previous, answer: `${previous.answer}${String(evt.content || "")}` };
            } else if (evt.type === "error") {
              streamedRef.current = { ...previous, answer: `${previous.answer}\n\nError: ${String(evt.content || "")}` };
            }
            const streamed = streamedRef.current;
            setConversations((prev) =>
              prev.map((c) => (c.id === activeConvId
                ? { ...c, messages: c.messages.map((m, j) => (j === c.messages.length - 1 ? { ...m, reasoning: streamed.reasoning, text: streamed.answer } : m)) }
                : c))
            );
          } catch { /* skip */ }
        }
      }
    } catch (e: unknown) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        setError(e instanceof Error ? e.message : t.ai.streamFailed);
      }
    } finally {
      clearTimeout(timeoutId);
      abortRef.current = null;
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void handleAsk(); }
  };

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const handleExecuteAction = async (action: AIActionRequest): Promise<boolean> => {
    if (actionBusy) return false;
    const destructive = isDestructiveAIAction(action);
    const spectrumId = actionSpectrumId(action);
    if (destructive && spectrumId === protectedSpectrumId) {
      setError(t.ai.unsavedActionBlocked);
      return false;
    }
    if (
      destructive
      && (
        typeof action.args.expected_revision !== "number"
        || !action.preview_token
      )
    ) {
      setError(t.ai.revisionRequired);
      return false;
    }
    setError("");
    setActionBusy(true);
    try {
      const data = await executeAIAction(action);
      const text = `${t.ai.actionExecuted} \`${data.action}\`.\n\n\`\`\`json\n${JSON.stringify(data.result, null, 2)}\n\`\`\``;
      const aiMsg: ChatMessage = { role: "ai", text, reasoning: "", timestamp: nextTimestamp() };
      setConversations((prev) =>
        prev.map((c) => (c.id === activeConvId ? { ...c, messages: [...c.messages, aiMsg] } : c))
      );
      if (destructive && spectrumId) {
        try {
          await onSpectrumMutated?.(spectrumId);
        } catch {
          setError(t.ai.refreshFailed);
        }
      }
      return true;
    } catch (e: unknown) {
      setError(undoBlockedMessage(e, t)
        ?? (e instanceof Error ? e.message : t.ai.actionFailed));
      return false;
    } finally {
      setActionBusy(false);
    }
  };

  const handlePreviewAction = async (action: AIActionRequest) => {
    setError("");
    try {
      const destructive = isDestructiveAIAction(action);
      const spectrumId = actionSpectrumId(action);
      if (destructive && spectrumId === protectedSpectrumId) {
        setError(t.ai.unsavedActionBlocked);
        return;
      }
      const revisionBoundAction = await withExpectedRevision(action);
      const data = await previewAIAction(revisionBoundAction);
      if (destructive) {
        if (
          typeof data.expected_revision !== "number"
          || !data.preview_token
          || data.expected_revision !== revisionBoundAction.args.expected_revision
        ) {
          setError(t.ai.revisionRequired);
          return;
        }
        setPreview({
          action: {
            ...revisionBoundAction,
            preview_token: data.preview_token,
            args: {
              ...revisionBoundAction.args,
              expected_revision: data.expected_revision,
            },
          },
          data: data.preview,
        });
        return;
      }
      setPreview({ action: revisionBoundAction, data: data.preview });
    } catch (e: unknown) {
      setError(undoBlockedMessage(e, t)
        ?? (isDestructiveAIAction(action)
          ? t.ai.revisionRequired
          : t.ai.actionPreviewFailed));
    }
  };

  const handleLoadSuggestions = async () => {
    setError("");
    try {
      const data = await suggestAIActions(selectedSpectra.map((s) => s.id));
      setSuggestions(data.suggestions);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.ai.suggestionFailed);
    }
  };

  const handleUndo = async () => {
    const first = selectedSpectra[0];
    if (!first) return;
    await handlePreviewAction({ name: "undo_last_ai_action", args: { spectrum_id: first.id } });
  };

  const buildAgentPayload = () => {
    if (!handout) throw new Error(t.ai.handoutRequired);
    if (selectedSpectra.length === 0) throw new Error(t.ai.spectraRequired);
    const settings = loadAISettings();
    if (agentForm.useLlm && !settings.allowDataSharing) throw new Error(t.ai.consentRequired);
    return {
      handout,
      ids: selectedSpectra.map((s) => s.id),
      ...agentForm,
      apiKey: settings.apiKey || undefined,
      baseUrl: settings.baseUrl || undefined,
      model: settings.model || undefined,
    };
  };

  const handleGenerateExperimentReport = async () => {
    setError("");
    setAgentBusy(true);
    try {
      const report = await generateExperimentAgentReport(buildAgentPayload());
      setAgentReport(report);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.ai.reportAgentFailed);
    } finally {
      setAgentBusy(false);
    }
  };

  const handleDownloadExperimentDocx = async () => {
    setError("");
    setAgentBusy(true);
    try {
      await downloadExperimentAgentDocx(buildAgentPayload());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.ai.docxExportFailed);
    } finally {
      setAgentBusy(false);
    }
  };

  const handleDownloadExperimentMarkdown = async () => {
    setError("");
    setAgentBusy(true);
    try {
      await downloadExperimentAgentMarkdown(buildAgentPayload());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : t.ai.markdownExportFailed);
    } finally {
      setAgentBusy(false);
    }
  };

  return (
    <>
      <button ref={toggleRef} type="button" className="ai-drawer-toggle" onClick={() => open ? closeDrawer() : setOpen(true)} aria-expanded={open} aria-controls={open ? drawerId : undefined} aria-label={open ? t.ai.close : t.ai.open}>
        <span className="ai-drawer-arrow">{open ? "▶" : "◀"}</span>
        <span className="ai-drawer-label">AI</span>
        {selectedCount > 0 && <span className="ai-drawer-badge">{selectedCount}</span>}
      </button>

      {open && (
        <aside id={drawerId} className="ai-drawer" aria-label={t.ai.assistant} onKeyDown={(event) => {
          if (event.key === "Escape" && !event.defaultPrevented && !event.nativeEvent.isComposing) {
            event.preventDefault();
            event.stopPropagation();
            closeDrawer();
          }
        }}>
          <div className="ai-drawer-header">
            <h3>{t.ai.assistant}</h3>
            <button type="button" className="ai-new-chat-btn" onClick={newConversation} title={t.ai.newChat} aria-label={t.ai.newChat}>+</button>
            <span className="ai-drawer-subtitle">
              {selectedCount > 0 ? `${selectedCount}/${spectra.length}` : t.ai.selectSpectra}
            </span>
            <button ref={closeRef} type="button" className="ai-drawer-close" onClick={closeDrawer} aria-label={t.ai.close} title={t.ai.close}>×</button>
          </div>

          {/* Conversation tabs */}
          {conversations.length > 0 && (
            <div className="ai-conv-tabs">
              {conversations.map((conv) => (
                <div key={conv.id} className={`ai-conv-tab ${conv.id === activeConvId ? "active" : ""}`}>
                  <button type="button" onClick={() => setActiveConvId(conv.id)} title={conv.title} aria-pressed={conv.id === activeConvId}>
                    {conv.title}
                  </button>
                  {conversations.length > 1 && (
                    <button type="button" className="ai-conv-del" onClick={() => setDeleteConvTarget(conv.id)} title={t.list.delete} aria-label={t.ai.deleteConversation.replace("{title}", () => conv.title)}>×</button>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Spectrum selection */}
          {spectra.length > 0 ? (
            <div className="ai-select-list">
              {spectra.map((s) => (
                <label key={s.id} className="ai-select-item">
                  <input type="checkbox" checked={selectedIds.has(s.id)} onChange={() => toggleSelect(s.id)} aria-label={t.list.selectAria.replace("{name}", () => s.name || s.id)} />
                  <span className="ai-select-tag" style={{ background: TECH_COLORS[s.technique] || "#6b7280" }}>{s.technique}</span>
                  <span className="ai-select-name">{s.name || s.id}</span>
                </label>
              ))}
            </div>
          ) : (
            <p className="ai-empty" style={{ padding: "12px 16px", textAlign: "left" }}>{t.ai.noSpectra}</p>
          )}

          {selectedCount > 0 && (
            <div className="ai-context-bar">
              {selectedSpectra.map((s) => (
                <span key={s.id} className="ai-context-tag" style={{ background: TECH_COLORS[s.technique] || "#6b7280" }}>{s.technique}</span>
              ))}
              <button type="button" className="ai-mini-btn" onClick={() => void handleLoadSuggestions()} disabled={actionBusy}>{t.ai.suggestions}</button>
              <button type="button" className="ai-mini-btn" onClick={() => void handleUndo()} disabled={actionBusy}>{t.ai.undo}</button>
              <button type="button" className="ai-mini-btn" onClick={() => setAgentOpen((v) => !v)} aria-expanded={agentOpen}>{t.ai.reportAgent}</button>
            </div>
          )}

          {agentOpen && (
            <div className="experiment-agent-panel">
              <div className="manual-subhead">
                <strong>{t.ai.reportTitle}</strong>
                <span>{selectedCount} {t.ai.datasets}</span>
              </div>
              <label className="agent-file">
                <span>{t.ai.handout}</span>
                <input
                  type="file"
                  accept=".docx,.md,.txt"
                  onChange={(e) => setHandout(e.target.files?.[0] || null)}
                />
              </label>
              {handout && <span className="agent-file-name">{handout.name}</span>}
              <div className="agent-grid">
                <input aria-label={t.ai.optionalTitle} placeholder={t.ai.optionalTitle} value={agentForm.title} onChange={(e) => setAgentForm((f) => ({ ...f, title: e.target.value }))} />
                <input aria-label={t.ai.name} placeholder={t.ai.name} value={agentForm.name} onChange={(e) => setAgentForm((f) => ({ ...f, name: e.target.value }))} />
                <input aria-label={t.ai.studentId} placeholder={t.ai.studentId} value={agentForm.studentId} onChange={(e) => setAgentForm((f) => ({ ...f, studentId: e.target.value }))} />
                <input aria-label={t.ai.college} placeholder={t.ai.college} value={agentForm.college} onChange={(e) => setAgentForm((f) => ({ ...f, college: e.target.value }))} />
                <input aria-label={t.ai.major} placeholder={t.ai.major} value={agentForm.major} onChange={(e) => setAgentForm((f) => ({ ...f, major: e.target.value }))} />
                <input aria-label={t.ai.teacher} placeholder={t.ai.teacher} value={agentForm.teacher} onChange={(e) => setAgentForm((f) => ({ ...f, teacher: e.target.value }))} />
                <input aria-label={t.ai.location} placeholder={t.ai.location} value={agentForm.location} onChange={(e) => setAgentForm((f) => ({ ...f, location: e.target.value }))} />
                <input aria-label={t.ai.date} placeholder={t.ai.date} value={agentForm.date} onChange={(e) => setAgentForm((f) => ({ ...f, date: e.target.value }))} />
              </div>
              <label className="agent-toggle">
                <input type="checkbox" checked={agentForm.useLlm} onChange={(e) => setAgentForm((f) => ({ ...f, useLlm: e.target.checked }))} />
                <span>{t.ai.polish}</span>
              </label>
              <div className="agent-actions">
                <button type="button" onClick={() => void handleGenerateExperimentReport()} disabled={!handout || selectedCount === 0 || agentBusy}>
                  {agentBusy ? t.ai.generating : t.ai.generatePreview}
                </button>
                <button type="button" onClick={() => void handleDownloadExperimentDocx()} disabled={!handout || selectedCount === 0 || agentBusy}>{t.ai.exportWord}</button>
                <button type="button" onClick={() => void handleDownloadExperimentMarkdown()} disabled={!handout || selectedCount === 0 || agentBusy}>{t.ai.exportMarkdown}</button>
              </div>
              {agentReport && (
                <details className="agent-preview" open>
                  <summary>{agentReport.title} · {agentReport.used_llm ? t.ai.polished : t.ai.ruleGenerated}</summary>
                  <div className="agent-steps">
                    {agentReport.steps.map((s, i) => (
                      <span key={i}>{s.name}: {s.detail}</span>
                    ))}
                  </div>
                  <SafeMarkdown className="ai-markdown" markdown={agentReport.markdown} />
                </details>
              )}
            </div>
          )}

          {suggestions.length > 0 && (
            <div className="ai-suggestions">
              {suggestions.slice(0, 5).map((item, i) => (
                <div key={i} className="ai-suggestion-item">
                  <strong>{item.title}</strong>
                  <span>{item.reason}</span>
                  <button type="button" onClick={() => void handlePreviewAction(item.action)} disabled={actionBusy}>{t.ai.preview}</button>
                </div>
              ))}
            </div>
          )}

          {/* Messages */}
          <div className="ai-messages" ref={scrollRef}>
            {messages.length === 0 && (
              <p className="ai-empty">{t.ai.empty}</p>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`ai-msg ${m.role}`}>
                <span className="ai-msg-role">{m.role === "user" ? t.ai.you : "AI"}</span>
                {m.role === "ai" && m.reasoning && (
                  <details className="ai-reasoning" open={loading && i === messages.length - 1}>
                    <summary>
                      {t.ai.reasoning}
                      {loading && i === messages.length - 1 && (
                        <span className="ai-reasoning-live"> · {t.ai.inProgress}</span>
                      )}
                    </summary>
                    <pre>{m.reasoning}</pre>
                  </details>
                )}
                {m.role === "ai" ? (
                  <>
                    <SafeMarkdown
                      className="ai-msg-text ai-markdown"
                      markdown={m.text || (loading && i === messages.length - 1 ? `_${t.ai.thinking}_` : "")}
                    />
                    {extractActions(m.text).length > 0 && (
                      <div className="ai-action-list">
                        {extractActions(m.text).map((action, actionIndex) => (
                          <button
                            key={`${action.name}-${actionIndex}`}
                            type="button"
                            className="ai-action-btn"
                            onClick={() => void handlePreviewAction(action)}
                            disabled={loading}
                            title={JSON.stringify(action.args)}
                          >
                            {t.ai.preview} {action.name}
                          </button>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="ai-msg-text">{m.text}</div>
                )}
              </div>
            ))}
          </div>

          {error && <p className="error" role="alert" aria-live="assertive" style={{ margin: "0 12px 8px" }}>{error}</p>}

          {preview && (
            <div className="ai-action-preview">
              <div className="manual-subhead">
                <strong>{t.ai.actionConfirm}</strong>
                <button type="button" onClick={() => setPreview(null)} disabled={actionBusy}>{t.action.cancel}</button>
              </div>
              <pre>{JSON.stringify(preview.data, null, 2)}</pre>
              <button
                type="button"
                className="ai-action-btn"
                onClick={() => {
                  void handleExecuteAction(preview.action).then((executed) => {
                    if (executed) setPreview(null);
                  });
                }}
                disabled={actionBusy}
              >
                {actionBusy ? t.ai.inProgress : `${t.ai.execute} ${preview.action.name}`}
              </button>
            </div>
          )}

          <div className="ai-input-area">
            <input className="ai-input"
              placeholder={selectedCount === 0 ? t.ai.askFirst : t.ai.placeholder}
              aria-label={selectedCount === 0 ? t.ai.askFirst : t.ai.placeholder}
              value={question} onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown} disabled={loading}
            />
            {loading ? (
              <button type="button" className="ai-stop-btn" onClick={handleStop}>■ {t.ai.stop}</button>
            ) : (
              <button type="button" className="ai-send-btn" onClick={() => void handleAsk()}
                disabled={!question.trim() || selectedCount === 0}>
                {t.ai.send}
              </button>
            )}
          </div>
        </aside>
      )}
      <ConfirmDialog
        open={deleteConvTarget !== null}
        title={t.ai.deleteTitle}
        message={t.ai.deleteMessage.replace("{title}", () => conversations.find((conv) => conv.id === deleteConvTarget)?.title || "")}
        confirmLabel={t.ai.confirmDelete}
        cancelLabel={t.action.cancel}
        danger
        onConfirm={deleteConversation}
        onCancel={() => setDeleteConvTarget(null)}
      />
    </>
  );
}
