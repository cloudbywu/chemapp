import { useEffect, useRef, useState } from "react";
import { useLang } from "../i18n/LangContext";
import {
  clearAIKey,
  isAllowedAIEndpoint,
  loadAISettings,
  saveAISettings,
  type AISettings,
} from "../services/aiSettings";
import {
  clearChemAppTokens,
  loadChemAppTokens,
  saveChemAppTokens,
} from "../services/authTokens";

interface Preset {
  label: string;
  baseUrl: string;
  model: string;
}

const PRESETS: Record<string, Preset> = {
  "deepseek-v4-flash": {
    label: "DeepSeek V4 Flash",
    baseUrl: "https://api.deepseek.com",
    model: "deepseek-v4-flash",
  },
  "deepseek-v4-pro": {
    label: "DeepSeek V4 Pro",
    baseUrl: "https://api.deepseek.com",
    model: "deepseek-v4-pro",
  },
  "deepseek-v3": {
    label: "DeepSeek V3 (Chat)",
    baseUrl: "https://api.deepseek.com/v1",
    model: "deepseek-chat",
  },
  custom: {
    // The custom option label is rendered from t.settings.custom on the
    // render path; keep the preset entry free of hardcoded copy.
    label: "",
    baseUrl: "",
    model: "",
  },
};

export default function SettingsPanel() {
  const { t } = useLang();
  const [settings, setSettings] = useState<AISettings>(loadAISettings);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const [tokens, setTokens] = useState(loadChemAppTokens);
  const isCustom = settings.preset === "custom";
  const savedTimerRef = useRef<number | null>(null);

  useEffect(() => () => {
    // Avoid a stale setSaved after the panel unmounts.
    if (savedTimerRef.current != null) {
      window.clearTimeout(savedTimerRef.current);
      savedTimerRef.current = null;
    }
  }, []);

  const handleSave = () => {
    setError("");
    if (!isAllowedAIEndpoint(settings.baseUrl)) {
      setError(t.settings.invalidEndpoint);
      return;
    }
    saveAISettings(settings);
    saveChemAppTokens(tokens);
    setSaved(true);
    if (savedTimerRef.current != null) window.clearTimeout(savedTimerRef.current);
    savedTimerRef.current = window.setTimeout(() => setSaved(false), 2000);
  };

  const handleClearKey = () => {
    clearAIKey();
    setSettings((previous) => ({ ...previous, apiKey: "" }));
    setSaved(false);
  };

  const handleClearChemAppTokens = () => {
    clearChemAppTokens();
    setTokens({ accessToken: "", adminToken: "", reviewerToken: "" });
    setSaved(false);
  };

  const handlePresetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const presetKey = e.target.value;
    const preset = PRESETS[presetKey];
    if (!preset) return;

    if (presetKey === "custom") {
      setSettings((prev) => ({ ...prev, preset: "custom" }));
    } else {
      setSettings((prev) => ({
        ...prev,
        preset: presetKey,
        baseUrl: preset.baseUrl,
        model: preset.model,
      }));
    }
  };

  const update = (field: "apiKey" | "baseUrl" | "model") => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => setSettings((prev) => ({ ...prev, [field]: e.target.value }));

  const presetOptions = Object.entries(PRESETS);

  return (
    <div className="settings-panel">
      <h3>{t.settings.title}</h3>
      <p className="settings-desc">{t.settings.description}</p>

      <div className="settings-form">
        <fieldset className="settings-fieldset">
          <legend>{t.settings.chemAppAccess}</legend>
          <div className="settings-field">
            <label className="settings-label" htmlFor="chemapp-access-token">{t.settings.accessToken}</label>
            <input
              id="chemapp-access-token"
              type="password"
              className="settings-input"
              value={tokens.accessToken}
              onChange={(event) => setTokens((previous) => ({ ...previous, accessToken: event.target.value }))}
              autoComplete="off"
            />
          </div>
          <div className="settings-field">
            <label className="settings-label" htmlFor="chemapp-admin-token">{t.settings.adminToken}</label>
            <input
              id="chemapp-admin-token"
              type="password"
              className="settings-input"
              value={tokens.adminToken}
              onChange={(event) => setTokens((previous) => ({ ...previous, adminToken: event.target.value }))}
              autoComplete="off"
            />
          </div>
          <div className="settings-field">
            <label className="settings-label" htmlFor="chemapp-reviewer-token">{t.settings.reviewerToken}</label>
            <input
              id="chemapp-reviewer-token"
              type="password"
              className="settings-input"
              value={tokens.reviewerToken}
              onChange={(event) => setTokens((previous) => ({ ...previous, reviewerToken: event.target.value }))}
              autoComplete="off"
            />
          </div>
          <p className="settings-hint">{t.settings.tokenHint}</p>
          <p className="settings-hint">{t.settings.tokensSessionOnly}</p>
          <button type="button" className="text-button" onClick={handleClearChemAppTokens} disabled={!tokens.accessToken && !tokens.adminToken && !tokens.reviewerToken}>
            {t.settings.clearTokens}
          </button>
        </fieldset>

        <fieldset className="settings-fieldset">
          <legend>{t.settings.externalAI}</legend>
        <div className="settings-field">
          <label className="settings-label" htmlFor="ai-model-preset">{t.settings.modelChoice}</label>
          <select
            id="ai-model-preset"
            value={settings.preset}
            onChange={handlePresetChange}
            className="settings-select"
          >
            {presetOptions.map(([key, p]) => (
              <option key={key} value={key}>
                {key === "custom" ? `— ${t.settings.custom} —` : p.label}
              </option>
            ))}
          </select>
          <span className="settings-hint">
            {isCustom
              ? t.settings.customHint
              : t.settings.presetHint}
          </span>
        </div>

        <div className="settings-field">
          <label className="settings-label" htmlFor="ai-api-key">API Key</label>
          <input
            id="ai-api-key"
            type="password"
            value={settings.apiKey}
            onChange={update("apiKey")}
            placeholder="sk-..."
            className="settings-input"
            autoComplete="off"
          />
          <span className="settings-hint" id="ai-api-key-hint">{t.settings.apiKeyHint}</span>
          <button type="button" className="text-button" onClick={handleClearKey} disabled={!settings.apiKey}>
            {t.settings.clearKey}
          </button>
        </div>

        <div className="settings-field">
          <label className="settings-label" htmlFor="ai-base-url">Base URL</label>
          <input
            id="ai-base-url"
            type="text"
            value={settings.baseUrl}
            onChange={update("baseUrl")}
            placeholder="https://api.deepseek.com"
            className="settings-input"
            disabled={!isCustom}
          />
          <span className="settings-hint">
            {isCustom ? t.settings.endpointHint : t.settings.presetHint}
          </span>
        </div>

        <div className="settings-field">
          <label className="settings-label" htmlFor="ai-model-name">Model</label>
          <input
            id="ai-model-name"
            type="text"
            value={settings.model}
            onChange={update("model")}
            placeholder="deepseek-v4-pro"
            className="settings-input"
            disabled={!isCustom}
          />
          <span className="settings-hint">
            {isCustom ? t.settings.modelHint : t.settings.presetHint}
          </span>
        </div>

        <label className="settings-consent">
          <input
            type="checkbox"
            checked={settings.allowDataSharing}
            onChange={(event) => setSettings((previous) => ({
              ...previous,
              allowDataSharing: event.target.checked,
            }))}
          />
          <span>{t.settings.dataConsent}</span>
        </label>
        <p className="settings-hint">{t.settings.dataConsentHint}</p>
        </fieldset>

        <div className="action-bar">
          <button type="button" onClick={handleSave}>
            {saved ? t.settings.saved : t.settings.save}
          </button>
        </div>
        <p className="settings-hint" role="status" aria-live="polite">
          {saved ? t.settings.keyMemoryOnly : ""}
        </p>
        {error && <p className="error" role="alert">{error}</p>}
      </div>
    </div>
  );
}
