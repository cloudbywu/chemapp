export interface AISettings {
  preset: string;
  apiKey: string;
  baseUrl: string;
  model: string;
  allowDataSharing: boolean;
}

const STORAGE_KEY = "chemapp-ai-settings";

const DEFAULTS: AISettings = {
  preset: "deepseek-v4-pro",
  apiKey: "",
  baseUrl: "https://api.deepseek.com",
  model: "deepseek-v4-pro",
  allowDataSharing: false,
};

// Deliberately memory-only. API keys must not survive a reload and are never
// written to localStorage/sessionStorage where an unrelated script can read them.
let inMemoryApiKey = "";

function readPersistedSettings(): Partial<AISettings> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Partial<AISettings> & { apiKey?: unknown };
    // Migrate away from the former persistent-key format.
    if ("apiKey" in parsed) {
      delete parsed.apiKey;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(parsed));
    }
    return parsed;
  } catch {
    return {};
  }
}

export function loadAISettings(): AISettings {
  return {
    ...DEFAULTS,
    ...readPersistedSettings(),
    apiKey: inMemoryApiKey,
  };
}

export function saveAISettings(settings: AISettings): void {
  inMemoryApiKey = settings.apiKey.trim();
  const { apiKey: _discarded, ...safeSettings } = settings;
  void _discarded;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(safeSettings));
}

export function clearAIKey(): void {
  inMemoryApiKey = "";
}

export function isAllowedAIEndpoint(rawUrl: string): boolean {
  try {
    const url = new URL(rawUrl);
    if (url.protocol === "https:") return true;
    return url.protocol === "http:" && (url.hostname === "localhost" || url.hostname === "127.0.0.1");
  } catch {
    return false;
  }
}
