import { useCallback, useEffect, useState } from "react";
import { useLang } from "../i18n/LangContext";
import { getMLStatus } from "../services/api";
import { record, text } from "../utils/number";

export default function MLTrainingPanel() {
  const { t } = useLang();
  const [status, setStatus] = useState<Record<string, unknown>>({});
  const [error, setError] = useState("");

  const refreshStatus = useCallback(async (signal?: AbortSignal) => {
    try {
      setStatus(await getMLStatus(signal));
    } catch (cause: unknown) {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) {
        setError(cause instanceof Error ? cause.message : t.error.network);
      }
    }
  }, [t.error.network]);

  // Poll every 5s while the tab is visible; pause polling in background
  // tabs and refresh immediately once the user returns to the foreground.
  useEffect(() => {
    const controller = new AbortController();
    let intervalId: number | null = null;

    const stopPolling = () => {
      if (intervalId != null) {
        window.clearInterval(intervalId);
        intervalId = null;
      }
    };
    const startPolling = () => {
      if (intervalId == null) {
        intervalId = window.setInterval(() => void refreshStatus(controller.signal), 5000);
      }
    };
    const handleVisibilityChange = () => {
      if (document.hidden) {
        stopPolling();
        return;
      }
      void refreshStatus(controller.signal);
      startPolling();
    };

    // Initial fetch via a promise chain (setState stays out of the effect body).
    getMLStatus(controller.signal)
      .then(setStatus)
      .catch((cause: unknown) => {
        if (!(cause instanceof DOMException && cause.name === "AbortError")) {
          setError(cause instanceof Error ? cause.message : t.error.network);
        }
      });
    startPolling();
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      controller.abort();
      stopPolling();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [refreshStatus, t.error.network]);

  const training = record(status.training) ?? {};
  const trainStatus = text(training.status);
  const modelLoaded = status.model_loaded === true;

  return (
    <div className="settings-panel">
      <h3>{t.training.title}</h3>
      <p className="settings-desc">{t.training.description}</p>
      <p className="settings-desc" style={{ color: "#f59e0b" }}>
        {t.training.legacyNotice}
      </p>

      <div className="ml-status-grid">
        <div className="metric">
          <span className="metric-label">{t.training.modelStatus}</span>
          <span className="metric-value" style={{ color: modelLoaded ? "#22d3a0" : "#f59e0b" }}>
            {modelLoaded ? t.training.trained : t.training.untrained}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.training.classes}</span>
          <span className="metric-value">{String(status.n_classes || "—")}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.training.device}</span>
          <span className="metric-value">{String(status.device || "Unknown")}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.training.trainingStatus}</span>
          <span className="metric-value" style={{ fontSize: 12 }}>{trainStatus || "idle"}</span>
        </div>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}
