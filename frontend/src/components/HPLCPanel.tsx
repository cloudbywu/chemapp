import { useEffect, useState } from "react";
import { analyzeSpectrum, ApiError } from "../services/api";
import type { AnalysisResult, HplcIntegrationEvent } from "../types/spectrum";
import { useLang } from "../i18n/LangContext";
import {
  MISSING_VALUE_PLACEHOLDER,
  finiteNumber,
  numberArray,
  text,
} from "../utils/number";
import ConfirmDialog from "./ConfirmDialog";

interface ChannelPeaks {
  wavelength_nm: number | null;
  color: string;
  peaks: Array<{
    position: number;
    intensity: number;
    width: number | null;
    area: number;
    area_percent?: number;
    type?: string;
    name?: string;
  }>;
  total_area?: number;
  source?: string;
}

interface Props {
  result: AnalysisResult;
  spectrumId?: string | null;
  onResultChanged?: (result: AnalysisResult) => void;
  onDirtyChange?: (dirty: boolean) => void;
}

export default function HPLCPanel({
  result,
  spectrumId,
  onResultChanged,
  onDirtyChange,
}: Props) {
  const { t } = useLang();
  const metrics = result.metrics;
  const totalPeaks = finiteNumber(metrics.n_peaks);
  const channelCount = finiteNumber(metrics.n_channels);
  const timeRange = numberArray(metrics.time_range_min);
  const integrationSource = text(metrics.integration_source);
  const channelPeaks = (metrics.channel_peaks as Record<string, ChannelPeaks>) || {};
  const manualConfirmed = Boolean(metrics.manual_confirmed);
  const [events, setEvents] = useState<HplcIntegrationEvent[]>(() => (metrics.integration_events as HplcIntegrationEvent[]) || []);
  const [eventsBaseline, setEventsBaseline] = useState(() => JSON.stringify(events));
  const [eventsRevision, setEventsRevision] = useState(
    () => result.result_revision ?? 0,
  );
  const [eventError, setEventError] = useState("");
  const [confirmEvents, setConfirmEvents] = useState(false);
  const [applyingEvents, setApplyingEvents] = useState(false);
  const eventsDirty = JSON.stringify(events) !== eventsBaseline;

  useEffect(() => {
    onDirtyChange?.(eventsDirty);
  }, [eventsDirty, onDirtyChange]);

  useEffect(() => () => onDirtyChange?.(false), [onDirtyChange]);

  useEffect(() => {
    const incomingRevision = result.result_revision ?? 0;
    if (eventsDirty || incomingRevision === eventsRevision) return;
    const incomingEvents = (metrics.integration_events as HplcIntegrationEvent[]) || [];
    // Synchronize an untouched editor to the newly committed server snapshot.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setEvents(incomingEvents);
    setEventsBaseline(JSON.stringify(incomingEvents));
    setEventsRevision(incomingRevision);
    setEventError("");
    setConfirmEvents(false);
  }, [eventsDirty, eventsRevision, metrics.integration_events, result.result_revision]);

  const channels = Object.keys(channelPeaks);
  const updateEvent = (index: number, patch: Partial<HplcIntegrationEvent>) => {
    setEvents((prev) => prev.map((event, i) => (i === index ? { ...event, ...patch } : event)));
  };

  const applyEvents = async (forceOverwrite = false) => {
    if (!spectrumId || !onResultChanged || applyingEvents) return;
    setConfirmEvents(false);
    setEventError("");
    setApplyingEvents(true);
    try {
      const next = await analyzeSpectrum(spectrumId, {
        ...(metrics.analysis_options as Record<string, unknown>),
        integration_events: events,
      }, undefined, eventsRevision, forceOverwrite);
      onResultChanged(next);
      const nextEvents = (next.metrics as Record<string, unknown>).integration_events as HplcIntegrationEvent[] | undefined;
      const committedEvents = nextEvents || events;
      setEvents(committedEvents);
      setEventsBaseline(JSON.stringify(committedEvents));
      setEventsRevision(next.result_revision ?? eventsRevision);
    } catch (cause: unknown) {
      setEventError(cause instanceof ApiError && cause.status === 409
        ? t.manual.revisionConflict
        : cause instanceof Error ? cause.message : t.hplc.eventFailed);
    } finally {
      setApplyingEvents(false);
    }
  };

  const requestApplyEvents = () => {
    if (manualConfirmed) setConfirmEvents(true);
    else void applyEvents(false);
  };

  return (
    <div className="analysis-panel">
      <h3>{t.hplc.title}</h3>
      <p className="summary">{result.summary}</p>
      <p className="summary">
        {t.hplc.integrationSource}: {integrationSource === "instrument_record" ? t.hplc.instrumentRecord : t.hplc.computed}
      </p>

      <div className="metrics-grid">
        <div className="metric">
          <span className="metric-label">{t.hplc.totalPeaks}</span>
          <span className="metric-value">{totalPeaks ?? MISSING_VALUE_PLACEHOLDER}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.hplc.channels}</span>
          <span className="metric-value">{channelCount ?? MISSING_VALUE_PLACEHOLDER}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.hplc.timeRange}</span>
          <span className="metric-value">
            {timeRange.length >= 2
              ? `${timeRange[0]} – ${timeRange[1]} min`
              : MISSING_VALUE_PLACEHOLDER}
          </span>
        </div>
      </div>

      {spectrumId && onResultChanged && (
        <div className="table-section hplc-event-editor">
          <div className="manual-subhead">
            <h4>{t.hplc.eventEditor}</h4>
            <button type="button" onClick={() => setEvents((prev) => [...prev, { channel: channels[0], start: 0, end: 0, mode: "force_bb" }])} disabled={applyingEvents}>{t.hplc.addEvent}</button>
          </div>
          {events.length > 0 && (
            <table>
              <thead>
                <tr>
                  <th>{t.hplc.channel}</th>
                  <th>{t.hplc.start}</th>
                  <th>{t.hplc.end}</th>
                  <th>{t.hplc.mode}</th>
                  <th>{t.hplc.nameFactor}</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {events.map((event, i) => (
                  <tr key={i}>
                    <td>
                      <select disabled={applyingEvents} value={event.channel || ""} onChange={(e) => updateEvent(i, { channel: e.target.value || undefined })}>
                        <option value="">{t.hplc.all}</option>
                        {channels.map((ch) => <option key={ch} value={ch}>{ch}</option>)}
                      </select>
                    </td>
                    <td><input disabled={applyingEvents} type="number" step={0.001} value={event.start} onChange={(e) => {
                      const raw = e.target.value;
                      // Clearing the field must not snap back to 0 (Number("") === 0);
                      // keep the last valid value until a finite number is typed.
                      if (raw.trim() === "") return;
                      const parsed = Number(raw);
                      if (Number.isFinite(parsed)) updateEvent(i, { start: parsed });
                    }} /></td>
                    <td><input disabled={applyingEvents} type="number" step={0.001} value={event.end} onChange={(e) => {
                      const raw = e.target.value;
                      if (raw.trim() === "") return;
                      const parsed = Number(raw);
                      if (Number.isFinite(parsed)) updateEvent(i, { end: parsed });
                    }} /></td>
                    <td>
                      <select disabled={applyingEvents} value={event.mode} onChange={(e) => updateEvent(i, { mode: e.target.value as HplcIntegrationEvent["mode"] })}>
                        <option value="force_bb">{t.hplc.forceBb}</option>
                        <option value="force_vv">{t.hplc.forceVv}</option>
                        <option value="delete">{t.hplc.deletePeak}</option>
                        <option value="name">{t.hplc.namePeak}</option>
                      </select>
                    </td>
                    <td>
                      <input disabled={applyingEvents} value={event.name ?? event.area_factor ?? ""} onChange={(e) => {
                        const raw = e.target.value;
                        const numeric = Number(raw);
                        updateEvent(i, Number.isFinite(numeric) && raw.trim() !== "" ? { area_factor: numeric, name: undefined } : { name: raw, area_factor: undefined });
                      }} />
                    </td>
                    <td><button type="button" className="mini-danger" disabled={applyingEvents} onClick={() => setEvents((prev) => prev.filter((_, j) => j !== i))} aria-label={t.hplc.deleteEvent.replace("{index}", String(i + 1))}>{t.hplc.delete}</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <button type="button" className="primary-inline" onClick={requestApplyEvents} disabled={!events.length || !eventsDirty || applyingEvents}>{t.hplc.applyEvents}</button>
          {eventError && <p className="error" role="alert">{eventError}</p>}
        </div>
      )}

      {manualConfirmed && result.peaks.length > 0 && (
        <div className="table-section">
          <h4>{t.hplc.manualPeaks}</h4>
          <table>
            <thead>
              <tr>
                <th>tR (min)</th>
                <th>{t.hplc.type}</th>
                <th>{t.hplc.area}</th>
                <th>{t.hplc.height}</th>
                <th>{t.hplc.width}</th>
                <th>{t.hplc.areaPercent}</th>
                <th>{t.hplc.name}</th>
              </tr>
            </thead>
            <tbody>
              {result.peaks.slice(0, 30).map((p, i) => (
                <tr key={i}>
                  <td>{p.position.toFixed(3)}</td>
                  <td>—</td>
                  <td>{p.area != null ? p.area.toFixed(1) : "—"}</td>
                  <td>{p.intensity.toFixed(2)}</td>
                  <td>{p.width != null ? p.width.toFixed(4) : "—"}</td>
                  <td>—</td>
                  <td>{p.assignment || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!manualConfirmed && Object.keys(channelPeaks).length > 0 && (
        <div className="table-section">
          {Object.entries(channelPeaks).map(([chName, chData]) => (
            <div key={chName} style={{ marginBottom: 16 }}>
              <h4 style={{ color: chData.color }}>
                {chName} ({chData.wavelength_nm}nm)
                {chData.total_area != null ? ` · total ${chData.total_area.toFixed(2)}` : ""}
              </h4>
              {chData.peaks.length > 0 ? (
                <table>
                  <thead>
                    <tr>
                      <th>tR (min)</th>
                      <th>{t.hplc.type}</th>
                      <th>{t.hplc.width}</th>
                      <th>{t.hplc.area}</th>
                      <th>{t.hplc.height}</th>
                      <th>{t.hplc.areaPercent}</th>
                      <th>{t.hplc.name}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {chData.peaks.slice(0, 30).map((p, i) => (
                      <tr key={i}>
                        <td>{p.position.toFixed(3)}</td>
                        <td>{p.type || "—"}</td>
                        <td>{p.width != null ? p.width.toFixed(2) : "—"}</td>
                        <td>{p.area.toFixed(1)}</td>
                        <td>{p.intensity.toFixed(2)}</td>
                        <td>{p.area_percent != null ? p.area_percent.toFixed(2) : "—"}</td>
                        <td>{p.name || "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p style={{ fontSize: 12, color: "#64748b" }}>{t.hplc.noPeaks}</p>
              )}
            </div>
          ))}
        </div>
      )}

      {!manualConfirmed && Object.keys(channelPeaks).length === 0 && result.peaks.length > 0 && (
        <div className="table-section">
          <h4>{t.hplc.peaks}</h4>
          <table>
            <thead>
              <tr>
                <th>tR (min)</th>
                <th>{t.hplc.type}</th>
                <th>{t.hplc.area}</th>
                <th>{t.hplc.height}</th>
                <th>{t.hplc.width}</th>
                <th>{t.hplc.areaPercent}</th>
              </tr>
            </thead>
            <tbody>
              {result.peaks.slice(0, 15).map((p, i) => (
                <tr key={i}>
                  <td>{p.position.toFixed(3)}</td>
                  <td>—</td>
                  <td>{p.area != null ? p.area.toFixed(1) : "—"}</td>
                  <td>{p.intensity.toFixed(2)}</td>
                  <td>{p.width != null ? p.width.toFixed(4) : "—"}</td>
                  <td>—</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <ConfirmDialog
        open={confirmEvents}
        title={t.workbench.confirmReanalysisTitle}
        message={t.workbench.confirmReanalysisMessage}
        confirmLabel={t.workbench.overwriteAndAnalyze}
        cancelLabel={t.action.cancel}
        busy={applyingEvents}
        busyLabel={t.workbench.busy}
        danger
        onConfirm={() => void applyEvents(true)}
        onCancel={() => setConfirmEvents(false)}
      />
    </div>
  );
}
