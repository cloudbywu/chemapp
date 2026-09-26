import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, integrateRanges, listResultVersions, restoreResultVersion, saveManualResult } from "../services/api";
import type { AnalysisResult, IntegralsItem, Peak, SpectrumData } from "../types/spectrum";
import { useLang } from "../i18n/LangContext";
import ConfirmDialog from "./ConfirmDialog";

interface Props {
  spectrum: SpectrumData;
  result: AnalysisResult;
  onSaved: (result: AnalysisResult) => void;
  activePlotPick?: PlotPickTarget;
  pickedPlotRange?: PlotPickedRange;
  onRequestPlotPick?: (target: PlotPickTarget) => void;
  onDirtyChange?: (dirty: boolean) => void;
}

type PeakWindow = Record<number, { start: number; end: number; channel?: string }>;
type PlotPickTarget = { mode: "nmr" | "hplc"; index: number; label: string } | null;
type PlotPickedRange = { mode: "nmr" | "hplc"; index: number; start: number; end: number; nonce: number } | null;
type HplcPeak = {
  position: number;
  intensity: number;
  area: number;
  width: number | null;
  type?: string;
  name?: string;
  begin_time?: number;
  end_time?: number;
  area_percent?: number;
  channel?: string;
};
type HplcChannelPeaks = Record<string, { peaks: HplcPeak[]; wavelength_nm?: number | null; color?: string; total_area?: number; source?: string }>;

function clonePeaks(peaks: Peak[]): Peak[] {
  return peaks.map((p) => ({ ...p }));
}

function cloneIntegrals(integrals?: IntegralsItem[]): IntegralsItem[] {
  return (integrals || []).map((item) => ({ ...item }));
}

function toNumber(raw: string, fallback = 0): number {
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function serializeReview(
  peaks: Peak[],
  integrals: IntegralsItem[],
  channels: HplcChannelPeaks,
  windows: PeakWindow,
): string {
  return JSON.stringify({ peaks, integrals, channels, windows });
}

function withRecalculatedHplcPeak(
  previous: HplcChannelPeaks,
  channel: string,
  position: number,
  recalculated: { area: number; height?: number; width?: number; start: number; end: number; apex_time?: number; baseline_start?: number; baseline_end?: number },
): HplcChannelPeaks {
  const channelData = previous[channel];
  if (!channelData) return previous;
  const peaksForChannel = channelData.peaks.map((peak) => (
    Math.abs(peak.position - position) < 0.02
      ? {
          ...peak,
          area: recalculated.area,
          intensity: recalculated.height ?? peak.intensity,
          width: recalculated.width ?? peak.width,
          begin_time: recalculated.start,
          end_time: recalculated.end,
          position: recalculated.apex_time ?? peak.position,
          baseline_start: recalculated.baseline_start,
          baseline_end: recalculated.baseline_end,
        }
      : peak
  ));
  const total = peaksForChannel.reduce((sum, peak) => sum + Number(peak.area || 0), 0);
  return {
    ...previous,
    [channel]: {
      ...channelData,
      peaks: peaksForChannel.map((peak) => ({
        ...peak,
        area_percent: total > 0 ? Number((peak.area / total * 100).toFixed(2)) : 0,
      })),
      total_area: Number(total.toFixed(2)),
      source: "manual_confirmed",
    },
  };
}

export default function ManualReviewPanel({
  spectrum,
  result,
  onSaved,
  activePlotPick,
  pickedPlotRange,
  onRequestPlotPick,
  onDirtyChange,
}: Props) {
  const { t } = useLang();
  const [peaks, setPeaks] = useState<Peak[]>(() => clonePeaks(result.peaks));
  const [integrals, setIntegrals] = useState<IntegralsItem[]>(() => cloneIntegrals(result.integrals));
  const [hplcChannels, setHplcChannels] = useState<HplcChannelPeaks>(() => (
    ((result.metrics.channel_peaks as HplcChannelPeaks | undefined) || {})
  ));
  const [newPeak, setNewPeak] = useState({ position: "", intensity: "" });
  const [windows, setWindows] = useState<PeakWindow>({});
  const [baseline, setBaseline] = useState(() => serializeReview(
    clonePeaks(result.peaks),
    cloneIntegrals(result.integrals),
    ((result.metrics.channel_peaks as HplcChannelPeaks | undefined) || {}),
    {},
  ));
  const [baselineRevision, setBaselineRevision] = useState(
    () => result.result_revision ?? 0,
  );
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [versions, setVersions] = useState<Array<{ version: number; note: string; created_at: string; n_peaks: number; summary: string }>>([]);
  const [restoreTarget, setRestoreTarget] = useState<number | null>(null);
  const consumedPickNonce = useRef<number | null>(null);

  useEffect(() => {
    listResultVersions(spectrum.id).then((data) => setVersions(data.versions)).catch(() => setVersions([]));
  }, [spectrum.id]);

  const dirty = useMemo(
    () => serializeReview(peaks, integrals, hplcChannels, windows) !== baseline,
    [baseline, hplcChannels, integrals, peaks, windows],
  );

  useEffect(() => {
    onDirtyChange?.(dirty);
  }, [dirty, onDirtyChange]);

  useEffect(() => () => onDirtyChange?.(false), [onDirtyChange]);

  const channels = useMemo(() => {
    const raw = spectrum.parameters?.channels as Array<{ name: string }> | undefined;
    return raw?.map((ch) => ch.name) || [];
  }, [spectrum.parameters]);

  useEffect(() => {
    if (!pickedPlotRange || consumedPickNonce.current === pickedPlotRange.nonce) return;
    consumedPickNonce.current = pickedPlotRange.nonce;

    const start = Number(pickedPlotRange.start.toFixed(4));
    const end = Number(pickedPlotRange.end.toFixed(4));
    const center = Number(((start + end) / 2).toFixed(4));

    if (pickedPlotRange.mode === "nmr" && spectrum.technique === "NMR") {
      // This state change is the intended response to an external Plotly selection event.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setIntegrals((prev) => prev.map((item, i) => (
        i === pickedPlotRange.index
          ? { ...item, start_ppm: start, end_ppm: end, center_ppm: center }
          : item
      )));
      integrateRanges(spectrum.id, [{ start, end, center }]).then((data) => {
        const area = data.integrals[0]?.area;
        if (area == null) return;
        setIntegrals((prev) => prev.map((item, i) => (
          i === pickedPlotRange.index ? { ...item, raw_area: area } : item
        )));
      }).catch((e: unknown) => {
        setMessage(e instanceof Error ? e.message : t.manual.recalcFailed);
      });
      setMessage(t.manual.updatedNmrRange);
    }

    if (pickedPlotRange.mode === "hplc" && spectrum.technique === "HPLC") {
      const peak = peaks[pickedPlotRange.index];
      if (!peak) return;
      const current = windows[pickedPlotRange.index] || { start, end, channel: channels[0] };
      const next = { ...current, start, end };
      setWindows((prev) => ({ ...prev, [pickedPlotRange.index]: next }));
      integrateRanges(spectrum.id, [{ ...next, center: peak.position }]).then((data) => {
        const recalculated = data.integrals[0];
        const area = recalculated?.area;
        if (area == null) return;
        setPeaks((prev) => prev.map((p, i) => i === pickedPlotRange.index ? { ...p, area } : p));
        if (next.channel) {
          setHplcChannels((previous) => withRecalculatedHplcPeak(
            previous,
            next.channel as string,
            peak.position,
            recalculated,
          ));
        }
      }).catch((e: unknown) => {
        setMessage(e instanceof Error ? e.message : t.manual.recalcFailed);
      });
      setMessage(t.manual.updatedHplcWindow);
    }
  }, [pickedPlotRange, spectrum.id, spectrum.technique, peaks, windows, channels, t.manual]);

  const addPeak = () => {
    const position = Number(newPeak.position);
    const intensity = Number(newPeak.intensity);
    if (!Number.isFinite(position) || !Number.isFinite(intensity)) {
      setMessage(t.manual.invalidPeak);
      return;
    }
    setPeaks((prev) => [...prev, {
      position,
      intensity,
      area: null,
      width: null,
      assignment: "manual",
      multiplicity: "",
      coupling_constant: null,
    }].sort((a, b) => a.position - b.position));
    setNewPeak({ position: "", intensity: "" });
    setMessage("");
  };

  const deletePeak = (index: number) => {
    setPeaks((prev) => prev.filter((_, i) => i !== index));
  };

  const updateIntegral = (index: number, key: keyof IntegralsItem, raw: string) => {
    setIntegrals((prev) => prev.map((item, i) => (
      i === index ? { ...item, [key]: toNumber(raw, Number(item[key]) || 0) } : item
    )));
  };

  const recalcNmrIntegrals = async () => {
    const ranges = integrals.map((item) => ({
      start: item.start_ppm,
      end: item.end_ppm,
      center: item.center_ppm,
    }));
    const data = await integrateRanges(spectrum.id, ranges);
    const recalculated = integrals.map((item, i) => ({
      ...item,
      raw_area: data.integrals[i]?.area ?? item.raw_area,
    }));
    setIntegrals(recalculated);
  };

  const recalcHplcPeak = async (index: number) => {
    const peak = peaks[index];
    const current = windows[index] || {
      start: peak.position - Math.max(peak.width || 0.05, 0.05),
      end: peak.position + Math.max(peak.width || 0.05, 0.05),
      channel: channels[0],
    };
    const data = await integrateRanges(spectrum.id, [{ ...current, center: peak.position, baseline: "linear" }]);
    const recalculated = data.integrals[0];
    const area = recalculated?.area;
    if (area == null) return;
    setPeaks((prev) => prev.map((p, i) => i === index ? { ...p, area } : p));
    if (current.channel) {
      setHplcChannels((previous) => withRecalculatedHplcPeak(
        previous,
        current.channel as string,
        peak.position,
        recalculated,
      ));
    }
  };

  const requestNmrPick = (index: number) => {
    onRequestPlotPick?.({ mode: "nmr", index, label: t.manual.nmrPickLabel.replace("{index}", String(index + 1)) });
    setMessage(t.manual.pickNmr);
  };

  const requestHplcPick = (index: number) => {
    onRequestPlotPick?.({ mode: "hplc", index, label: t.manual.hplcPickLabel.replace("{index}", String(index + 1)) });
    setMessage(t.manual.pickHplc);
  };

  const save = async () => {
    setSaving(true);
    setMessage("");
    try {
      const saved = await saveManualResult(spectrum.id, {
        peaks,
        integrals: spectrum.technique === "NMR" ? integrals : undefined,
        metrics: {
          ...result.metrics,
          n_peaks: peaks.length,
          channel_peaks: spectrum.technique === "HPLC" ? hplcChannels : result.metrics.channel_peaks,
        },
        channel_peaks: spectrum.technique === "HPLC" ? hplcChannels : undefined,
        summary: result.summary,
        expected_revision: baselineRevision,
      });
      setBaseline(serializeReview(
        clonePeaks(saved.peaks),
        cloneIntegrals(saved.integrals),
        ((saved.metrics.channel_peaks as HplcChannelPeaks | undefined) || {}),
        windows,
      ));
      setBaselineRevision(saved.result_revision ?? baselineRevision);
      onSaved(saved);
      listResultVersions(spectrum.id).then((data) => setVersions(data.versions)).catch(() => undefined);
      setMessage(t.manual.savedMessage);
    } catch (e: unknown) {
      setMessage(e instanceof ApiError && e.status === 409
        ? t.manual.revisionConflict
        : e instanceof Error ? e.message : t.manual.saveFailed);
    } finally {
      setSaving(false);
    }
  };

  const restoreVersion = async () => {
    if (restoreTarget === null || saving) return;
    const version = restoreTarget;
    setSaving(true);
    setMessage("");
    try {
      const restored = await restoreResultVersion(spectrum.id, version, baselineRevision);
      const restoredPeaks = clonePeaks(restored.peaks);
      const restoredIntegrals = cloneIntegrals(restored.integrals);
      const restoredChannels = ((restored.metrics.channel_peaks as HplcChannelPeaks | undefined) || {});
      setPeaks(restoredPeaks);
      setIntegrals(restoredIntegrals);
      setHplcChannels(restoredChannels);
      setWindows({});
      setBaseline(serializeReview(restoredPeaks, restoredIntegrals, restoredChannels, {}));
      setBaselineRevision(restored.result_revision ?? baselineRevision);
      onSaved(restored);
      setRestoreTarget(null);
      setMessage(t.manual.restoredMessage.replace("{version}", String(version)));
    } catch (e: unknown) {
      setMessage(e instanceof ApiError && e.status === 409
        ? t.manual.revisionConflict
        : e instanceof Error ? e.message : t.manual.restoreFailed);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="manual-panel">
      <div className="manual-head">
        <h3>{t.manual.title}</h3>
        {Boolean(result.metrics.manual_confirmed) && <span>{t.manual.confirmed} v{String(result.metrics.manual_version || 1)}</span>}
      </div>

      <div className="manual-add-row">
        <label><span className="visually-hidden">{t.manual.position}</span><input aria-label={t.manual.position} placeholder={spectrum.technique === "HPLC" ? "tR" : t.manual.position} value={newPeak.position}
          onChange={(e) => setNewPeak((prev) => ({ ...prev, position: e.target.value }))} /></label>
        <label><span className="visually-hidden">{t.manual.intensity}</span><input aria-label={t.manual.intensity} placeholder={t.manual.intensity} value={newPeak.intensity}
          onChange={(e) => setNewPeak((prev) => ({ ...prev, intensity: e.target.value }))} /></label>
        <button type="button" onClick={addPeak}>{t.manual.addPeak}</button>
      </div>

      <div className="table-section compact-table">
        <h4>{t.manual.peaks}</h4>
        <table>
          <thead>
            <tr>
              <th>{t.manual.position}</th>
              <th>{t.manual.intensity}</th>
              <th>{t.manual.area}</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {peaks.slice(0, 50).map((peak, index) => (
              <tr key={`${peak.position}-${index}`}>
                <td>{peak.position.toFixed(spectrum.technique === "HPLC" ? 3 : 4)}</td>
                <td>{peak.intensity.toExponential(2)}</td>
                <td>{peak.area != null ? peak.area.toFixed(2) : "-"}</td>
                <td><button className="mini-danger" type="button" onClick={() => deletePeak(index)} aria-label={t.manual.deletePeakAria.replace("{position}", () => String(peak.position))}>{t.manual.deletePeak}</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {spectrum.technique === "NMR" && integrals.length > 0 && (
        <div className="table-section compact-table">
          <div className="manual-subhead">
            <h4>{t.manual.nmrRanges}</h4>
            <button type="button" onClick={recalcNmrIntegrals} disabled={saving}>{t.manual.recalculateIntegrals}</button>
          </div>
          <table>
            <thead>
              <tr>
                <th>{t.manual.center}</th>
                <th>{t.manual.start}</th>
                <th>{t.manual.end}</th>
                <th>{t.manual.area}</th>
                <th>{t.manual.relative}</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {integrals.slice(0, 30).map((item, index) => (
                <tr key={`${item.center_ppm}-${index}`}>
                  <td><input value={item.center_ppm} onChange={(e) => updateIntegral(index, "center_ppm", e.target.value)} /></td>
                  <td><input value={item.start_ppm} onChange={(e) => updateIntegral(index, "start_ppm", e.target.value)} /></td>
                  <td><input value={item.end_ppm} onChange={(e) => updateIntegral(index, "end_ppm", e.target.value)} /></td>
                  <td>{item.raw_area.toFixed(2)}</td>
                  <td>{item.relative_area}</td>
                  <td>
                    <button
                      type="button"
                      className={activePlotPick?.mode === "nmr" && activePlotPick.index === index ? "pick-active" : ""}
                      onClick={() => requestNmrPick(index)}
                    >
                      {t.manual.selectOnPlot}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {spectrum.technique === "HPLC" && (peaks.length > 0 || Object.keys(hplcChannels).length > 0) && (
        <div className="table-section compact-table">
          <h4>{t.manual.hplcReintegration}</h4>
          <table>
            <thead>
              <tr>
                <th>tR</th>
                <th>{t.manual.start}</th>
                <th>{t.manual.end}</th>
                {channels.length > 0 && <th>{t.manual.channel}</th>}
                <th></th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {peaks.slice(0, 20).map((peak, index) => {
                const current = windows[index] || {
                  start: Number((peak.position - Math.max(peak.width || 0.05, 0.05)).toFixed(4)),
                  end: Number((peak.position + Math.max(peak.width || 0.05, 0.05)).toFixed(4)),
                  channel: peak.assignment || channels[0],
                };
                return (
                  <tr key={`hplc-${peak.position}-${index}`}>
                    <td>{peak.position.toFixed(3)}</td>
                    <td><input value={current.start} onChange={(e) => setWindows((prev) => ({ ...prev, [index]: { ...current, start: toNumber(e.target.value, current.start) } }))} /></td>
                    <td><input value={current.end} onChange={(e) => setWindows((prev) => ({ ...prev, [index]: { ...current, end: toNumber(e.target.value, current.end) } }))} /></td>
                    {channels.length > 0 && (
                      <td>
                        <select value={current.channel} onChange={(e) => setWindows((prev) => ({ ...prev, [index]: { ...current, channel: e.target.value } }))}>
                          {channels.map((ch) => <option key={ch} value={ch}>{ch}</option>)}
                        </select>
                      </td>
                    )}
                    <td><button type="button" onClick={() => recalcHplcPeak(index)} disabled={saving}>{t.manual.recalculate}</button></td>
                    <td>
                      <button
                        type="button"
                        className={activePlotPick?.mode === "hplc" && activePlotPick.index === index ? "pick-active" : ""}
                        onClick={() => requestHplcPick(index)}
                      >
                        {t.manual.selectOnPlot}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="manual-actions">
        <button type="button" onClick={save} disabled={saving}>{saving ? t.manual.saving : t.manual.save}</button>
        {dirty && <span className="dirty-indicator">{t.manual.unsaved}</span>}
        {message && <span role="status" aria-live="polite">{message}</span>}
      </div>

      {versions.length > 0 && (
        <div className="table-section compact-table">
          <h4>{t.manual.versions}</h4>
          <table>
            <thead>
              <tr>
                <th>{t.manual.version}</th>
                <th>{t.manual.time}</th>
                <th>{t.manual.peakCount}</th>
                <th>{t.manual.note}</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {versions.slice(0, 8).map((item) => (
                <tr key={item.version}>
                  <td>v{item.version}</td>
                  <td>{item.created_at}</td>
                  <td>{item.n_peaks}</td>
                  <td>{item.note || "manual confirmation"}</td>
                  <td><button type="button" onClick={() => setRestoreTarget(item.version)} disabled={saving} aria-label={`${t.manual.restore} v${item.version}`}>{t.manual.restore}</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <ConfirmDialog
        open={restoreTarget !== null}
        title={t.manual.restoreTitle}
        message={t.manual.restoreMessage.replace("{version}", String(restoreTarget ?? ""))}
        confirmLabel={t.manual.confirmRestore}
        cancelLabel={t.action.cancel}
        busyLabel={t.manual.saving}
        busy={saving}
        onConfirm={restoreVersion}
        onCancel={() => setRestoreTarget(null)}
      />
    </div>
  );
}
