import { useEffect, useMemo, useRef } from "react";
import type PlotlyType from "plotly.js";
import type { Data as PlotlyData, Layout as PlotlyLayout } from "plotly.js";
import { useLang } from "../i18n/LangContext";
import { displayText } from "../utils/number";
import type { AnalysisResult, ChannelData, SpectrumData } from "../types/spectrum";

export interface IntegrationRegion {
  start: number;
  end: number;
  label?: string;
  color?: string;
}

export interface IntegrationSelection {
  label: string;
  regions: IntegrationRegion[];
}

interface Props {
  spectrum: SpectrumData;
  result?: AnalysisResult | null;
  integrationSelection?: IntegrationSelection | null;
  onIntegrationRangeSelected?: (start: number, end: number) => void;
}

const MAX_POINTS = 8000;

function downsamplePairs(x: number[], y: number[], maxLen: number): { x: number[]; y: number[] } {
  const len = Math.min(x.length, y.length);
  if (len <= maxLen) return { x: x.slice(0, len), y: y.slice(0, len) };

  const bucketCount = Math.max(1, Math.floor(maxLen / 4));
  const step = len / bucketCount;
  const picked = new Set<number>();

  for (let bucket = 0; bucket < bucketCount; bucket++) {
    const start = Math.floor(bucket * step);
    const end = Math.min(len, Math.floor((bucket + 1) * step));
    if (start >= end) continue;

    let minIdx = start;
    let maxIdx = start;
    for (let i = start + 1; i < end; i++) {
      if (y[i] < y[minIdx]) minIdx = i;
      if (y[i] > y[maxIdx]) maxIdx = i;
    }

    picked.add(start);
    picked.add(minIdx);
    picked.add(maxIdx);
    picked.add(end - 1);
  }

  const indices = Array.from(picked).sort((a, b) => a - b).slice(0, maxLen);
  return {
    x: indices.map((i) => x[i]),
    y: indices.map((i) => y[i]),
  };
}

function percentile(values: number[], q: number): number {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  const next = sorted[base + 1];
  return next === undefined ? sorted[base] : sorted[base] + rest * (next - sorted[base]);
}

function minMax(values: number[]): { min: number; max: number } {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return { min, max };
}

function findNearestIndex(x: number[], target: number): number {
  if (x.length <= 1) return 0;
  const descending = x[0] > x[x.length - 1];
  let lo = 0;
  let hi = x.length - 1;
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (x[mid] === target) return mid;
    if (descending ? x[mid] > target : x[mid] < target) lo = mid + 1;
    else hi = mid - 1;
  }
  const a = Math.max(0, Math.min(x.length - 1, lo));
  const b = Math.max(0, Math.min(x.length - 1, lo - 1));
  return Math.abs(x[a] - target) < Math.abs(x[b] - target) ? a : b;
}

function getNmrWindow(spectrum: SpectrumData): { min: number; max: number } {
  const { min: fullMin, max: fullMax } = minMax(spectrum.x_data);
  if (!Number.isFinite(fullMin) || !Number.isFinite(fullMax)) return { min: 0, max: 1 };
  return { min: fullMin, max: fullMax };
}

function buildNmrDisplay(spectrum: SpectrumData) {
  const window = getNmrWindow(spectrum);
  const phaseCorrected = Boolean(spectrum.parameters?.phase_corrected);
  const sourceY = spectrum.y_data;
  const baseline = spectrum.parameters?.baseline_corrected ? 0 : percentile(sourceY, 0.5);
  const corrected = sourceY.map((value) => value - baseline);
  const visibleCorrected = corrected.filter((_, i) => spectrum.x_data[i] >= window.min && spectrum.x_data[i] <= window.max);
  const absoluteVisible = visibleCorrected.map(Math.abs);
  const scale = percentile(absoluteVisible, 0.9995) || minMax(absoluteVisible).max || 1;

  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < spectrum.x_data.length; i++) {
    const ppm = spectrum.x_data[i];
    if (ppm < window.min || ppm > window.max) continue;
    x.push(ppm);
    y.push(Math.max(-1.08, Math.min(corrected[i] / scale, 1.08)));
  }

  const sampled = downsamplePairs(x, y, MAX_POINTS);
  const extrema = minMax(sampled.y);
  const span = Math.max(extrema.max - extrema.min, 0.05);
  return {
    ...sampled,
    window,
    baseline,
    scale,
    yRange: [Math.min(0, extrema.min - span * 0.05), Math.max(0, extrema.max + span * 0.05)] as [number, number],
    phaseCorrected,
    shown: sampled.x.length,
    total: spectrum.x_data.length,
  };
}

function nmrDisplayIntensity(spectrum: SpectrumData, position: number, baseline: number, scale: number): number {
  const idx = findNearestIndex(spectrum.x_data, position);
  const corrected = spectrum.y_data[idx] - baseline;
  return Math.max(-1.08, Math.min(corrected / scale, 1.08));
}

function fmtPos(pos: number, technique: string): string {
  if (technique === "NMR") return pos.toFixed(4);
  if (technique === "HPLC") return pos.toFixed(3);
  return pos.toFixed(1);
}

// Dynamically load plotly.js to reduce initial bundle size
let _plotlyModule: typeof PlotlyType | null = null;
async function getPlotly(): Promise<typeof PlotlyType> {
  if (!_plotlyModule) {
    const module = await import("../plotlyCustom");
    const loaded = module as unknown as { default?: typeof PlotlyType };
    _plotlyModule = loaded.default || (module as unknown as typeof PlotlyType);
  }
  return _plotlyModule;
}

export default function SpectrumViewer({ spectrum, result, integrationSelection, onIntegrationRangeSelected }: Props) {
  const { t } = useLang();
  const containerRef = useRef<HTMLDivElement>(null);
  const xReverse = spectrum.technique === "NMR";
  const channels = spectrum.parameters?.channels as ChannelData[] | undefined;
  const isNmr = spectrum.technique === "NMR" && !(channels && channels.length > 1);
  const displayData = useMemo(() => {
    if (isNmr) return buildNmrDisplay(spectrum);
    const sampled = downsamplePairs(spectrum.x_data, spectrum.y_data, MAX_POINTS);
    return {
      ...sampled,
      shown: sampled.x.length,
      total: spectrum.x_data.length,
      window: null,
      baseline: 0,
      scale: 1,
      yRange: null,
      phaseCorrected: true,
    };
  }, [isNmr, spectrum]);

  useEffect(() => {
    let cancelled = false;
    let Plotly: typeof PlotlyType | null = null;

    void (async () => {
      Plotly = await getPlotly();
      if (cancelled || !containerRef.current) return;

      const traces: Array<Partial<PlotlyData>> = [];

      if (channels && channels.length > 1) {
        for (const ch of channels) {
          const sampled = downsamplePairs(spectrum.x_data, ch.y_data, MAX_POINTS);
          traces.push({
            x: sampled.x, y: sampled.y, type: "scatter", mode: "lines",
            name: `${ch.name} (${ch.wavelength_nm}nm)`,
            line: { color: ch.color, width: 1.5 },
            hoverinfo: "skip",
          });
        }
      } else {
        traces.push({
          x: displayData.x, y: displayData.y, type: "scatter", mode: "lines",
          name: spectrum.technique,
          line: { color: "#3b82f6", width: 1.5 },
          hoverinfo: "skip",
        });
      }

      // Channel peaks
      const channelPeaks = (result?.metrics as Record<string, unknown>)?.channel_peaks as
        Record<string, { wavelength_nm: number | null; color: string; peaks: Array<{ position: number; intensity: number; width: number | null; area: number }> }> | undefined;

      const manualConfirmed = Boolean((result?.metrics as Record<string, unknown>)?.manual_confirmed);
      if (!manualConfirmed && channelPeaks && Object.keys(channelPeaks).length > 0) {
        for (const [, chData] of Object.entries(channelPeaks)) {
          if (chData.peaks.length === 0) continue;
          const chPeaks = chData.peaks;
          const chLabelPeaks = chPeaks.length <= 15;
          traces.push({
            x: chPeaks.map((p) => p.position), y: chPeaks.map((p) => p.intensity),
            type: "scatter", mode: chLabelPeaks ? "text+markers" : "markers",
            name: `Peaks (${chData.wavelength_nm}nm)`,
            marker: { color: chData.color, size: 5, symbol: "x" },
            text: chLabelPeaks ? chPeaks.map((p) => p.position.toFixed(3)) : undefined,
            textposition: "top center",
            textfont: { family: "Inter, sans-serif", size: 9, color: chData.color },
            hoverinfo: "text",
            hovertext: chPeaks.map((p) => {
              const tR = p.position.toFixed(3);
              return `<b>tR ${tR} min</b><br>Height: ${p.intensity.toFixed(2)} mAU<br>Area: ${p.area.toFixed(1)}<br>Width: ${p.width != null ? p.width.toFixed(4) : "—"} min`;
            }),
            hovertemplate: "%{hovertext}<extra></extra>",
            hoverlabel: { bgcolor: "#1e293b", bordercolor: "#334155", font: { color: "#e2e8f0", size: 12, family: "Inter, sans-serif" } },
          });
        }
      } else if ((result?.peaks && result.peaks.length > 0) || (spectrum.peaks && spectrum.peaks.length > 0)) {
        const sourcePeaks = result?.peaks && result.peaks.length > 0 ? result.peaks : spectrum.peaks;
        const peaks = isNmr && displayData.window
          ? sourcePeaks.filter((p) => p.position >= displayData.window.min && p.position <= displayData.window.max)
          : sourcePeaks;
        const nPeaks = peaks.length;
        const labelPeaks = nPeaks <= 20;
        const unit = spectrum.technique === "NMR" ? "ppm" : spectrum.x_unit;

        const integrals = result?.integrals;
        const areaMap = new Map<number, number>();
        if (integrals) {
          for (const integ of integrals) {
            const dist = peaks.map((p, i) => ({ i, d: Math.abs(p.position - integ.center_ppm) }));
            dist.sort((a, b) => a.d - b.d);
            if (dist[0].d < 0.02) areaMap.set(dist[0].i, integ.raw_area);
          }
        }

        traces.push({
          x: peaks.map((p) => p.position),
          y: isNmr
            ? peaks.map((p) => nmrDisplayIntensity(spectrum, p.position, displayData.baseline, displayData.scale))
            : peaks.map((p) => p.intensity),
          type: "scatter", mode: labelPeaks ? "text+markers" : "markers",
          name: "Peaks", marker: { color: "#ef4444", size: nPeaks > 50 ? 4 : 6, symbol: "x" },
          text: labelPeaks ? peaks.map((p) => fmtPos(p.position, spectrum.technique)) : undefined,
          textposition: "top center", textfont: { family: "Inter, sans-serif", size: 9, color: "#f87171" },
          hoverinfo: "text",
          hovertext: peaks.map((p, i) => {
            const posStr = fmtPos(p.position, spectrum.technique);
            let html = `<b>${posStr} ${unit}</b><br>Intensity: ${p.intensity.toExponential(2)}`;
            const area = areaMap.get(i);
            if (area !== undefined) html += `<br>Area: ${area.toFixed(2)}`;
            return html;
          }),
          hovertemplate: "%{hovertext}<extra></extra>",
          hoverlabel: { bgcolor: "#1e293b", bordercolor: "#334155", font: { color: "#e2e8f0", size: 12, family: "Inter, sans-serif" } },
        });
      }

      const titleText = channels && channels.length > 1
        ? `HPLC — ${channels.map((c) => `${c.name}`).join(" + ")}`
        : `${spectrum.technique} — ${displayText(spectrum.parameters?.nucleus) || displayText(spectrum.metadata?.name)}`;

      const layout: Partial<PlotlyLayout> = {
        title: { text: titleText, font: { color: "#cbd5e1", size: 14 } },
        font: { family: "Inter, sans-serif", color: "#cbd5e1" },
        xaxis: {
          title: { text: `${spectrum.x_label} (${spectrum.x_unit})`, font: { color: "#94a3b8" } },
          autorange: isNmr ? false : xReverse ? "reversed" : true,
          range: isNmr && displayData.window ? [displayData.window.max, displayData.window.min] : undefined,
          gridcolor: "rgba(148,163,184,0.18)",
          zerolinecolor: "rgba(148,163,184,0.25)",
          tickfont: { color: "#94a3b8" },
        },
        yaxis: {
          title: { text: isNmr ? "Normalized intensity" : `${spectrum.y_label} (${spectrum.y_unit})`, font: { color: "#94a3b8" } },
          range: isNmr ? displayData.yRange || undefined : undefined,
          gridcolor: "rgba(148,163,184,0.18)",
          zerolinecolor: "rgba(148,163,184,0.25)",
          tickfont: { color: "#94a3b8" },
        },
        margin: { l: 60, r: 20, t: 50, b: 50 },
        paper_bgcolor: "transparent", plot_bgcolor: "transparent",
        hovermode: "closest",
        dragmode: integrationSelection ? "select" : "zoom",
        selectdirection: "h",
        showlegend: !!(channels && channels.length > 1),
        legend: { x: 1, y: 1, xanchor: "right", bgcolor: "rgba(17,24,39,0.9)", bordercolor: "#334155", font: { color: "#e2e8f0", size: 11 } },
        shapes: integrationSelection?.regions.map((region) => ({
          type: "rect",
          xref: "x",
          yref: "paper",
          x0: Math.min(region.start, region.end),
          x1: Math.max(region.start, region.end),
          y0: 0,
          y1: 1,
          fillcolor: region.color || "rgba(79,140,255,0.18)",
          line: { color: region.color || "rgba(79,140,255,0.65)", width: 1 },
          layer: "below",
        })),
        annotations: integrationSelection ? [{
          xref: "paper",
          yref: "paper",
          x: 0,
          y: 1.08,
          xanchor: "left",
          showarrow: false,
          text: `${t.viewer.selecting}: ${integrationSelection.label}`,
          font: { color: "#fbbf24", size: 12 },
        }] : undefined,
      };

      await Plotly.react(containerRef.current, traces, layout, {
        responsive: true, displayModeBar: true,
        modeBarButtonsToRemove: integrationSelection ? ["lasso2d", "autoScale2d"] : ["lasso2d", "select2d", "autoScale2d"],
      });

      const plotDiv = containerRef.current as unknown as {
        on?: (name: string, handler: (event: unknown) => void) => void;
        removeAllListeners?: (name: string) => void;
      };
      plotDiv.removeAllListeners?.("plotly_selected");
      if (integrationSelection && onIntegrationRangeSelected) {
        plotDiv.on?.("plotly_selected", (event: unknown) => {
          const range = (event as { range?: { x?: [number, number] } | undefined } | null)?.range?.x;
          if (!range || range.length < 2) return;
          onIntegrationRangeSelected(Math.min(range[0], range[1]), Math.max(range[0], range[1]));
        });
      }
    })();

    return () => { cancelled = true; };
  }, [displayData, spectrum, xReverse, result, channels, isNmr,
      integrationSelection, onIntegrationRangeSelected, t.viewer.selecting]);

  useEffect(() => {
    const plotElement = containerRef.current;
    return () => {
      if (plotElement && _plotlyModule) _plotlyModule.purge(plotElement);
    };
  }, []);

  const resetView = async () => {
    if (!containerRef.current) return;
    const Plotly = await getPlotly();
    // plotly.js 4 的 relayout 类型只声明 Partial<Layout>，但运行时仍支持
    // "xaxis.range" 形式的更新键；交叉 Record<string, unknown> 保留该用法。
    type RelayoutUpdate = Partial<PlotlyLayout> & Record<string, unknown>;
    if (isNmr && displayData.window) {
      const update: RelayoutUpdate = {
        "xaxis.range": [displayData.window.max, displayData.window.min],
        "yaxis.range": displayData.yRange,
      };
      await Plotly.relayout(containerRef.current, update);
    } else {
      const update: RelayoutUpdate = {
        "xaxis.autorange": true,
        "yaxis.autorange": true,
      };
      await Plotly.relayout(containerRef.current, update);
    }
  };

  return (
    <div className="spectrum-viewer">
      <div className="viewer-toolbar">
        <button type="button" className="secondary-btn" onClick={() => void resetView()}>
          {t.viewer.fullSpectrum}
        </button>
        {isNmr && !displayData.phaseCorrected && (
          <span className="warning-note">{t.viewer.unphased}</span>
        )}
      </div>
      {displayData.total > displayData.shown && (
        <p style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>
          {t.viewer.downsampled.replace("{shown}", String(displayData.shown)).replace("{total}", String(displayData.total))}
        </p>
      )}
      {integrationSelection && (
        <p className="plot-selection-hint">
          {t.viewer.selectRange} {t.viewer.selectRangeKeyboard}
        </p>
      )}
      <div
        ref={containerRef}
        role="img"
        aria-label={t.viewer.plotAria.replace("{technique}", spectrum.technique)}
        style={{ width: "100%", height: "400px" }}
      />
    </div>
  );
}
