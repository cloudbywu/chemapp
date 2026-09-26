import type { AnalysisOptions } from "../types/spectrum";
import { useLang } from "../i18n/LangContext";

interface Props {
  technique: string;
  value: AnalysisOptions;
  onChange: (value: AnalysisOptions) => void;
  disabled?: boolean;
}

type NumericOptionKey = Exclude<keyof AnalysisOptions, "baseline_correct" | "auto_reference" | "hide_solvent_peaks" | "multiplet_ranges" | "custom_phases" | "integration_events" | "rietveld_enabled">;

function updateNumber(value: AnalysisOptions, key: NumericOptionKey, raw: string): AnalysisOptions {
  if (raw.trim() === "") {
    // Clearing the field removes the override instead of writing 0
    // (Number("") === 0 would snap the input back to zero mid-edit).
    const next = { ...value };
    delete next[key];
    return next;
  }
  const parsed = Number(raw);
  if (Number.isFinite(parsed)) return { ...value, [key]: parsed };
  return value;
}

export default function AnalysisControls({ technique, value, onChange, disabled }: Props) {
  const { t } = useLang();
  const isNmr = technique === "NMR";
  const isUv = technique === "UV-Vis";
  const isHplc = technique === "HPLC";
  const isXrd = technique === "XRD";
  const customPhasesText = value.custom_phases ? JSON.stringify(value.custom_phases, null, 2) : "";

  if (!isNmr && !isUv && !isHplc && !isXrd) return null;

  return (
    <div className="analysis-controls">
      <div className="control-header">
        <h4>{t.controls.title}</h4>
        <button type="button" onClick={() => onChange({})} disabled={disabled}>{t.controls.reset}</button>
      </div>

      <div className="control-grid">
        {isNmr && (
          <>
            <label className="checkbox-control">
              <input type="checkbox" checked={value.auto_reference ?? false}
                onChange={(e) => onChange({ ...value, auto_reference: e.target.checked })} disabled={disabled} />
              {t.controls.autoReference}
            </label>
            <label className="checkbox-control">
              <input type="checkbox" checked={value.hide_solvent_peaks ?? false}
                onChange={(e) => onChange({ ...value, hide_solvent_peaks: e.target.checked })} disabled={disabled} />
              {t.controls.hideSolvent}
            </label>
            <label>
              {t.controls.solventTolerance}
              <input type="number" min={0.005} max={0.2} step={0.005} value={value.solvent_tolerance_ppm ?? 0.04}
                onChange={(e) => onChange(updateNumber(value, "solvent_tolerance_ppm", e.target.value))} disabled={disabled} />
            </label>
            <label>
              {t.controls.baselinePercent}
              <input type="number" min={0} max={50} step={1} value={value.baseline_percentile ?? 10}
                onChange={(e) => onChange(updateNumber(value, "baseline_percentile", e.target.value))} disabled={disabled} />
            </label>
            <label>
              {t.controls.noiseFactor}
              <input type="number" min={1} max={20} step={0.5} value={value.noise_factor ?? 3}
                onChange={(e) => onChange(updateNumber(value, "noise_factor", e.target.value))} disabled={disabled} />
            </label>
            <label>
              {t.controls.prominence}
              <input type="number" min={1} max={30} step={0.5} value={value.prominence_factor ?? 8}
                onChange={(e) => onChange(updateNumber(value, "prominence_factor", e.target.value))} disabled={disabled} />
            </label>
          </>
        )}

        {isUv && (
          <>
            <label className="checkbox-control">
              <input type="checkbox" checked={!!value.baseline_correct}
                onChange={(e) => onChange({ ...value, baseline_correct: e.target.checked })} disabled={disabled} />
              {t.controls.baselineCorrection}
            </label>
            <label>
              {t.controls.heightFraction}
              <input type="number" min={0.001} max={0.2} step={0.001} value={value.height_fraction ?? 0.01}
                onChange={(e) => onChange(updateNumber(value, "height_fraction", e.target.value))} disabled={disabled} />
            </label>
            <label>
              {t.controls.prominenceFraction}
              <input type="number" min={0.001} max={0.2} step={0.001} value={value.prominence_fraction ?? 0.02}
                onChange={(e) => onChange(updateNumber(value, "prominence_fraction", e.target.value))} disabled={disabled} />
            </label>
          </>
        )}

        {(isHplc || isXrd) && (
          <>
            <label>
              {t.controls.noiseFactor}
              <input type="number" min={0.5} max={20} step={0.5} value={value.noise_factor ?? (isXrd ? 3 : undefined) ?? 1.5}
                onChange={(e) => onChange(updateNumber(value, isXrd ? "noise_factor" : "height_factor", e.target.value))} disabled={disabled} />
            </label>
            <label>
              {t.controls.prominence}
              <input type="number" min={0.5} max={30} step={0.5} value={value.prominence_factor ?? (isXrd ? 5 : 2)}
                onChange={(e) => onChange(updateNumber(value, "prominence_factor", e.target.value))} disabled={disabled} />
            </label>
          </>
        )}

        {isXrd && (
          <>
            <label className="checkbox-control">
              <input type="checkbox" checked={value.rietveld_enabled ?? true}
                onChange={(e) => onChange({ ...value, rietveld_enabled: e.target.checked })} disabled={disabled} />
              {t.controls.rietveld}
            </label>
            <label>
              {t.controls.rietveldSigma}
              <input type="number" min={0.03} max={1.2} step={0.01} value={value.rietveld_sigma_deg ?? 0.18}
                onChange={(e) => onChange(updateNumber(value, "rietveld_sigma_deg", e.target.value))} disabled={disabled} />
            </label>
            <label className="wide-control">
              {t.controls.customPhases}
              <textarea
                rows={4}
                value={customPhasesText}
                placeholder='[{"name":"My phase","formula":"AB","crystal_system":"cubic","peaks":[{"two_theta":26.5,"hkl":"111","rel_intensity":100}]}]'
                onChange={(e) => {
                  const raw = e.target.value.trim();
                  if (!raw) {
                    onChange({ ...value, custom_phases: undefined });
                    return;
                  }
                  try {
                    const parsed = JSON.parse(raw);
                    onChange({ ...value, custom_phases: Array.isArray(parsed) ? parsed : [parsed] });
                  } catch {
                    onChange(value);
                  }
                }}
                disabled={disabled}
              />
            </label>
          </>
        )}

        <label>
          {t.controls.minDistance}
          <input type="number" min={1} max={200} step={1} value={value.min_peak_distance ?? (isNmr ? 5 : isHplc ? 2 : 10)}
            onChange={(e) => onChange(updateNumber(value, "min_peak_distance", e.target.value))} disabled={disabled} />
        </label>
      </div>
    </div>
  );
}
