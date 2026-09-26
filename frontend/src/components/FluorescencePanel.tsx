import type { AnalysisResult, StokesResult } from "../types/spectrum";
import { useLang } from "../i18n/LangContext";
import { MISSING_VALUE_PLACEHOLDER, text } from "../utils/number";

interface Props {
  result?: AnalysisResult;
  stokes?: StokesResult | null;
}

function StokesShiftSummary({ stokes }: { stokes: StokesResult }) {
  const { t } = useLang();
  return (
    <div className="stokes-result">
      <h4>{t.fluor.paired}</h4>
      <p>{t.fluor.exToEm.replace("{ex}", String(stokes.excitation_peak_nm)).replace("{em}", String(stokes.emission_peak_nm))}</p>
      <p>Δλ = {stokes.stokes_shift_nm} nm | Δν = {stokes.stokes_shift_cm1} cm⁻¹</p>
    </div>
  );
}

export default function FluorescencePanel({ result, stokes }: Props) {
  const { t } = useLang();

  // Stokes-only lightweight path (e.g. CompareView pairing an excitation with
  // an emission spectrum): render the paired Stokes block without requiring a
  // full fluorescence analysis result.
  if (!result) {
    return stokes ? <StokesShiftSummary stokes={stokes} /> : null;
  }

  const metrics = result.metrics;

  return (
    <div className="analysis-panel">
      <h3>{t.fluor.title}</h3>
      <p className="summary">{result.summary}</p>

      <div className="metrics-grid">
        {result.ex_peak != null && (
          <div className="metric">
            <span className="metric-label">{t.fluor.exPeak}</span>
            <span className="metric-value">{result.ex_peak} nm</span>
          </div>
        )}
        {result.em_peak != null && (
          <div className="metric">
            <span className="metric-label">{t.fluor.emPeak}</span>
            <span className="metric-value">{result.em_peak} nm</span>
          </div>
        )}
        {result.stokes_shift_nm != null && (
          <div className="metric">
            <span className="metric-label">{t.fluor.stokesNm}</span>
            <span className="metric-value">{result.stokes_shift_nm} nm</span>
          </div>
        )}
        {result.stokes_shift_cm1 != null && (
          <div className="metric">
            <span className="metric-label">{t.fluor.stokesCm1}</span>
            <span className="metric-value">{result.stokes_shift_cm1} cm⁻¹</span>
          </div>
        )}
        <div className="metric">
          <span className="metric-label">{t.fluor.type}</span>
          <span className="metric-value">{text(metrics.sub_type) ?? MISSING_VALUE_PLACEHOLDER}</span>
        </div>
      </div>

      {stokes && <StokesShiftSummary stokes={stokes} />}
    </div>
  );
}
