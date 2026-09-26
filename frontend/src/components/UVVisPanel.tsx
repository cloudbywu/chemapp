import type { AnalysisResult } from "../types/spectrum";
import { useLang } from "../i18n/LangContext";

interface Props {
  result: AnalysisResult;
}

export default function UVVisPanel({ result }: Props) {
  const { t } = useLang();
  const lambdaMax = result.lambda_max || [];
  const calibration = result.calibration;
  const concentration = result.sample_concentration;

  return (
    <div className="analysis-panel">
      <h3>{t.uvvis.title}</h3>
      <p className="summary">{result.summary}</p>

      {lambdaMax.length > 0 && (
        <div className="table-section">
          <h4>{t.uvvis.lambdaMax}</h4>
          <div className="lambda-list">
            {lambdaMax.map((wl, i) => (
              <span key={i} className="lambda-tag">{wl} nm</span>
            ))}
          </div>
        </div>
      )}

      {calibration && (
        <div className="calibration-section">
          <h4>{t.uvvis.calibration}</h4>
          <div className="metrics-grid">
            <div className="metric">
              <span className="metric-label">{t.uvvis.slope}</span>
              <span className="metric-value">{calibration.slope.toFixed(6)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">{t.uvvis.intercept}</span>
              <span className="metric-value">{calibration.intercept.toFixed(6)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">{t.uvvis.rSquared}</span>
              <span className="metric-value">{calibration.r_squared.toFixed(4)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">{t.uvvis.points}</span>
              <span className="metric-value">{calibration.n_points}</span>
            </div>
          </div>
          <p className="formula">
            Abs = {calibration.slope.toFixed(4)} · C + {calibration.intercept.toFixed(4)}
          </p>
        </div>
      )}

      {concentration != null && (
        <div className="concentration-result">
          <span className="metric-label">{t.uvvis.concentration}:</span>
          <span className="metric-value">{concentration.toFixed(2)} {result.concentration_unit || "mg/mL"}</span>
        </div>
      )}
    </div>
  );
}
