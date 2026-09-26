import type { AnalysisResult, IntegralsItem, Multiplet } from "../types/spectrum";
import { useLang } from "../i18n/LangContext";
import {
  MISSING_VALUE_PLACEHOLDER,
  finiteNumber,
  formatExponential,
} from "../utils/number";

interface Props {
  result: AnalysisResult;
}

export default function NMRPanel({ result }: Props) {
  const { t } = useLang();
  const metrics = result.metrics;
  const nPeaks = finiteNumber(metrics.n_peaks);
  const nMultiplets = finiteNumber(metrics.n_multiplets);
  const noiseLevel = finiteNumber(result.noise_level);
  const totalIntegral = finiteNumber(result.total_integral);
  const solventShift = finiteNumber(result.solvent_shift);
  const integrals: IntegralsItem[] = result.integrals || [];
  const multiplets: Multiplet[] = result.multiplets || [];

  return (
    <div className="analysis-panel">
      <h3>{t.nmr.title}</h3>
      <p className="summary">{result.summary}</p>

      <div className="metrics-grid">
        <div className="metric">
          <span className="metric-label">{t.nmr.peaks}</span>
          <span className="metric-value">{nPeaks ?? MISSING_VALUE_PLACEHOLDER}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.nmr.multiplets}</span>
          <span className="metric-value">{nMultiplets ?? MISSING_VALUE_PLACEHOLDER}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.nmr.noiseLevel}</span>
          <span className="metric-value">{formatExponential(noiseLevel, 2)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.nmr.totalIntegral}</span>
          <span className="metric-value">{formatExponential(totalIntegral, 2)}</span>
        </div>
        {result.reference_corrected && (
          <div className="metric">
            <span className="metric-label">{t.nmr.refShift}</span>
            <span className="metric-value">{solventShift ?? MISSING_VALUE_PLACEHOLDER} ppm</span>
          </div>
        )}
      </div>

      {integrals.length > 0 && (
        <div className="table-section">
          <h4>{t.nmr.integrals}</h4>
          <table>
            <thead>
              <tr>
                <th>{t.nmr.ppm}</th>
                <th>{t.nmr.area}</th>
                <th>{t.nmr.relArea}</th>
              </tr>
            </thead>
            <tbody>
              {integrals.slice(0, 20).map((item, i) => (
                <tr key={i}>
                  <td>{item.center_ppm.toFixed(4)}</td>
                  <td>{item.raw_area.toFixed(2)}</td>
                  <td>{item.relative_area}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {multiplets.length > 0 && (
        <div className="table-section">
          <h4>{t.nmr.multipletsTitle}</h4>
          <table>
            <thead>
              <tr>
                <th>{t.nmr.center}</th>
                <th>{t.nmr.type}</th>
                <th>{t.nmr.range}</th>
                <th>{t.nmr.count}</th>
                <th>{t.nmr.jHz}</th>
              </tr>
            </thead>
            <tbody>
              {multiplets.map((mp, i) => (
                <tr key={i}>
                  <td>{mp.center_ppm.toFixed(4)}</td>
                  <td>{mp.multiplicity || "m"}</td>
                  <td>{mp.range_ppm[0].toFixed(4)} – {mp.range_ppm[1].toFixed(4)}</td>
                  <td>{mp.n_components}</td>
                  <td>{mp.j_values_hz?.length ? mp.j_values_hz.map((j) => j.toFixed(2)).join(", ") : mp.estimated_j_hz?.toFixed(2) || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
