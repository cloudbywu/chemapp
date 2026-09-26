import type { AnalysisResult } from "../types/spectrum";
import { useLang } from "../i18n/LangContext";
import {
  MISSING_VALUE_PLACEHOLDER,
  finiteNumber,
  formatFixed,
  numberArray,
  text,
} from "../utils/number";

interface Props {
  result: AnalysisResult;
}

export default function ElectrochemPanel({ result }: Props) {
  const { t } = useLang();
  const metrics = result.metrics;
  const subType = text(metrics.sub_type);

  if (subType === "EIS") {
    const zRangeReal = numberArray(metrics.z_range_real);
    return (
      <div className="analysis-panel">
        <h3>{t.electrochem.eisTitle}</h3>
        <p className="summary">{result.summary}</p>
        <div className="metrics-grid">
          <div className="metric">
            <span className="metric-label">{t.electrochem.rs}</span>
            <span className="metric-value">{formatFixed(finiteNumber(metrics.rs_ohm), 1)}</span>
          </div>
          <div className="metric">
            <span className="metric-label">{t.electrochem.rct}</span>
            <span className="metric-value">{formatFixed(finiteNumber(metrics.rct_ohm), 1)}</span>
          </div>
          <div className="metric">
            <span className="metric-label">{t.electrochem.zdMax}</span>
            <span className="metric-value">{formatFixed(finiteNumber(metrics.zd_max_ohm), 1)}</span>
          </div>
          <div className="metric">
            <span className="metric-label">{t.electrochem.zRange}</span>
            <span className="metric-value">
              {zRangeReal.length >= 2
                ? `${zRangeReal[0]} – ${zRangeReal[1]}`
                : MISSING_VALUE_PLACEHOLDER}
            </span>
          </div>
        </div>
      </div>
    );
  }

  const epA = finiteNumber(metrics.ep_anodic_v);
  const epC = finiteNumber(metrics.ep_cathodic_v);
  const ipA = finiteNumber(metrics.ip_anodic_a);
  const ipC = finiteNumber(metrics.ip_cathodic_a);
  const deltaEp = finiteNumber(metrics.delta_ep_v);
  const eFormal = finiteNumber(metrics.e_formal_v);
  const ipRatio = finiteNumber(metrics.ip_ratio);
  const qForward = finiteNumber(metrics.q_forward_c);
  const instrumentEp = finiteNumber(metrics.instrument_ep_v);
  const instrumentIp = finiteNumber(metrics.instrument_ip_a);
  const scanRate = finiteNumber(metrics.scan_rate_v_s);

  return (
    <div className="analysis-panel">
      <h3>
        {t.electrochem.cvTitle.replace(
          "{scanRate}",
          () => String(scanRate ?? MISSING_VALUE_PLACEHOLDER),
        )}
      </h3>
      <p className="summary">{result.summary}</p>

      <div className="metrics-grid">
        <div className="metric">
          <span className="metric-label">{t.electrochem.epAnodic}</span>
          <span className="metric-value">{formatFixed(epA, 4)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.electrochem.epCathodic}</span>
          <span className="metric-value">{formatFixed(epC, 4)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.electrochem.deltaEp}</span>
          <span className="metric-value">{deltaEp != null ? (deltaEp * 1000).toFixed(1) : MISSING_VALUE_PLACEHOLDER}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.electrochem.eFormal}</span>
          <span className="metric-value">{formatFixed(eFormal, 4)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.electrochem.ipAnodic}</span>
          <span className="metric-value">{ipA != null ? (ipA * 1e6).toFixed(2) : MISSING_VALUE_PLACEHOLDER}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.electrochem.ipCathodic}</span>
          <span className="metric-value">{ipC != null ? (ipC * 1e6).toFixed(2) : MISSING_VALUE_PLACEHOLDER}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.electrochem.ipRatio}</span>
          <span className="metric-value">{formatFixed(ipRatio, 3)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.electrochem.qForward}</span>
          <span className="metric-value">{qForward != null ? (qForward * 1e6).toFixed(2) : MISSING_VALUE_PLACEHOLDER}</span>
        </div>
      </div>

      {instrumentEp != null && (
        <div className="table-section">
          <h4>{t.electrochem.instrumentComparison}</h4>
          <table>
            <thead>
              <tr>
                <th></th>
                <th>{t.electrochem.ep}</th>
                <th>{t.electrochem.ip}</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{t.electrochem.parser}</td>
                <td>{formatFixed(epA, 4)}</td>
                <td>{ipA != null ? (ipA * 1e6).toFixed(2) : MISSING_VALUE_PLACEHOLDER}</td>
              </tr>
              <tr>
                <td>{t.electrochem.instrument}</td>
                <td>{formatFixed(instrumentEp, 3)}</td>
                <td>{formatFixed(instrumentIp, 3)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
