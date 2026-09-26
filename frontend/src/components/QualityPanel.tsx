import type { QualityReport } from "../types/spectrum";
import { useLang } from "../i18n/LangContext";

interface Props {
  quality?: QualityReport;
}

export default function QualityPanel({ quality }: Props) {
  const { t } = useLang();
  if (!quality) return null;
  const pct = Math.round(quality.score * 100);
  const statusLabel: Record<string, string> = {
    good: t.quality.good,
    review: t.quality.review,
    poor: t.quality.poor,
  };

  return (
    <div className={`quality-panel ${quality.status}`}>
      <div className="quality-head">
        <span>{t.quality.title}</span>
        <strong>{statusLabel[quality.status] || quality.status} · {pct}%</strong>
      </div>
      <div className="quality-bar">
        <div style={{ width: `${pct}%` }} />
      </div>
      {(quality.warnings.length > 0 || quality.info.length > 0) && (
        <ul className="quality-list">
          {quality.warnings.map((item, i) => <li key={`w-${i}`} className="warn">{item}</li>)}
          {quality.info.map((item, i) => <li key={`i-${i}`}>{item}</li>)}
        </ul>
      )}
    </div>
  );
}
