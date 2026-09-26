import { useLang } from "../i18n/LangContext";
import type { SpectrumListItem } from "../types/spectrum";

interface Props {
  spectra: SpectrumListItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

const TECHNIQUE_COLORS: Record<string, string> = {
  NMR: "#3b82f6",
  "UV-Vis": "#10b981",
  Fluorescence: "#f59e0b",
  IR: "#ef4444",
  XRD: "#8b5cf6",
  HPLC: "#06b6d4",
  ElectroChem: "#ec4899",
};

export default function SpectrumList({ spectra, selectedId, onSelect, onDelete }: Props) {
  const { t } = useLang();

  if (spectra.length === 0) {
    return <p className="empty-hint">{t.list.empty}</p>;
  }

  return (
    <div className="spectrum-list">
      {spectra.map((s) => (
        <div
          key={s.id}
          className={`spectrum-card ${selectedId === s.id ? "selected" : ""}`}
        >
          <button
            type="button"
            className="spectrum-card-main"
            onClick={() => onSelect(s.id)}
            aria-pressed={selectedId === s.id}
            aria-label={t.list.selectAria.replace("{name}", () => s.name || s.id)}
          >
            <span
              className="tech-badge"
              style={{ background: TECHNIQUE_COLORS[s.technique] || "#6b7280" }}
            >
              {s.technique}
            </span>
            <span className="card-info">
              <span className="card-name">{s.name || s.id}</span>
              <span className="card-meta">{s.points} pts</span>
            </span>
            {s.has_result && <span className="badge-done">{t.list.analyzed}</span>}
          </button>
          <button
            type="button"
            className="delete-btn"
            onClick={() => onDelete(s.id)}
            title={t.list.delete}
            aria-label={t.list.deleteAria.replace("{name}", () => s.name || s.id)}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
