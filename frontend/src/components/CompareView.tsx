import { useState } from "react";
import { compareSpectra } from "../services/api";
import { useLang } from "../i18n/LangContext";
import { MISSING_VALUE_PLACEHOLDER, finiteNumber } from "../utils/number";
import type { ComparisonResult, SpectrumListItem } from "../types/spectrum";
import FluorescencePanel from "./FluorescencePanel";

interface Props {
  spectra: SpectrumListItem[];
}

export default function CompareView({ spectra }: Props) {
  const { t } = useLang();
  const [id1, setId1] = useState("");
  const [id2, setId2] = useState("");
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState("");

  const handleCompare = async () => {
    if (!id1 || !id2) return;
    setError("");
    setComparing(true);
    try {
      const data = await compareSpectra(id1, id2);
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Comparison failed");
    } finally {
      setComparing(false);
    }
  };

  return (
    <div className="compare-view">
      <h3>{t.compare.title}</h3>
      <div className="compare-selects">
        <select value={id1} onChange={(e) => setId1(e.target.value)}>
          <option value="">{t.compare.select1}</option>
          {spectra.map((s) => (
            <option key={s.id} value={s.id}>
              [{s.technique}] {s.name || s.id}
            </option>
          ))}
        </select>
        <select value={id2} onChange={(e) => setId2(e.target.value)}>
          <option value="">{t.compare.select2}</option>
          {spectra.map((s) => (
            <option key={s.id} value={s.id}>
              [{s.technique}] {s.name || s.id}
            </option>
          ))}
        </select>
        <button onClick={handleCompare} disabled={comparing || !id1 || !id2}>
          {comparing ? t.compare.comparing : t.compare.compare}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="compare-result">
          <p>{result.technique1} vs {result.technique2}</p>
          <p>{t.compare.points}: {finiteNumber(result.shared.points1) ?? MISSING_VALUE_PLACEHOLDER} vs {finiteNumber(result.shared.points2) ?? MISSING_VALUE_PLACEHOLDER}</p>
          {result.stokes && (
            <FluorescencePanel stokes={result.stokes} />
          )}
          {!result.stokes && (
            <p className="hint">{t.compare.stokesHint}</p>
          )}
        </div>
      )}
    </div>
  );
}
