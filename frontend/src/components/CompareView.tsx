import { useEffect, useRef, useState } from "react";
import { compareSpectra } from "../services/api";
import { useLang } from "../i18n/LangContext";
import { MISSING_VALUE_PLACEHOLDER, finiteNumber } from "../utils/number";
import type { ComparisonResult, SpectrumListItem } from "../types/spectrum";
import FluorescencePanel from "./FluorescencePanel";

interface Props {
  spectra: SpectrumListItem[];
}

type ComparisonState = {
  selectionKey: string;
  pending: boolean;
  result: ComparisonResult | null;
  error: string;
};

export default function CompareView({ spectra }: Props) {
  const { t } = useLang();
  const [id1, setId1] = useState("");
  const [id2, setId2] = useState("");
  const [comparison, setComparison] = useState<ComparisonState | null>(null);
  const requestSequence = useRef(0);
  const first = spectra.find((item) => item.id === id1);
  const second = spectra.find((item) => item.id === id2);
  // Bind feedback to the selected records, including later edits or deletions.
  const selectionKey = JSON.stringify([first?.id, first?.spectrum_revision, first?.result_revision,
    second?.id, second?.spectrum_revision, second?.result_revision]);
  const current = first && second && comparison?.selectionKey === selectionKey ? comparison : null;
  const result = current?.result;
  const error = current?.error;
  const comparing = current?.pending ?? false;

  useEffect(() => () => {
    requestSequence.current += 1;
  }, [selectionKey]);

  const changeSelection = (value: string, position: "first" | "second") => {
    requestSequence.current += 1;
    setComparison(null);
    if (position === "first") setId1(value);
    else setId2(value);
  };

  const handleCompare = async () => {
    if (!first || !second || comparing) return;
    const sequence = ++requestSequence.current;
    setComparison({ selectionKey, pending: true, result: null, error: "" });
    try {
      const data = await compareSpectra(id1, id2);
      if (sequence !== requestSequence.current) return;
      setComparison({ selectionKey, pending: false, result: data, error: "" });
    } catch (e: unknown) {
      if (sequence !== requestSequence.current) return;
      setComparison({ selectionKey, pending: false, result: null, error: e instanceof Error ? e.message : t.error.network });
    }
  };

  return (
    <div className="compare-view">
      <h3>{t.compare.title}</h3>
      <div className="compare-selects">
        <select aria-label={t.compare.select1} value={first ? id1 : ""} onChange={(e) => changeSelection(e.target.value, "first")}>
          <option value="">{t.compare.select1}</option>
          {spectra.map((s) => (
            <option key={s.id} value={s.id}>
              [{s.technique}] {s.name || s.id}
            </option>
          ))}
        </select>
        <select aria-label={t.compare.select2} value={second ? id2 : ""} onChange={(e) => changeSelection(e.target.value, "second")}>
          <option value="">{t.compare.select2}</option>
          {spectra.map((s) => (
            <option key={s.id} value={s.id}>
              [{s.technique}] {s.name || s.id}
            </option>
          ))}
        </select>
        <button type="button" onClick={() => void handleCompare()} disabled={comparing || !first || !second}>
          {comparing ? t.compare.comparing : t.compare.compare}
        </button>
      </div>
      {comparing && <p className="visually-hidden" role="status">{t.compare.comparing}</p>}
      {error && <p className="error" role="alert">{error}</p>}
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
