import type { AnalysisResult } from "../types/spectrum";
import { useLang } from "../i18n/LangContext";
import NMRPanel from "./NMRPanel";
import UVVisPanel from "./UVVisPanel";
import FluorescencePanel from "./FluorescencePanel";
import XRDSPanel from "./XRDSPanel";
import HPLCPanel from "./HPLCPanel";
import ElectrochemPanel from "./ElectrochemPanel";

interface Props {
  result: AnalysisResult;
  spectrumId?: string | null;
  onResultChanged?: (result: AnalysisResult) => void;
  onDirtyChange?: (dirty: boolean) => void;
}

export default function AnalysisPanel({
  result,
  spectrumId,
  onResultChanged,
  onDirtyChange,
}: Props) {
  const { t } = useLang();
  if (result.technique === "NMR") {
    return <NMRPanel result={result} />;
  }
  if (result.technique === "UV-Vis") {
    return <UVVisPanel result={result} />;
  }
  if (result.technique === "Fluorescence") {
    return <FluorescencePanel result={result} />;
  }
  if (result.technique === "XRD") {
    return <XRDSPanel result={result} />;
  }
  if (result.technique === "HPLC") {
    return <HPLCPanel result={result} spectrumId={spectrumId} onResultChanged={onResultChanged} onDirtyChange={onDirtyChange} />;
  }
  if (result.technique === "ElectroChem") {
    return <ElectrochemPanel result={result} />;
  }
  return (
    <div className="analysis-panel">
      <h3>{t.action.results}</h3>
      <p>{result.summary}</p>
      <pre>{JSON.stringify(result.metrics, null, 2)}</pre>
    </div>
  );
}
