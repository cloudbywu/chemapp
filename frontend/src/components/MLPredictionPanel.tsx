import { useEffect, useId, useMemo, useRef, useState } from "react";
import MoleculeViewer from "./MoleculeViewer";
import {
  elucidateCombined,
  elucidateStructure,
  type ElucidationResponse,
} from "../services/api";
import type { AnalysisResult, Peak, SpectrumListItem } from "../types/spectrum";
import { useLang } from "../i18n/LangContext";
import { displayText, finiteNumber, record } from "../utils/number";

interface Candidate {
  rank: number;
  compoundName: string;
  rankingScore: number | null;
  smiles: string;
  molecularFormula: string;
  molecularWeight: number | null;
  source: string;
  sourceId: string;
  matched13c: number;
  matched1h: number;
  evidenceLevel: string;
  scoreBreakdown?: Record<string, unknown>;
  forwardEvidence?: ForwardEvidence;
}

interface ForwardAtomPrediction {
  atomIndex: number;
  shiftPpm: number;
}

interface ForwardEvidence {
  status: string;
  reasonCode: string;
  relativeFitRank: number | null;
  maePpm: number | null;
  rmsePpm: number | null;
  observedCoverage: number | null;
  predictedCoverage: number | null;
  atomPredictions: ForwardAtomPrediction[];
}

interface Props {
  spectrumId: string | null;
  hasResult: boolean;
  technique: string;
  result?: AnalysisResult | null;
  nucleus?: string;
  solvent?: string;
  spectra?: SpectrumListItem[];
}

function parseForwardEvidence(raw: unknown): ForwardEvidence | undefined {
  const value = record(raw);
  if (!value) return undefined;
  const predictionObject = record(value.prediction);
  const atomValues = Array.isArray(predictionObject?.atom_predictions)
    ? predictionObject.atom_predictions
    : [];
  const atomPredictions = atomValues.flatMap((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return [];
    const atom = item as Record<string, unknown>;
    const atomIndex = finiteNumber(atom.atom_index);
    const shiftPpm = finiteNumber(atom.shift_ppm);
    if (atomIndex == null || shiftPpm == null) return [];
    return [{ atomIndex, shiftPpm }];
  });
  return {
    status: displayText(value.status),
    reasonCode: displayText(value.reason_code),
    relativeFitRank: finiteNumber(value.relative_fit_rank),
    maePpm: finiteNumber(value.mae_ppm),
    rmsePpm: finiteNumber(value.rmse_ppm),
    observedCoverage: finiteNumber(value.observed_coverage),
    predictedCoverage: finiteNumber(value.predicted_coverage),
    atomPredictions,
  };
}

function parseCandidate(raw: Record<string, unknown>, index: number): Candidate {
  const scoreValue = raw.ranking_score ?? raw.match_score ?? raw.score;
  const numericScore = scoreValue == null ? null : Number(scoreValue);
  return {
    rank: Number(raw.rank || index + 1),
    compoundName: displayText(raw.compound_name || raw.name || raw.smiles || "?"),
    smiles: displayText(raw.smiles),
    rankingScore: numericScore != null && Number.isFinite(numericScore) ? numericScore : null,
    molecularFormula: displayText(raw.molecular_formula || raw.formula),
    molecularWeight: raw.molecular_weight == null ? null : Number(raw.molecular_weight),
    source: displayText(raw.source),
    sourceId: displayText(raw.source_id || raw.candidate_id),
    matched13c: Number(raw.matched_13c || 0),
    matched1h: Number(raw.matched_1h || 0),
    evidenceLevel: displayText(raw.evidence_level),
    scoreBreakdown: record(raw.score_breakdown) ?? undefined,
    forwardEvidence: parseForwardEvidence(raw.forward_evidence),
  };
}

function nearestIntegral(result: AnalysisResult | null | undefined, peak: Peak): number | null {
  if (peak.area != null && Number.isFinite(peak.area)) return peak.area;
  const nearest = (result?.integrals || [])
    .map((integral) => ({
      distance: Math.abs(integral.center_ppm - peak.position),
      value: integral.relative_area,
    }))
    .sort((a, b) => a.distance - b.distance)[0];
  return nearest && nearest.distance <= 0.08 ? nearest.value : null;
}

function nearestIntegralAtShift(
  result: AnalysisResult | null | undefined,
  shift: number,
): number | null {
  const nearest = (result?.integrals || [])
    .map((integral) => ({
      distance: Math.abs(integral.center_ppm - shift),
      value: integral.relative_area,
    }))
    .sort((a, b) => a.distance - b.distance)[0];
  return nearest && nearest.distance <= 0.08 ? nearest.value : null;
}

function candidateList(data: ElucidationResponse): Candidate[] {
  return (data.candidates || []).map(parseCandidate);
}

function nonEmptyLines(value: string): string[] {
  return value
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export default function MLPredictionPanel({
  spectrumId,
  hasResult,
  technique,
  result,
  nucleus,
  solvent,
  spectra,
}: Props) {
  const { t } = useLang();
  const formulaId = useId();
  const pairedIdLabel = useId();
  const experimentalGenerationId = useId();
  const candidateSmilesId = useId();
  const requiredSmartsId = useId();
  const forbiddenSmartsId = useId();
  const abortRef = useRef<AbortController | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");
  const [response, setResponse] = useState<ElucidationResponse | null>(null);
  const [pairedId, setPairedId] = useState("");
  const [formula, setFormula] = useState("");
  const [generateExperimental, setGenerateExperimental] = useState(false);
  const [candidateSmiles, setCandidateSmiles] = useState("");
  const [requiredSmarts, setRequiredSmarts] = useState("");
  const [forbiddenSmarts, setForbiddenSmarts] = useState("");

  useEffect(() => () => abortRef.current?.abort(), []);

  const pairedSpectra = useMemo(
    () => (spectra || []).filter((item) => item.id !== spectrumId && item.technique === "NMR"),
    [spectra, spectrumId],
  );

  if (technique !== "NMR") return null;

  const preparePeaks = (peaks: Peak[]) => peaks.map((peak) => ({
    shift: peak.position,
    intensity: peak.intensity || 1,
    integral: nearestIntegral(result, peak),
    multiplicity: peak.multiplicity || undefined,
    assignment: peak.assignment || undefined,
  }));
  const prepareObservedPeaks = () => {
    if (nucleus !== "13C" && result?.multiplets?.length) {
      return result.multiplets.map((multiplet) => ({
        shift: multiplet.center_ppm,
        intensity: multiplet.intensity_max || 1,
        integral: nearestIntegralAtShift(result, multiplet.center_ppm),
        multiplicity: multiplet.multiplicity || undefined,
      }));
    }
    return preparePeaks((result?.peaks as Peak[]) || []);
  };

  const runRequest = async (combined: boolean) => {
    if (!spectrumId) {
      setError(t.prediction.noSpectrum);
      return;
    }
    if (combined && !pairedId) {
      setError(t.prediction.choosePaired);
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setPredicting(true);
    setError("");
    setResponse(null);

    try {
      const constraints = {
        candidate_smiles: nonEmptyLines(candidateSmiles),
        required_smarts: nonEmptyLines(requiredSmarts),
        forbidden_smarts: nonEmptyLines(forbiddenSmarts),
      };
      if (combined) {
        setResponse(await elucidateCombined(
          spectrumId,
          pairedId,
          formula.trim() || undefined,
          generateExperimental,
          controller.signal,
          constraints,
        ));
      } else {
        const peaks = prepareObservedPeaks();
        setResponse(await elucidateStructure({
          peaks_1h: nucleus !== "13C" ? peaks : [],
          peaks_13c: nucleus === "13C" ? peaks : [],
          formula: formula.trim() || undefined,
          solvent: solvent || undefined,
          generate_experimental: generateExperimental,
          ...constraints,
          top_k: 5,
        }, controller.signal));
      }
    } catch (cause: unknown) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setError(cause instanceof Error ? cause.message : t.prediction.failed);
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setPredicting(false);
      }
    }
  };

  const candidates = response ? candidateList(response) : [];
  const experimental = (
    response?.experimental_hypotheses
    || response?.generated_candidates
    || []
  ).map(parseCandidate);
  const mixture = response?.mixture_analysis || null;
  const preprocessing = response?.query?.preprocessing;
  const evidenceLevel = response?.evidence_level
    || candidates[0]?.evidenceLevel
    || "";
  const evidenceLabels: Record<string, string> = {
    strong: t.prediction.evidenceStrong,
    moderate: t.prediction.evidenceModerate,
    weak: t.prediction.evidenceWeak,
    no_match: t.prediction.evidenceNoMatch,
    no_reference: t.prediction.evidenceNoReference,
  };
  const evidenceText = (value: string) => (
    evidenceLabels[value] || t.prediction.unknownEvidence
  );
  const evidenceClass = Object.hasOwn(evidenceLabels, evidenceLevel)
    ? evidenceLevel
    : "unknown";
  const decision = response?.decision;
  const abstained = decision?.action === "abstain";
  const top1Probability = response?.top1_calibrated_probability;
  const forwardModel = response?.forward_model;
  const forwardStatusLabels: Record<string, string> = {
    completed: t.prediction.forwardCompleted,
    partial: t.prediction.forwardPartial,
    unsupported_modality: t.prediction.forwardUnsupported,
    no_candidates: t.prediction.forwardNoCandidates,
    disabled: t.prediction.forwardDisabled,
    not_configured: t.prediction.forwardNotConfigured,
    busy: t.prediction.forwardBusy,
    timed_out: t.prediction.forwardTimedOut,
    unavailable: t.prediction.forwardUnavailable,
  };
  const forwardStatusText = forwardStatusLabels[String(forwardModel?.status || "")]
    || t.prediction.forwardUnknown;

  return (
    <div className="analysis-panel ml-prediction-panel">
      <h3>{t.prediction.title}</h3>
      <p className="summary">{t.prediction.description}</p>

      <div className="prediction-controls">
        <div className="settings-field">
          <label className="settings-label" htmlFor={formulaId}>{t.prediction.formula}</label>
          <input
            id={formulaId}
            value={formula}
            onChange={(event) => setFormula(event.target.value)}
            placeholder={t.prediction.formulaPlaceholder}
            autoComplete="off"
          />
          <span className="settings-hint">{t.prediction.formulaHint}</span>
        </div>
        <div className="settings-field">
          <label className="settings-label" htmlFor={pairedIdLabel}>{t.prediction.pairedSpectrum}</label>
          <select
            id={pairedIdLabel}
            value={pairedId}
            onChange={(event) => setPairedId(event.target.value)}
          >
            <option value="">{t.prediction.choosePaired}</option>
            {pairedSpectra.map((item) => (
              <option key={item.id} value={item.id}>{item.name || item.id}</option>
            ))}
          </select>
        </div>
        <div className="settings-field">
          <label className="checkbox-label" htmlFor={experimentalGenerationId}>
            <input
              id={experimentalGenerationId}
              type="checkbox"
              checked={generateExperimental}
              onChange={(event) => setGenerateExperimental(event.target.checked)}
            />
            <span>{t.prediction.enableExperimentalGeneration}</span>
          </label>
          <span className="settings-hint">{t.prediction.experimentalGenerationHint}</span>
        </div>
        <details className="prediction-constraints">
          <summary>{t.prediction.advancedConstraints}</summary>
          <p className="settings-hint">{t.prediction.advancedConstraintsHint}</p>
          <div className="settings-field">
            <label className="settings-label" htmlFor={candidateSmilesId}>
              {t.prediction.candidateSmiles}
            </label>
            <textarea
              id={candidateSmilesId}
              rows={3}
              value={candidateSmiles}
              onChange={(event) => setCandidateSmiles(event.target.value)}
              placeholder={t.prediction.candidateSmilesHint}
            />
          </div>
          <div className="settings-field">
            <label className="settings-label" htmlFor={requiredSmartsId}>
              {t.prediction.requiredSmarts}
            </label>
            <textarea
              id={requiredSmartsId}
              rows={2}
              value={requiredSmarts}
              onChange={(event) => setRequiredSmarts(event.target.value)}
              placeholder={t.prediction.smartsLineHint}
            />
          </div>
          <div className="settings-field">
            <label className="settings-label" htmlFor={forbiddenSmartsId}>
              {t.prediction.forbiddenSmarts}
            </label>
            <textarea
              id={forbiddenSmartsId}
              rows={2}
              value={forbiddenSmarts}
              onChange={(event) => setForbiddenSmarts(event.target.value)}
              placeholder={t.prediction.smartsLineHint}
            />
          </div>
        </details>
        <div className="prediction-actions">
          <button type="button" onClick={() => void runRequest(false)} disabled={predicting || !hasResult}>
            {predicting ? t.prediction.running : t.prediction.predict}
          </button>
          <button type="button" className="secondary-btn" onClick={() => void runRequest(true)} disabled={predicting || !hasResult || !pairedId}>
            {t.prediction.combined}
          </button>
          {predicting && (
            <button type="button" className="secondary-btn" onClick={() => abortRef.current?.abort()}>
              {t.prediction.cancel}
            </button>
          )}
        </div>
      </div>

      <div role="status" aria-live="polite" className="prediction-live">
        {predicting ? t.prediction.progress : ""}
      </div>
      {error && <p className="error" role="alert">{error}</p>}

      {response && (
        <>
          <section className={`evidence-banner evidence-${evidenceClass}`}>
            <strong>{t.prediction.evidence}: {evidenceText(evidenceLevel)}</strong>
            <span>
              {t.prediction.method}: {response.method || "—"}
              {response.inference_time_ms != null ? ` · ${Math.round(response.inference_time_ms)} ms` : ""}
            </span>
            {(response.query?.canonical_formula || response.query?.formula) && (
              <span>{t.prediction.canonicalFormula}: {response.query.canonical_formula || response.query.formula}</span>
            )}
            {response.candidate_pool_status && <span>{response.candidate_pool_status}</span>}
          </section>

          {abstained && (
            <section className="prediction-decision" role="status">
              <strong>{t.prediction.decisionAbstain}</strong>
              <span>
                {t.prediction.decisionReason}: {String(decision?.reason_code || "—")}
              </span>
              <span>
                {top1Probability != null
                  ? t.prediction.calibratedProbabilityHint
                  : t.prediction.notProbability}
              </span>
            </section>
          )}

          {top1Probability != null && (
            <section className="prediction-calibration" role="status">
              <strong>
                {t.prediction.calibratedProbability}:{" "}
                {(top1Probability * 100).toFixed(1)}%
              </strong>
              <span>{t.prediction.calibratedProbabilityHint}</span>
            </section>
          )}

          {(response.pipeline?.stages || []).length > 0 && (
            <details className="prediction-pipeline">
              <summary>{t.prediction.pipelineStages}</summary>
              <ol>
                {response.pipeline?.stages?.map((stage, index) => (
                  <li key={`${displayText(stage.stage) || "stage"}-${index}`}>
                    <strong>{displayText(stage.stage) || "—"}</strong>
                    <span>{displayText(stage.status) || "—"}</span>
                    {stage.reason_code ? <small>{displayText(stage.reason_code)}</small> : null}
                  </li>
                ))}
              </ol>
            </details>
          )}

          {(response.warnings || []).length > 0 && (
            <section className="prediction-warnings" aria-labelledby={`${formulaId}-warnings`}>
              <h4 id={`${formulaId}-warnings`}>{t.prediction.warnings}</h4>
              <ul>{response.warnings?.map((warning) => <li key={warning}>{warning}</li>)}</ul>
            </section>
          )}

          {forwardModel && (
            <section
              className="forward-shadow-panel"
              aria-labelledby={`${formulaId}-forward`}
            >
              <h4 id={`${formulaId}-forward`}>{t.prediction.forwardTitle}</h4>
              <p>{t.prediction.forwardDescription}</p>
              <div className="forward-shadow-status" role="status">
                <strong>{t.prediction.forwardStatus}: {forwardStatusText}</strong>
                {forwardModel.elapsed_ms != null && forwardModel.model_called
                  ? <span>{Math.round(forwardModel.elapsed_ms)} ms</span>
                  : null}
              </div>
            </section>
          )}

          {preprocessing && (
            <details className="prediction-preprocessing">
              <summary>{t.prediction.preprocessing}</summary>
              <dl>
                {Object.entries(preprocessing).map(([key, value]) => (
                  <div key={key}><dt>{key}</dt><dd>{String(value)}</dd></div>
                ))}
              </dl>
            </details>
          )}

          <section aria-labelledby={`${formulaId}-candidates`}>
            <h4 id={`${formulaId}-candidates`}>{t.prediction.rankedCandidates}</h4>
            {candidates.length === 0 ? (
              <p className="empty-hint">{t.prediction.noCandidates}</p>
            ) : (
              <div className="ml-candidates">
                {candidates.map((candidate) => (
                  <article key={candidate.sourceId || `${candidate.rank}-${candidate.smiles}`} className="ml-card">
                    <div className="ml-rank">#{candidate.rank}</div>
                    <div className="ml-info">
                      <div className="ml-name">{candidate.compoundName}</div>
                      <div className="ml-formula">
                        {candidate.molecularFormula || t.prediction.unknownFormula}
                        {candidate.source ? ` · ${candidate.source}` : ""}
                        {candidate.sourceId ? ` #${candidate.sourceId}` : ""}
                      </div>
                      <div className="ml-formula">
                        {t.prediction.matched}: <sup>13</sup>C {candidate.matched13c} · <sup>1</sup>H {candidate.matched1h}
                      </div>
                      {candidate.forwardEvidence?.status === "evaluated" && (
                        <div className="forward-fit-summary">
                          <span>
                            {t.prediction.forwardRelativeRank}: {candidate.forwardEvidence.relativeFitRank ?? "—"}
                          </span>
                          <span>
                            {t.prediction.forwardMae}: {candidate.forwardEvidence.maePpm?.toFixed(3) ?? "—"} ppm
                          </span>
                          <span>
                            {t.prediction.forwardRmse}: {candidate.forwardEvidence.rmsePpm?.toFixed(3) ?? "—"} ppm
                          </span>
                          <span>
                            {t.prediction.forwardObservedCoverage}: {
                              candidate.forwardEvidence.observedCoverage == null
                                ? "—"
                                : `${(candidate.forwardEvidence.observedCoverage * 100).toFixed(0)}%`
                            }
                          </span>
                          <span>
                            {t.prediction.forwardPredictedCoverage}: {
                              candidate.forwardEvidence.predictedCoverage == null
                                ? "—"
                                : `${(candidate.forwardEvidence.predictedCoverage * 100).toFixed(0)}%`
                            }
                          </span>
                          {candidate.forwardEvidence.atomPredictions.length > 0 && (
                            <details>
                              <summary>{t.prediction.forwardAtomDetails}</summary>
                              <table>
                                <thead>
                                  <tr>
                                    <th scope="col">{t.prediction.forwardAtom}</th>
                                    <th scope="col">{t.prediction.forwardShift}</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {candidate.forwardEvidence.atomPredictions.map((atom) => (
                                    <tr key={atom.atomIndex}>
                                      <td>C{atom.atomIndex}</td>
                                      <td>{atom.shiftPpm.toFixed(3)} ppm</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </details>
                          )}
                        </div>
                      )}
                      {candidate.forwardEvidence?.status === "skipped" && (
                        <div className="forward-fit-skipped">
                          {t.prediction.forwardSkipped}
                        </div>
                      )}
                      {candidate.smiles && <div className="ml-smiles">{candidate.smiles}</div>}
                    </div>
                    {candidate.smiles && <MoleculeViewer smiles={candidate.smiles} width={100} height={72} />}
                    <div className="ranking-score">
                      <span>{t.prediction.rankingScore}</span>
                      <strong>{candidate.rankingScore == null ? "—" : candidate.rankingScore.toFixed(4)}</strong>
                      {candidate.evidenceLevel && (
                        <small>{evidenceText(candidate.evidenceLevel)}</small>
                      )}
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>

          {experimental.length > 0 && (
            <details className="experimental-hypotheses">
              <summary>{t.prediction.experimentalHypotheses}</summary>
              <p className="warning-note">{t.prediction.experimentalWarning}</p>
              <ol>
                {experimental.map((candidate) => (
                  <li key={`${candidate.rank}-${candidate.smiles}`}>
                    <code>{candidate.smiles}</code>
                  </li>
                ))}
              </ol>
            </details>
          )}

          {mixture && (
            <section className="table-section">
              <h4>{t.prediction.mixtureTitle}</h4>
              <p className="hint">
                {mixture.is_mixture ? t.prediction.possibleMixture : t.prediction.singleComponent}
              </p>
            </section>
          )}

          {(response.data_attribution || []).length > 0 && (
            <footer className="prediction-attribution">
              <strong>{t.prediction.dataAttribution}</strong>
              <ul>
                {response.data_attribution?.map((item) => (
                  <li key={`${item.source}-${item.license_uri}`}>
                    {item.notice}{" "}
                    <a href={item.license_uri} target="_blank" rel="noreferrer">
                      {t.prediction.license}
                    </a>
                  </li>
                ))}
              </ul>
            </footer>
          )}
        </>
      )}
    </div>
  );
}
