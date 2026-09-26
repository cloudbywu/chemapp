import type { ReactNode } from "react";
import type {
  AnalysisResult,
  XrdCrystalliteSize,
  XrdLatticeParameters,
  XrdPeakAssignment,
  XrdPhaseMatch,
  XrdQuantitativePhase,
} from "../types/spectrum";
import { useLang } from "../i18n/LangContext";
import {
  MISSING_VALUE_PLACEHOLDER,
  finiteNumber,
  formatExponential,
  numberArray,
} from "../utils/number";

interface Props {
  result: AnalysisResult;
}

export default function XRDSPanel({ result }: Props) {
  const { t } = useLang();
  const metrics = result.metrics;
  const nPeaks = finiteNumber(metrics.n_peaks);
  const wavelengthAngstrom = finiteNumber(metrics.wavelength_a);
  const twoThetaRange = numberArray(metrics.two_theta_range);
  const maxIntensity = finiteNumber(metrics.max_intensity);
  const dSpacings = result.d_spacings || [];
  const crystalSizes = result.crystallite_sizes || [];
  const phaseMatches = result.phase_matches || [];
  const peakAssignments = result.peak_assignments || [];
  const latticeParams = result.lattice_parameters || [];
  const quantitative = result.quantitative_analysis || [];
  const rietveld = result.rietveld_refinement;
  const crystallinity = result.crystallinity;
  const williamsonHall = result.williamson_hall;
  const sizeDistribution = result.size_distribution;
  const structures = result.crystal_structure || [];

  return (
    <div className="analysis-panel">
      <h3>{t.xrd.title}</h3>
      <p className="summary">{result.summary}</p>

      <div className="metrics-grid">
        <div className="metric">
          <span className="metric-label">{t.xrd.peaks}</span>
          <span className="metric-value">{nPeaks ?? MISSING_VALUE_PLACEHOLDER}</span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.xrd.wavelength}</span>
          <span className="metric-value">
            {wavelengthAngstrom != null ? `${wavelengthAngstrom} Å` : MISSING_VALUE_PLACEHOLDER}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.xrd.twoThetaRange}</span>
          <span className="metric-value">
            {twoThetaRange.length >= 2
              ? `${twoThetaRange[0]}° – ${twoThetaRange[1]}°`
              : MISSING_VALUE_PLACEHOLDER}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">{t.xrd.maxIntensity}</span>
          <span className="metric-value">{formatExponential(maxIntensity, 2)}</span>
        </div>
        {typeof metrics.crystallinity_percent === "number" && (
          <div className="metric">
            <span className="metric-label">{t.xrd.crystallinity}</span>
            <span className="metric-value">{Number(metrics.crystallinity_percent).toFixed(1)}%</span>
          </div>
        )}
        {typeof metrics.dominant_phase === "string" && (
          <div className="metric">
            <span className="metric-label">{t.xrd.dominantPhase}</span>
            <span className="metric-value">{String(metrics.dominant_phase)}</span>
          </div>
        )}
      </div>

      {phaseMatches.length > 0 && (
        <TableSection title={t.xrd.qualitativeTitle}>
          <table>
            <thead>
              <tr>
                <th>{t.xrd.phase}</th>
                <th>{t.xrd.formula}</th>
                <th>{t.xrd.card}</th>
                <th>{t.xrd.system}</th>
                <th>{t.xrd.matched}</th>
                <th>{t.xrd.score}</th>
              </tr>
            </thead>
            <tbody>
              {phaseMatches.slice(0, 6).map((phase: XrdPhaseMatch, i: number) => (
                <tr key={i}>
                  <td>{phase.phase_name}</td>
                  <td>{phase.formula}</td>
                  <td>{phase.card_number}</td>
                  <td>{phase.crystal_system}</td>
                  <td>{phase.matched_peaks}/{phase.reference_peaks}</td>
                  <td>{phase.match_score.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableSection>
      )}

      {dSpacings.length > 0 && (
        <TableSection title={t.xrd.peaksDTitle}>
          <table>
            <thead>
              <tr>
                <th>2θ (°)</th>
                <th>{t.xrd.d}</th>
                <th>{t.xrd.intensity}</th>
              </tr>
            </thead>
            <tbody>
              {dSpacings.slice(0, 20).map((ds, i) => (
                <tr key={i}>
                  <td>{ds.two_theta.toFixed(3)}</td>
                  <td>{ds.d_angstrom.toFixed(4)}</td>
                  <td>{ds.intensity.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableSection>
      )}

      {peakAssignments.length > 0 && (
        <TableSection title={t.xrd.peakShiftTitle}>
          <table>
            <thead>
              <tr>
                <th>{t.xrd.obs2theta}</th>
                <th>{t.xrd.ref2theta}</th>
                <th>{t.xrd.shift}</th>
                <th>{t.xrd.hkl}</th>
                <th>{t.xrd.phase}</th>
                <th>{t.xrd.relI}</th>
              </tr>
            </thead>
            <tbody>
              {peakAssignments.slice(0, 24).map((row: XrdPeakAssignment, i: number) => (
                <tr key={i}>
                  <td>{row.observed_two_theta.toFixed(3)}</td>
                  <td>{row.reference_two_theta.toFixed(3)}</td>
                  <td>{row.peak_shift.toFixed(4)}</td>
                  <td>{row.hkl}</td>
                  <td>{row.phase_name}</td>
                  <td>{row.relative_intensity.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableSection>
      )}

      {latticeParams.length > 0 && (
        <TableSection title={t.xrd.latticeTitle}>
          <table>
            <thead>
              <tr>
                <th>{t.xrd.phase}</th>
                <th>{t.xrd.system}</th>
                <th>{t.xrd.a}</th>
                <th>{t.xrd.c}</th>
                <th>{t.xrd.ca}</th>
                <th>{t.xrd.indexed}</th>
              </tr>
            </thead>
            <tbody>
              {latticeParams.map((row: XrdLatticeParameters, i: number) => (
                <tr key={i}>
                  <td>{row.phase_name}</td>
                  <td>{row.crystal_system}</td>
                  <td>{row.a_angstrom?.toFixed(4) || "—"}</td>
                  <td>{row.c_angstrom?.toFixed(4) || "—"}</td>
                  <td>{row.c_over_a?.toFixed(4) || "—"}</td>
                  <td>{row.indexed_peaks}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableSection>
      )}

      {(crystallinity || williamsonHall || sizeDistribution) && (
        <div className="metrics-grid">
          {crystallinity && (
            <div className="metric">
              <span className="metric-label">{t.xrd.crystallineArea}</span>
              <span className="metric-value">{crystallinity.crystalline_area.toExponential(2)}</span>
            </div>
          )}
          {williamsonHall && Object.keys(williamsonHall).length > 0 && (
            <>
              <div className="metric">
                <span className="metric-label">{t.xrd.whSize}</span>
                <span className="metric-value">{williamsonHall.crystallite_size_nm.toFixed(2)} nm</span>
              </div>
              <div className="metric">
                <span className="metric-label">{t.xrd.microstrain}</span>
                <span className="metric-value">{williamsonHall.microstrain.toExponential(2)}</span>
              </div>
            </>
          )}
          {sizeDistribution && Object.keys(sizeDistribution).length > 0 && (
            <div className="metric">
              <span className="metric-label">{t.xrd.sizeMean}</span>
              <span className="metric-value">{(sizeDistribution.mean_a / 10).toFixed(2)} nm</span>
            </div>
          )}
        </div>
      )}

      {crystalSizes.length > 0 && (
        <TableSection title={t.xrd.scherrerTitle}>
          <table>
            <thead>
              <tr>
                <th>2θ (°)</th>
                <th>{t.xrd.fwhm}</th>
                <th>{t.xrd.sizeA}</th>
                <th>{t.xrd.sizeNm}</th>
              </tr>
            </thead>
            <tbody>
              {crystalSizes.map((cs: XrdCrystalliteSize, i: number) => (
                <tr key={i}>
                  <td>{cs.two_theta.toFixed(3)}</td>
                  <td>{cs.fwhm_deg.toFixed(4)}</td>
                  <td>{cs.size_a.toFixed(1)}</td>
                  <td>{(cs.size_nm ?? cs.size_a / 10).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableSection>
      )}

      {quantitative.length > 0 && (
        <TableSection title={t.xrd.quantitativeTitle}>
          <table>
            <thead>
              <tr>
                <th>{t.xrd.phase}</th>
                <th>{t.xrd.formula}</th>
                <th>{t.xrd.rir}</th>
                <th>{t.xrd.weightPercent}</th>
              </tr>
            </thead>
            <tbody>
              {quantitative.map((row: XrdQuantitativePhase, i: number) => (
                <tr key={i}>
                  <td>{row.phase_name}</td>
                  <td>{row.formula}</td>
                  <td>{row.rir.toFixed(2)}</td>
                  <td>{row.weight_percent.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableSection>
      )}

      {rietveld && Object.keys(rietveld).length > 0 && (
        <TableSection title={t.xrd.rietveld}>
          <div className="metrics-grid">
            <div className="metric">
              <span className="metric-label">{t.xrd.rwp}</span>
              <span className="metric-value">{rietveld.rwp.toFixed(2)}%</span>
            </div>
            <div className="metric">
              <span className="metric-label">{t.xrd.rb}</span>
              <span className="metric-value">{rietveld.rb.toFixed(2)}%</span>
            </div>
            <div className="metric">
              <span className="metric-label">{t.xrd.peakSigma}</span>
              <span className="metric-value">{rietveld.sigma_deg.toFixed(3)}°</span>
            </div>
          </div>
          <table>
            <thead>
              <tr>
                <th>{t.xrd.phase}</th>
                <th>{t.xrd.formula}</th>
                <th>{t.xrd.scale}</th>
                <th>{t.xrd.weightPercent}</th>
                <th>{t.xrd.matched}</th>
              </tr>
            </thead>
            <tbody>
              {rietveld.phases.map((row, i) => (
                <tr key={i}>
                  <td>{row.phase_name}</td>
                  <td>{row.formula}</td>
                  <td>{row.scale.toFixed(3)}</td>
                  <td>{row.weight_percent.toFixed(2)}%</td>
                  <td>{row.matched_peaks}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableSection>
      )}

      {structures.length > 0 && (
        <TableSection title={t.xrd.structureTitle}>
          <table>
            <thead>
              <tr>
                <th>{t.xrd.phase}</th>
                <th>{t.xrd.spaceGroup}</th>
                <th>{t.xrd.indexedPeaks}</th>
                <th>{t.xrd.refinement}</th>
              </tr>
            </thead>
            <tbody>
              {structures.slice(0, 5).map((row, i) => (
                <tr key={i}>
                  <td>{row.phase_name}</td>
                  <td>{row.space_group}</td>
                  <td>{row.indexed_peaks}</td>
                  <td>{row.refinement.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableSection>
      )}
    </div>
  );
}

function TableSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="table-section">
      <h4>{title}</h4>
      {children}
    </div>
  );
}
