export interface Peak {
  position: number;
  intensity: number;
  area: number | null;
  width: number | null;
  assignment: string;
  multiplicity: string;
  coupling_constant: number | null;
}

export interface SpectrumData {
  id: string;
  spectrum_revision?: number;
  result_revision?: number;
  technique: string;
  x_data: number[];
  y_data: number[];
  x_label: string;
  y_label: string;
  x_unit: string;
  y_unit: string;
  parameters: Record<string, unknown>;
  metadata: Record<string, unknown>;
  peaks: Peak[];
  source_file: string;
}

export interface ChannelData {
  name: string;
  wavelength_nm: number | null;
  description: string;
  y_data: number[];
  color: string;
}

export interface SpectrumListItem {
  id: string;
  technique: string;
  points: number;
  name: string;
  has_result: boolean;
  summary: string;
  spectrum_revision?: number;
  result_revision?: number;
}

export type SpectrumReviewStatus =
  | "pending"
  | "awaiting_second"
  | "conflict"
  | "accepted"
  | "rejected"
  | "stale";

export type SpectrumReviewCheck = "pass" | "fail" | "uncertain";

export interface SpectrumReviewChecks {
  structure: SpectrumReviewCheck;
  nucleus: SpectrumReviewCheck;
  axis: SpectrumReviewCheck;
  peaks: SpectrumReviewCheck;
  solvent: SpectrumReviewCheck;
}

export interface SpectrumReviewObservations {
  structure_smiles: string;
  nucleus: string;
  axis_unit: string;
  axis_direction:
    | "ascending"
    | "descending"
    | "non_monotonic"
    | "unknown";
  solvent: string;
  peak_count: number;
}

export interface SpectrumReviewItem {
  schema_version: string;
  id: string;
  spectrum_id: string;
  cycle: number;
  status: SpectrumReviewStatus;
  queue_revision: number;
  structure_smiles: string;
  structure_source: string;
  molecule_id: string;
  source_collection: string;
  source_record_id: string;
  independence_group: string;
  license_id: string;
  provenance_uri: string;
  rights_confirmed: boolean;
  snapshot_sha256: string;
  spectrum_sha256: string;
  result_sha256: string | null;
  spectrum_revision: number;
  result_revision: number;
  facts: {
    technique: string;
    source_file: string;
    source_checksum_sha256: string | null;
    nucleus: string;
    solvent: string;
    axis_unit: string;
    axis_direction:
      | "ascending"
      | "descending"
      | "non_monotonic"
      | "unknown";
    point_count: number;
    peak_count: number;
    axis_min: number | null;
    axis_max: number | null;
  };
  review_count: number;
  has_submitted: boolean;
  own_review: {
    id: string;
    verdict: "accept" | "reject";
    checks: SpectrumReviewChecks;
    observations: SpectrumReviewObservations;
    notes: string;
    snapshot_sha256: string;
    created_at: string;
  } | null;
  final_decision: "accept" | "reject" | null;
  final_reason: string;
  created_at: string;
  updated_at: string;
}

export interface AnalysisOptions {
  baseline_percentile?: number;
  noise_factor?: number;
  prominence_factor?: number;
  min_peak_distance?: number;
  baseline_correct?: boolean;
  auto_reference?: boolean;
  hide_solvent_peaks?: boolean;
  solvent_tolerance_ppm?: number;
  multiplet_ranges?: Array<{ start: number; end: number }>;
  custom_phases?: Array<{
    name: string;
    formula?: string;
    database?: string;
    card_number?: string;
    crystal_system?: string;
    space_group?: string;
    rir?: number;
    peaks: Array<{ two_theta: number; hkl?: string; rel_intensity?: number; intensity?: number }>;
  }>;
  height_fraction?: number;
  prominence_fraction?: number;
  relative_prominence?: number;
  height_factor?: number;
  phase_zero_deg?: number;
  phase_first_deg?: number;
  phase_pivot_ppm?: number;
  rietveld_enabled?: boolean;
  rietveld_sigma_deg?: number;
  integration_events?: HplcIntegrationEvent[];
}

export interface HplcIntegrationEvent {
  channel?: string;
  start: number;
  end: number;
  mode: "delete" | "force_bb" | "force_vv" | "name" | "label" | "off";
  name?: string;
  baseline?: string;
  area_factor?: number;
}

export interface QualityReport {
  status: "good" | "review" | "poor";
  score: number;
  warnings: string[];
  info: string[];
  points: number;
}

export interface ExampleSpectrum {
  technique: string;
  label: string;
  path: string;
  filename: string;
  size: number;
}

export interface AnalysisResult {
  result_revision?: number;
  technique: string;
  peaks: Peak[];
  metrics: Record<string, unknown> & {
    quality?: QualityReport;
    analysis_options?: AnalysisOptions;
  };
  summary: string;
  integrals?: IntegralsItem[];
  multiplets?: Multiplet[];
  noise_level?: number;
  total_integral?: number;
  solvent_shift?: number | null;
  reference_corrected?: boolean;
  lambda_max?: number[];
  calibration?: Calibration | null;
  sample_concentration?: number | null;
  concentration_unit?: string;
  ex_peak?: number | null;
  em_peak?: number | null;
  stokes_shift_nm?: number | null;
  stokes_shift_cm1?: number | null;
  quantum_yield_ref?: unknown;
  normalized?: boolean;
  d_spacings?: XrdDSpacing[];
  crystallite_sizes?: XrdCrystalliteSize[];
  peak_assignments?: XrdPeakAssignment[];
  phase_matches?: XrdPhaseMatch[];
  lattice_parameters?: XrdLatticeParameters[];
  crystallinity?: XrdCrystallinity;
  williamson_hall?: XrdWilliamsonHall;
  size_distribution?: XrdSizeDistribution;
  quantitative_analysis?: XrdQuantitativePhase[];
  crystal_structure?: XrdCrystalStructure[];
  rietveld_refinement?: XrdRietveldRefinement;
}

export interface XrdDSpacing {
  two_theta: number;
  d_angstrom: number;
  intensity: number;
}

export interface XrdCrystalliteSize {
  two_theta: number;
  fwhm_deg: number;
  size_a: number;
  size_nm?: number;
}

export interface XrdPeakAssignment {
  phase_name: string;
  formula: string;
  hkl: string;
  observed_two_theta: number;
  reference_two_theta: number;
  peak_shift: number;
  fwhm_deg: number | null;
  d_angstrom: number;
  relative_intensity: number;
}

export interface XrdPhaseMatch {
  phase_name: string;
  formula: string;
  database: string;
  card_number: string;
  crystal_system: string;
  space_group: string;
  matched_peaks: number;
  reference_peaks: number;
  match_score: number;
}

export interface XrdLatticeParameters {
  phase_name: string;
  crystal_system: string;
  a_angstrom?: number;
  c_angstrom?: number;
  c_over_a?: number;
  a_std?: number;
  indexed_peaks: number;
  angular_correction: string;
}

export interface XrdCrystallinity {
  data_set_name: string;
  crystallinity_percent: number;
  crystalline_area: number;
  amorphous_area: number;
  total_area: number;
  method: string;
  peak_count_used: number;
}

export interface XrdWilliamsonHall {
  method: string;
  crystallite_size_a: number;
  crystallite_size_nm: number;
  microstrain: number;
  intercept: number;
  r_squared: number;
  points_used: number;
}

export interface XrdSizeDistribution {
  mean_a: number;
  median_a: number;
  std_a: number;
  min_a: number;
  max_a: number;
  bins: Array<{ min_a: number; max_a: number; count: number }>;
}

export interface XrdQuantitativePhase {
  phase_name: string;
  formula: string;
  rir: number;
  weight_percent: number;
  method: string;
}

export interface XrdCrystalStructure {
  phase_name: string;
  formula: string;
  crystal_system: string;
  space_group: string;
  database: string;
  card_number: string;
  indexed_peaks: number;
  lattice: XrdLatticeParameters;
  refinement: {
    measurement_range: string;
    refinement_range: string;
    refined_parameters: number;
    status: string;
  };
}

export interface XrdRietveldRefinement {
  method: string;
  status: string;
  rwp: number;
  rb: number;
  sigma_deg: number;
  background_constant: number;
  n_observations: number;
  n_parameters: number;
  phases: Array<{ phase_name: string; formula: string; scale: number; weight_percent: number; matched_peaks: number }>;
}

export interface IntegralsItem {
  center_ppm: number;
  start_ppm: number;
  end_ppm: number;
  raw_area: number;
  relative_area: number;
  intensity: number;
}

export interface Multiplet {
  center_ppm: number;
  range_ppm: [number, number];
  component_positions: number[];
  n_peaks: number;
  n_components: number;
  n_lines_used?: number;
  multiplicity?: string;
  j_values_hz?: number[];
  estimated_j_hz: number | null;
  intensity_max: number;
  grouping_source?: "automatic_peak_grouping" | "provided_range";
  is_isolated_singlet?: boolean;
  signal_to_noise?: number | null;
}

export interface Calibration {
  slope: number;
  intercept: number;
  r_squared: number;
  n_points: number;
}

export interface ComparisonResult {
  id1: string;
  id2: string;
  technique1: string;
  technique2: string;
  shared: Record<string, unknown>;
  stokes?: StokesResult;
}

export interface StokesResult {
  stokes_shift_nm: number;
  stokes_shift_cm1: number;
  excitation_peak_nm: number;
  emission_peak_nm: number;
}

export interface CrossValidationItem {
  pair: [string, string];
  metric: string;
  description: string;
  score: number;
  detail: string;
}

export interface InferenceResponse {
  sample_name: string;
  techniques: string[];
  inference: {
    technique_results: Record<string, Record<string, unknown>> & {
      _evidence_table?: { items: EvidenceItem[] };
    };
    cross_validations: CrossValidationItem[];
    consistency_score: number;
    confidence: number;
    anomalies: string[];
    conclusions: string[];
    overall_assessment: string;
  };
  generated_at: string;
  report_markdown: string;
}

export interface EvidenceItem {
  technique: string;
  evidence: string;
  support: string;
  confidence: number;
}
