import axios from "axios";
import type {
  AnalysisOptions,
  AnalysisResult,
  ComparisonResult,
  ExampleSpectrum,
  InferenceResponse,
  SpectrumData,
  SpectrumListItem,
  SpectrumReviewChecks,
  SpectrumReviewItem,
  SpectrumReviewObservations,
} from "../types/spectrum";
import { chemAppAuthHeaders } from "./authTokens";
import { needsAdmin, needsReviewer } from "./authRoutes";

export const API_BASE = String(import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export const api = axios.create({ baseURL: API_BASE, timeout: 30000 });

/**
 * Per-request timeout overrides for large transfers.
 *
 * nginx (frontend/nginx.conf) allows request bodies up to 50MB
 * (client_max_body_size 50m) and keeps the upstream connection open for
 * 600s (proxy_read_timeout 600s) under /api/. The 30s global axios default
 * would abort long before those limits are reached, so uploads get the full
 * 600s window — aligned with the nginx proxy_read_timeout ceiling — and
 * large report/data downloads get a 300s budget, comfortably inside it.
 */
export const UPLOAD_TIMEOUT_MS = 600_000;
export const REPORT_DOWNLOAD_TIMEOUT_MS = 300_000;

/**
 * Trigger a browser download for a blob payload, then release the object URL.
 * Shared by every export/download endpoint instead of repeating the
 * createObjectURL + anchor + revoke boilerplate.
 */
export function downloadBlob(data: Blob, filename: string): void {
  const url = URL.createObjectURL(data);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function requestNeedsAdmin(method: string, requestUrl: string): boolean {
  return needsAdmin(method, requestUrl);
}

export function requestNeedsReviewer(
  method: string,
  requestUrl: string,
): boolean {
  return needsReviewer(method, requestUrl);
}

api.interceptors.request.use((config) => {
  const method = String(config.method || "get").toLowerCase();
  const url = String(config.url || "");
  Object.assign(
    config.headers,
    chemAppAuthHeaders(
      requestNeedsAdmin(method, url),
      requestNeedsReviewer(method, url),
    ),
  );
  return config;
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (axios.isCancel(err) || err.code === "ERR_CANCELED") {
      return Promise.reject(new DOMException("Request aborted", "AbortError"));
    }
    if (err.response) {
      const detail = err.response.data?.detail || err.response.statusText;
      const message = typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((item) => String(item?.msg || item)).join("; ")
          : detail && typeof detail === "object"
            ? String(detail.message || detail.code || JSON.stringify(detail))
            : "Request failed";
      return Promise.reject(new ApiError(message, err.response.status));
    }
    if (err.code === "ECONNABORTED") {
      return Promise.reject(new ApiError("Request timed out — try again"));
    }
    return Promise.reject(new ApiError("Network error — check the backend connection"));
  }
);

export async function uploadFile(file: File): Promise<SpectrumListItem[]> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/api/upload", form, { timeout: UPLOAD_TIMEOUT_MS });
  return Array.isArray(data) ? data : [data];
}

export async function listSpectra(signal?: AbortSignal): Promise<SpectrumListItem[]> {
  const { data } = await api.get("/api/spectra", { signal });
  return data;
}

export async function getSpectrum(id: string, signal?: AbortSignal): Promise<SpectrumData> {
  const { data } = await api.get(`/api/spectra/${id}`, { signal });
  return data;
}

export async function processNmrSpectrum(
  id: string,
  payload: {
    baseline_correct?: boolean;
    baseline_method?: "als" | "asymmetric_least_squares" | "percentile";
    baseline_smoothness?: number;
    baseline_asymmetry?: number;
    baseline_iterations?: number;
    baseline_percentile?: number;
    auto_phase?: boolean;
    auto_phase_first_order?: boolean;
    auto_phase_max_first_deg?: number;
    auto_reference?: boolean;
    reference_solvent?: string;
    reference_window_ppm?: number;
    reference_min_snr?: number;
    reference_current_ppm?: number;
    reference_target_ppm?: number;
    normalize?: boolean;
    normalize_ppm?: number;
    normalize_window_points?: number;
    invert?: boolean;
    smoothing_window?: number;
    crop_min_ppm?: number;
    crop_max_ppm?: number;
    phase_zero_deg?: number;
    phase_first_deg?: number;
    phase_pivot_ppm?: number;
    preview_only?: boolean;
    replay_from_original?: boolean;
    expected_revision: number;
  }
): Promise<SpectrumData & {
  processing_applied?: Array<Record<string, unknown>>;
  processing_skipped?: Array<Record<string, unknown>>;
  warnings?: string[];
  quality_before?: Record<string, unknown>;
  quality_after?: Record<string, unknown>;
  processing_status?: {
    state?: string;
    committed?: boolean;
    phase?: string;
    revision?: number;
    available_views?: string[];
    source?: Record<string, unknown> | null;
  };
  state?: "preview" | "applied" | "reset";
  committed?: boolean;
}> {
  const { data } = await api.post(`/api/nmr/${id}/process`, payload);
  return data;
}

export async function getNmrSpectrumView(
  id: string,
  state: "original" | "current",
  signal?: AbortSignal,
): Promise<SpectrumData & {
  view?: "original" | "current";
  quality_metrics?: Record<string, unknown>;
  spectrum_revision?: number;
}> {
  const { data } = await api.get(`/api/nmr/${id}/view`, {
    params: { state },
    signal,
  });
  return data;
}

export async function resetNmrSpectrum(
  id: string,
  expectedRevision: number,
): Promise<SpectrumData & {
  processing_applied?: Array<Record<string, unknown>>;
  warnings?: string[];
  state?: "reset";
  committed?: boolean;
}> {
  const { data } = await api.post(`/api/nmr/${id}/reset`, {
    expected_revision: expectedRevision,
  });
  return data;
}

export async function deleteSpectrum(
  id: string,
  expectedSpectrumRevision: number,
  expectedResultRevision: number,
): Promise<void> {
  await api.delete(`/api/spectra/${id}`, {
    params: {
      expected_spectrum_revision: expectedSpectrumRevision,
      expected_result_revision: expectedResultRevision,
    },
  });
}

export async function analyzeSpectrum(
  id: string,
  options: AnalysisOptions | undefined,
  signal: AbortSignal | undefined,
  expectedRevision: number,
  forceOverwrite = false,
): Promise<AnalysisResult> {
  const { data } = await api.post(`/api/analyze/${id}`, {
    options: options || {},
    expected_revision: expectedRevision,
    force_overwrite: forceOverwrite,
  }, { signal });
  return data;
}

export async function analyzeBatch(
  ids: string[],
  expectedRevisions: Record<string, number>,
  options?: AnalysisOptions,
  forceOverwrite = false,
): Promise<{
  results: Array<{ id: string; technique: string; name: string; n_peaks: number; summary: string; quality?: unknown; metrics: Record<string, unknown>; result_revision: number }>;
  errors: Array<{ id: string; error: string; current_revision?: number }>;
}> {
  const { data } = await api.post("/api/analyze/batch", {
    ids,
    options: options || {},
    expected_revisions: expectedRevisions,
    force_overwrite: forceOverwrite,
  });
  return data;
}

export async function getBatchWorkbench(
  ids?: string[],
  limit: number = 1000,
  offset: number = 0,
): Promise<{
  items: Array<{ id: string; name: string; technique: string; points: number; n_peaks: number; quality: unknown; manual_confirmed: boolean; ai_modified: boolean; summary: string; export_ready: boolean }>;
  summary: { count: number; technique_counts: Record<string, number>; quality_counts: Record<string, number>; manual_confirmed: number; total_points: number };
}> {
  const { data } = await api.post("/api/batch/workbench", { ids: ids || [] }, {
    params: { limit, offset },
  });
  return data;
}

export async function getResult(id: string, signal?: AbortSignal): Promise<AnalysisResult> {
  const { data } = await api.get(`/api/results/${id}`, { signal });
  return data;
}

export async function integrateRanges(
  id: string,
  ranges: Array<{ start: number; end: number; center?: number; channel?: string; baseline?: string }>
): Promise<{ integrals: Array<{ start: number; end: number; center: number; area: number; height?: number; width?: number; apex_time?: number; channel?: string; baseline_start?: number; baseline_end?: number }> }> {
  const { data } = await api.post(`/api/results/${id}/integrate`, { ranges });
  return data;
}

export async function saveManualResult(
  id: string,
  payload: Pick<AnalysisResult, "peaks" | "metrics" | "summary"> & { integrals?: AnalysisResult["integrals"]; multiplets?: AnalysisResult["multiplets"]; channel_peaks?: unknown; note?: string; expected_revision: number }
): Promise<AnalysisResult> {
  const { data } = await api.put(`/api/results/${id}/manual`, payload);
  return data;
}

export async function listResultVersions(id: string): Promise<{
  versions: Array<{ version: number; note: string; created_at: string; n_peaks: number; manual_confirmed: boolean; summary: string }>;
}> {
  const { data } = await api.get(`/api/results/${id}/versions`);
  return data;
}

export async function restoreResultVersion(id: string, version: number, expectedRevision: number): Promise<AnalysisResult> {
  const { data } = await api.post(`/api/results/${id}/versions/${version}/restore`, {
    expected_revision: expectedRevision,
  });
  return data;
}

export async function getSpectrumReview(
  id: string,
  signal?: AbortSignal,
): Promise<{ reviewer_id: string; item: SpectrumReviewItem }> {
  const { data } = await api.get(`/api/reviews/${id}`, { signal });
  return data;
}

export async function getSpectrumReviewIdentity(): Promise<{
  reviewer_id: string;
}> {
  const { data } = await api.get("/api/reviews/me");
  return data;
}

export async function getSpectrumReviewCapabilities(): Promise<{
  reviewer: {
    configured: boolean;
    authenticated: boolean;
    subject: string;
  };
  admin: {
    configured: boolean;
    authenticated: boolean;
    subject: string;
  };
  separation_ok: boolean;
  can_review: boolean;
  can_manage_queue: boolean;
  can_view_audit: boolean;
  can_adjudicate: boolean;
  can_export_gold: boolean;
}> {
  const { data } = await api.get("/api/reviews/capabilities");
  return data;
}

export async function listSpectrumReviewQueue(
  status?: string,
  signal?: AbortSignal,
  asAdmin = false,
): Promise<{
  reviewer_id?: string;
  items: SpectrumReviewItem[];
  count: number;
}> {
  const path = asAdmin ? "/api/reviews/admin/queue" : "/api/reviews/queue";
  const { data } = await api.get(path, {
    params: status ? { status } : undefined,
    signal,
  });
  return data;
}

export async function getSpectrumReviewAudit(id: string): Promise<{
  queue: SpectrumReviewItem;
  reviews: Array<{
    id: string;
    reviewer_id: string;
    verdict: "accept" | "reject";
    checks: SpectrumReviewChecks;
    observations: SpectrumReviewObservations;
    notes: string;
    snapshot_sha256: string;
    created_at: string;
  }>;
  events: Array<Record<string, unknown>>;
}> {
  const { data } = await api.get(`/api/reviews/${id}/audit`);
  return data;
}

export async function enqueueSpectrumReview(payload: {
  spectrum_id: string;
  structure_smiles: string;
  structure_source: string;
  molecule_id: string;
  source_collection: string;
  source_record_id: string;
  independence_group: string;
  license_id: string;
  provenance_uri: string;
  rights_confirmed: boolean;
  expected_spectrum_revision: number;
  expected_result_revision: number;
}): Promise<SpectrumReviewItem> {
  const { data } = await api.post("/api/reviews/queue", payload);
  return data;
}

export async function submitSpectrumReview(
  id: string,
  payload: {
    verdict: "accept" | "reject";
    checks: SpectrumReviewChecks;
    observations: SpectrumReviewObservations;
    notes: string;
    expected_queue_revision: number;
    expected_snapshot_sha256: string;
  },
): Promise<{ reviewer_id: string; item: SpectrumReviewItem }> {
  const { data } = await api.post(`/api/reviews/${id}/submit`, payload);
  return data;
}

export async function adjudicateSpectrumReview(
  id: string,
  payload: {
    decision: "accept" | "reject";
    checks: SpectrumReviewChecks;
    reason: string;
    expected_queue_revision: number;
    expected_snapshot_sha256: string;
  },
): Promise<SpectrumReviewItem> {
  const { data } = await api.post(`/api/reviews/${id}/adjudicate`, payload);
  return data;
}

export async function downloadGoldSpectrumManifest(): Promise<void> {
  const { data } = await api.get("/api/reviews/gold-manifest", {
    responseType: "blob",
    timeout: REPORT_DOWNLOAD_TIMEOUT_MS,
  });
  downloadBlob(data, "chemapp-nmr-gold-manifest-v1.json");
}

export async function compareSpectra(id1: string, id2: string): Promise<ComparisonResult> {
  const { data } = await api.post("/api/compare", { id1, id2 });
  return data;
}

export async function compareHplcBatch(ids: string[], channel?: string, rtTolerance = 0.08): Promise<{
  reference_id: string;
  channel: string;
  rt_tolerance: number;
  rows: Array<{
    peak_index: number;
    reference_rt: number;
    reference_area: number;
    reference_type: string;
    name: string;
    samples: Array<{ id: string; name: string; matched: boolean; rt?: number; rt_shift: number | null; area: number; area_percent: number; height?: number }>;
  }>;
  drift: Array<{ id: string; name: string; channel: string; mean_rt_shift: number | null; max_abs_rt_shift: number | null; matched_peaks: number; total_area: number }>;
}> {
  const { data } = await api.post("/api/compare/hplc", { ids, channel, rt_tolerance: rtTolerance });
  return data;
}

export async function downloadHplcComparisonCsv(ids: string[], channel?: string, rtTolerance = 0.08): Promise<void> {
  const { data } = await api.post("/api/compare/hplc.csv", { ids, channel, rt_tolerance: rtTolerance }, { responseType: "blob", timeout: REPORT_DOWNLOAD_TIMEOUT_MS });
  downloadBlob(data, "hplc-comparison.csv");
}

export async function getBatchQuality(
  ids?: string[],
  limit: number = 1000,
  offset: number = 0,
): Promise<{
  items: Array<{ id: string; technique: string; name: string; status: string; score: number; warnings: string[]; info: string[]; n_peaks: number; manual_confirmed: boolean }>;
  counts: Record<string, number>;
}> {
  const { data } = await api.post("/api/quality/batch", { ids: ids || [] }, {
    params: { limit, offset },
  });
  return data;
}

export interface AIActionRequest {
  name: string;
  args: Record<string, unknown>;
  preview_token?: string;
}

export async function executeAIAction(action: AIActionRequest): Promise<{ action: string; result: Record<string, unknown> }> {
  const { data } = await api.post("/api/ai/actions/execute", action);
  return data;
}

export async function previewAIAction(action: AIActionRequest): Promise<{
  action: string;
  preview: Record<string, unknown>;
  expected_revision?: number;
  preview_token?: string;
}> {
  const { data } = await api.post("/api/ai/actions/preview", action);
  return data;
}

export async function suggestAIActions(ids: string[]): Promise<{
  suggestions: Array<{ title: string; reason: string; action: AIActionRequest; severity: string }>;
}> {
  const { data } = await api.post("/api/ai/actions/suggest", { ids });
  return data;
}

export async function runInference(ids: string[]): Promise<InferenceResponse> {
  const { data } = await api.post("/api/inference", { ids });
  return data;
}

export async function listExamples(): Promise<ExampleSpectrum[]> {
  const { data } = await api.get("/api/spectra/examples");
  return data;
}

export async function loadExample(path: string): Promise<SpectrumListItem[]> {
  const { data } = await api.post("/api/spectra/examples/load", { path });
  return Array.isArray(data) ? data : [data];
}

export async function downloadMarkdownReport(ids: string[], title = "ChemApp Analysis Report"): Promise<void> {
  const { data } = await api.post("/api/reports/markdown", { ids, title, include_peaks: true }, { responseType: "blob", timeout: REPORT_DOWNLOAD_TIMEOUT_MS });
  downloadBlob(data, "chemapp-report.md");
}

export async function downloadHtmlReport(ids: string[], title = "ChemApp Analysis Report"): Promise<void> {
  const { data } = await api.post("/api/reports/html", { ids, title, include_peaks: true }, { responseType: "blob", timeout: REPORT_DOWNLOAD_TIMEOUT_MS });
  downloadBlob(data, "chemapp-report.html");
}

export async function downloadDocxReport(ids: string[], title = "ChemApp Analysis Report"): Promise<void> {
  const { data } = await api.post("/api/reports/docx", { ids, title, include_peaks: true }, { responseType: "blob", timeout: REPORT_DOWNLOAD_TIMEOUT_MS });
  downloadBlob(data, "chemapp-report.docx");
}

export interface ExperimentAgentPayload {
  handout: File;
  ids: string[];
  title?: string;
  name?: string;
  studentId?: string;
  college?: string;
  major?: string;
  teacher?: string;
  location?: string;
  date?: string;
  useLlm?: boolean;
  apiKey?: string;
  baseUrl?: string;
  model?: string;
}

export interface ExperimentAgentReport {
  title: string;
  markdown: string;
  used_llm: boolean;
  steps: Array<{ name: string; status: string; detail: string }>;
  sections: string[];
  questions: string[];
  data_summary: Array<Record<string, unknown>>;
}

function experimentAgentForm(payload: ExperimentAgentPayload): FormData {
  const form = new FormData();
  form.append("handout", payload.handout);
  form.append("ids", JSON.stringify(payload.ids));
  form.append("title", payload.title || "");
  form.append("name", payload.name || "");
  form.append("student_id", payload.studentId || "");
  form.append("college", payload.college || "");
  form.append("major", payload.major || "");
  form.append("teacher", payload.teacher || "");
  form.append("location", payload.location || "");
  form.append("date", payload.date || "");
  form.append("use_llm", payload.useLlm ? "true" : "false");
  form.append("api_key", payload.apiKey || "");
  form.append("base_url", payload.baseUrl || "");
  form.append("model", payload.model || "");
  return form;
}

export async function generateExperimentAgentReport(payload: ExperimentAgentPayload): Promise<ExperimentAgentReport> {
  const { data } = await api.post("/api/experiment-agent/report", experimentAgentForm(payload), { timeout: 600000 });
  return data;
}

export async function downloadExperimentAgentDocx(payload: ExperimentAgentPayload): Promise<void> {
  const { data } = await api.post("/api/experiment-agent/report/docx", experimentAgentForm(payload), {
    responseType: "blob",
    timeout: 600000,
  });
  downloadBlob(data, "experiment-report.docx");
}

export async function downloadExperimentAgentMarkdown(payload: ExperimentAgentPayload): Promise<void> {
  const { data } = await api.post("/api/experiment-agent/report/markdown", experimentAgentForm(payload), {
    responseType: "blob",
    timeout: 600000,
  });
  downloadBlob(data, "experiment-report.md");
}

export async function listStandards(technique?: string, query?: string): Promise<{
  records: Array<{ id: string; name: string; technique: string; formula: string; source: string; tags: string[]; peaks: unknown[]; metadata: Record<string, unknown> }>;
}> {
  const { data } = await api.get("/api/standards", { params: { technique, query } });
  return data;
}

export async function matchStandardsForSpectrum(id: string, tolerance = 0.05): Promise<{ spectrum_id: string; matches: unknown[] }> {
  const { data } = await api.post(`/api/standards/match/${id}`, { tolerance });
  return data;
}

export async function downloadBatchCsvZip(ids: string[]): Promise<void> {
  const { data } = await api.post("/api/spectra/export/csv.zip", { ids }, { responseType: "blob", timeout: REPORT_DOWNLOAD_TIMEOUT_MS });
  downloadBlob(data, "chemapp-export.zip");
}

export async function downloadSpectrumCsv(id: string, filename = "spectrum.csv"): Promise<void> {
  const { data } = await api.get(`/api/spectra/${id}/csv`, { responseType: "blob", timeout: REPORT_DOWNLOAD_TIMEOUT_MS });
  downloadBlob(data, filename);
}

export async function predictStructure(
  peaks13c: { shift: number; intensity?: number; multiplicity?: string }[],
  peaks1h: { shift: number; intensity?: number; multiplicity?: string }[],
  topK: number = 5
): Promise<{ candidates: Array<{ rank: number; smiles: string; similarity: number; source: string }>; status: string; inference_time_ms: number }> {
  const { data } = await api.post("/api/ml/elucidate/predict", {
    peaks_13c: peaks13c,
    peaks_1h: peaks1h,
    top_k: topK,
  });
  return data;
}

export interface ElucidationPeakInput {
  shift: number;
  intensity?: number;
  integral?: number | null;
  multiplicity?: string;
  assignment?: string;
}

export interface ElucidationRequest {
  peaks_13c: ElucidationPeakInput[];
  peaks_1h: ElucidationPeakInput[];
  formula?: string;
  solvent?: string;
  generate_experimental?: boolean;
  candidate_smiles?: string[];
  required_smarts?: string[];
  forbidden_smarts?: string[];
  top_k?: number;
}

export interface ElucidationResponse {
  schema_version?: string;
  pipeline_version?: string;
  candidates?: Array<Record<string, unknown>>;
  experimental_hypotheses?: Array<Record<string, unknown>>;
  generated_candidates?: Array<Record<string, unknown>>;
  mixture_analysis?: Record<string, unknown> | null;
  index?: Record<string, unknown> | null;
  data_attribution?: Array<{
    source: string;
    notice: string;
    license_uri: string;
  }>;
  method?: string;
  inference_time_ms?: number;
  evidence_level?: string;
  uncertainty_kind?: string;
  calibrated_probability?: boolean | {
    probability: number;
    method?: string;
    feature_names?: string[];
    selected_lambda?: number;
    validated_ece?: Record<string, number>;
    semantics?: string;
    automatic_selection_allowed?: boolean;
    external_holder_pending?: boolean;
  };
  top1_calibrated_probability?: number | null;
  decision?: {
    action?: string;
    reason_code?: string;
    selected_candidate_id?: string | null;
    leading_hypothesis_candidate_id?: string | null;
    evidence_state?: string;
    uncertainty_kind?: string;
    calibrated_probability?: boolean;
    top1_probability?: number | null;
  };
  pipeline?: {
    stages?: Array<Record<string, unknown>>;
    reference_context?: Record<string, unknown>;
    forward_context?: Record<string, unknown>;
  };
  result_type?: string;
  candidate_pool_status?: string;
  warnings?: string[];
  query?: {
    formula?: string;
    canonical_formula?: string;
    preprocessing?: Record<string, unknown>;
  };
  forward_model?: {
    schema_version?: string;
    provider?: string;
    mode?: string;
    status?: string;
    reason_code?: string | null;
    nucleus?: string;
    model_called?: boolean;
    used_for_ranking?: boolean;
    diagnostic_only?: boolean;
    evidence_kind?: string;
    calibrated_probability?: boolean;
    quantile_enabled?: boolean;
    evaluated_candidate_count?: number;
    skipped_candidate_count?: number;
    elapsed_ms?: number;
    assignment_limitation?: string;
  };
}

export async function elucidateStructure(
  payload: ElucidationRequest,
  signal?: AbortSignal,
): Promise<ElucidationResponse> {
  const { data } = await api.post("/api/ml/elucidate/predict", payload, {
    signal,
    timeout: 120000,
  });
  return data;
}

export async function elucidateCombined(
  id1: string,
  id2: string,
  formula?: string,
  generateExperimental: boolean = false,
  signal?: AbortSignal,
  constraints?: {
    candidate_smiles?: string[];
    required_smarts?: string[];
    forbidden_smarts?: string[];
  },
): Promise<ElucidationResponse> {
  const { data } = await api.post("/api/ml/elucidate/predict/combined", {
    id1,
    id2,
    formula: formula || undefined,
    generate_experimental: generateExperimental,
    ...constraints,
  }, {
    signal,
    timeout: 120000,
  });
  return data;
}

export async function getMLStatus(signal?: AbortSignal): Promise<Record<string, unknown>> {
  const { data } = await api.get("/api/ml/status", { signal, timeout: 300000 });
  return data;
}
