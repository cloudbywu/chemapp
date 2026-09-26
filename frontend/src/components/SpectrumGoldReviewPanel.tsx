import { useEffect, useId, useMemo, useRef, useState } from "react";
import { useLang } from "../i18n/LangContext";
import {
  adjudicateSpectrumReview,
  ApiError,
  downloadGoldSpectrumManifest,
  enqueueSpectrumReview,
  getSpectrumReview,
  getSpectrumReviewAudit,
  getSpectrumReviewCapabilities,
  listSpectrumReviewQueue,
  submitSpectrumReview,
} from "../services/api";
import type {
  SpectrumData,
  SpectrumReviewCheck,
  SpectrumReviewChecks,
  SpectrumReviewItem,
  SpectrumReviewObservations,
} from "../types/spectrum";
import ConfirmDialog from "./ConfirmDialog";

interface Props {
  spectrum: SpectrumData;
  onDirtyChange?: (dirty: boolean) => void;
}

type PendingAction = "enqueue" | "submit" | "adjudicate" | null;
type Verdict = "accept" | "reject";

interface EnqueueDraft {
  structure_smiles: string;
  structure_source: string;
  molecule_id: string;
  source_collection: string;
  source_record_id: string;
  independence_group: string;
  license_id: string;
  provenance_uri: string;
  rights_confirmed: boolean;
}

interface ReviewDraft {
  verdict: Verdict;
  checks: SpectrumReviewChecks;
  observations: SpectrumReviewObservations;
  notes: string;
}

interface AdjudicationDraft {
  decision: Verdict;
  checks: SpectrumReviewChecks;
  reason: string;
}

interface ReviewCapabilities {
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
}

const uncertainChecks = (): SpectrumReviewChecks => ({
  structure: "uncertain",
  nucleus: "uncertain",
  axis: "uncertain",
  peaks: "uncertain",
  solvent: "uncertain",
});

const initialAdjudicationDraft = (): AdjudicationDraft => ({
  decision: "reject",
  checks: uncertainChecks(),
  reason: "",
});

function metadataValue(spectrum: SpectrumData, ...keys: string[]): string {
  const extra = (
    typeof spectrum.metadata.extra === "object"
    && spectrum.metadata.extra !== null
  )
    ? spectrum.metadata.extra as Record<string, unknown>
    : {};
  for (const key of keys) {
    const value = spectrum.metadata[key] ?? extra[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function initialEnqueueDraft(spectrum: SpectrumData): EnqueueDraft {
  return {
    structure_smiles: metadataValue(spectrum, "smiles", "canonical_smiles"),
    structure_source: "",
    molecule_id: metadataValue(spectrum, "inchi_key", "inchikey"),
    source_collection: "",
    source_record_id: spectrum.source_file || spectrum.id,
    independence_group: "",
    license_id: "",
    provenance_uri: "",
    rights_confirmed: false,
  };
}

function reviewDraftFromItem(item: SpectrumReviewItem): ReviewDraft {
  return {
    verdict: "accept",
    checks: uncertainChecks(),
    observations: {
      structure_smiles: item.structure_smiles,
      nucleus: item.facts.nucleus,
      axis_unit: item.facts.axis_unit,
      axis_direction: item.facts.axis_direction,
      solvent: item.facts.solvent,
      peak_count: item.facts.peak_count,
    },
    notes: "",
  };
}

const checkKeys: Array<keyof SpectrumReviewChecks> = [
  "structure",
  "nucleus",
  "axis",
  "peaks",
  "solvent",
];

export function SpectrumReviewQueuePanel({
  onOpenSpectrum,
}: {
  onOpenSpectrum: (spectrumId: string) => void;
}) {
  const { t } = useLang();
  const filterId = useId();
  const [status, setStatus] = useState("");
  const [items, setItems] = useState<SpectrumReviewItem[]>([]);
  const [reviewerId, setReviewerId] = useState("");
  const [adminId, setAdminId] = useState("");
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    const load = async () => {
      setLoading(true);
      setMessage("");
      try {
        const access = await getSpectrumReviewCapabilities();
        setReviewerId(access.reviewer.subject);
        setAdminId(access.admin.subject);
        if (!access.can_review && !access.can_manage_queue) {
          setItems([]);
          setMessage(t.goldReview.capabilityRequired);
          return;
        }
        const response = await listSpectrumReviewQueue(
          status || undefined,
          controller.signal,
          !access.can_review && access.can_manage_queue,
        );
        setItems(response.items);
        if (response.reviewer_id) setReviewerId(response.reviewer_id);
      } catch (error: unknown) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setItems([]);
        setReviewerId("");
        setAdminId("");
        setMessage(
          error instanceof Error ? error.message : t.goldReview.loadFailed,
        );
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    };
    void load();
    return () => controller.abort();
  }, [
    status,
    t.goldReview.capabilityRequired,
    t.goldReview.loadFailed,
  ]);

  return (
    <section className="gold-review-panel">
      <div className="gold-review-head">
        <div>
          <h3>{t.goldReview.queueTitle}</h3>
          <p>{t.goldReview.queueDescription}</p>
        </div>
        <label className="gold-review-filter" htmlFor={filterId}>
          <span>{t.goldReview.filterStatus}</span>
          <select
            id={filterId}
            value={status}
            onChange={(event) => setStatus(event.target.value)}
          >
            <option value="">{t.goldReview.allStatuses}</option>
            {([
              "pending",
              "awaiting_second",
              "conflict",
              "accepted",
              "rejected",
              "stale",
            ] as const).map((value) => (
              <option key={value} value={value}>
                {t.goldReview.status[value]}
              </option>
            ))}
          </select>
        </label>
      </div>
      {reviewerId && (
        <p className="gold-review-identity">
          {t.goldReview.authenticatedAs}: <strong>{reviewerId}</strong>
        </p>
      )}
      {adminId && (
        <p className="gold-review-identity">
          {t.goldReview.adminAuthenticatedAs}: <strong>{adminId}</strong>
        </p>
      )}
      {loading && <p>{t.action.loading}</p>}
      {!loading && items.length === 0 && !message && (
        <p>{t.goldReview.queueEmpty}</p>
      )}
      <ul className="gold-review-queue-list">
        {items.map((item) => (
          <li key={item.id}>
            <button
              type="button"
              onClick={() => onOpenSpectrum(item.spectrum_id)}
              aria-label={t.goldReview.openSpectrum.replace(
                "{id}",
                item.spectrum_id,
              )}
            >
              <span>
                <strong>{item.facts.source_file || item.spectrum_id}</strong>
                <code>{item.structure_smiles}</code>
              </span>
              <span className={`gold-review-status ${item.status}`}>
                {t.goldReview.status[item.status]} · {item.review_count}/2
              </span>
            </button>
          </li>
        ))}
      </ul>
      {message && <p role="alert">{message}</p>}
    </section>
  );
}

export default function SpectrumGoldReviewPanel({
  spectrum,
  onDirtyChange,
}: Props) {
  const { t } = useLang();
  const baseId = useId();
  const initialEnqueue = useMemo(
    () => initialEnqueueDraft(spectrum),
    [spectrum],
  );
  const [item, setItem] = useState<SpectrumReviewItem | null>(null);
  const [reviewerId, setReviewerId] = useState("");
  const [adminId, setAdminId] = useState("");
  const [capabilities, setCapabilities] = useState<ReviewCapabilities | null>(
    null,
  );
  const [enqueueDraft, setEnqueueDraft] = useState(initialEnqueue);
  const [enqueueBaseline, setEnqueueBaseline] = useState(
    JSON.stringify(initialEnqueue),
  );
  const [reviewDraft, setReviewDraft] = useState<ReviewDraft | null>(null);
  const [reviewBaseline, setReviewBaseline] = useState("");
  const [adjudication, setAdjudication] = useState<AdjudicationDraft>(
    initialAdjudicationDraft,
  );
  const [adjudicationBaseline, setAdjudicationBaseline] = useState(
    () => JSON.stringify(initialAdjudicationDraft()),
  );
  const [auditReviews, setAuditReviews] = useState<Array<{
    id: string;
    reviewer_id: string;
    verdict: Verdict;
    checks: SpectrumReviewChecks;
    notes: string;
  }>>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [pendingAction, setPendingAction] = useState<PendingAction>(null);
  const goldReviewMessagesRef = useRef({
    capabilityRequired: t.goldReview.capabilityRequired,
    loadFailed: t.goldReview.loadFailed,
  });

  useEffect(() => {
    goldReviewMessagesRef.current = {
      capabilityRequired: t.goldReview.capabilityRequired,
      loadFailed: t.goldReview.loadFailed,
    };
  }, [t.goldReview.capabilityRequired, t.goldReview.loadFailed]);

  useEffect(() => {
    const controller = new AbortController();
    const load = async () => {
      setLoading(true);
      setMessage("");
      setItem(null);
      setAuditReviews([]);
      const nextAdjudication = initialAdjudicationDraft();
      setAdjudication(nextAdjudication);
      setAdjudicationBaseline(JSON.stringify(nextAdjudication));
      const nextEnqueue = initialEnqueueDraft(spectrum);
      setEnqueueDraft(nextEnqueue);
      setEnqueueBaseline(JSON.stringify(nextEnqueue));
      try {
        const access = await getSpectrumReviewCapabilities();
        setCapabilities(access);
        setReviewerId(access.reviewer.subject);
        setAdminId(access.admin.subject);
        if (!access.can_review && !access.can_manage_queue) {
          setMessage(goldReviewMessagesRef.current.capabilityRequired);
          return;
        }
        try {
          const response = access.can_review
            ? await getSpectrumReview(spectrum.id, controller.signal)
            : {
                reviewer_id: "",
                item: (await getSpectrumReviewAudit(spectrum.id)).queue,
              };
          if (controller.signal.aborted) return;
          setItem(response.item);
          const nextReview = reviewDraftFromItem(response.item);
          setReviewDraft(nextReview);
          setReviewBaseline(JSON.stringify(nextReview));
          if (
            response.item.status === "conflict"
            && access.can_view_audit
          ) {
            try {
              const audit = await getSpectrumReviewAudit(spectrum.id);
              if (!controller.signal.aborted) {
                setAuditReviews(audit.reviews);
              }
            } catch {
              // The protected audit is optional for a reviewer without admin rights.
            }
          }
        } catch (error) {
          if (!(error instanceof ApiError && error.status === 404)) throw error;
        }
      } catch (error) {
        if (controller.signal.aborted) return;
        setReviewerId("");
        setAdminId("");
        setCapabilities(null);
        setMessage(
          error instanceof Error
            ? error.message
            : goldReviewMessagesRef.current.loadFailed,
        );
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    };
    void load();
    return () => controller.abort();
  }, [spectrum]);

  const enqueueDirty = !item
    && JSON.stringify(enqueueDraft) !== enqueueBaseline;
  const reviewDirty = Boolean(
    item
    && reviewDraft
    && !item.has_submitted
    && ["pending", "awaiting_second"].includes(item.status)
    && JSON.stringify(reviewDraft) !== reviewBaseline,
  );
  const adjudicationDirty = Boolean(
    item?.status === "conflict"
    && JSON.stringify(adjudication) !== adjudicationBaseline,
  );
  const dirty = enqueueDirty || reviewDirty || adjudicationDirty;

  useEffect(() => {
    onDirtyChange?.(dirty);
  }, [dirty, onDirtyChange]);

  useEffect(() => () => onDirtyChange?.(false), [onDirtyChange]);

  const updateEnqueue = (
    field: keyof EnqueueDraft,
    value: string | boolean,
  ) => {
    setEnqueueDraft((previous) => ({ ...previous, [field]: value }));
  };

  const updateCheck = (
    field: keyof SpectrumReviewChecks,
    value: SpectrumReviewCheck,
  ) => {
    setReviewDraft((previous) => previous ? ({
      ...previous,
      checks: { ...previous.checks, [field]: value },
    }) : previous);
  };

  const updateObservation = <K extends keyof SpectrumReviewObservations>(
    field: K,
    value: SpectrumReviewObservations[K],
  ) => {
    setReviewDraft((previous) => previous ? ({
      ...previous,
      observations: { ...previous.observations, [field]: value },
    }) : previous);
  };

  const updateAdjudicationCheck = (
    field: keyof SpectrumReviewChecks,
    value: SpectrumReviewCheck,
  ) => {
    setAdjudication((previous) => ({
      ...previous,
      checks: { ...previous.checks, [field]: value },
    }));
  };

  const canEnqueue = Object.entries(enqueueDraft)
    .filter(([key]) => key !== "rights_confirmed")
    .every(([, value]) => String(value).trim())
    && enqueueDraft.rights_confirmed;
  const canSubmit = Boolean(
    reviewDraft
    && (
      reviewDraft.verdict === "reject"
      || Object.values(reviewDraft.checks).every((value) => value === "pass")
    ),
  );
  const canAdjudicate = adjudication.reason.trim().length >= 3
    && (
      adjudication.decision === "reject"
      || Object.values(adjudication.checks).every(
        (value) => value === "pass",
      )
    );

  const enqueue = async () => {
    setBusy(true);
    setMessage("");
    try {
      const queued = await enqueueSpectrumReview({
        spectrum_id: spectrum.id,
        ...enqueueDraft,
        expected_spectrum_revision: spectrum.spectrum_revision || 1,
        expected_result_revision: spectrum.result_revision || 0,
      });
      setItem(queued);
      const nextReview = reviewDraftFromItem(queued);
      setReviewDraft(nextReview);
      setReviewBaseline(JSON.stringify(nextReview));
      setEnqueueBaseline(JSON.stringify(enqueueDraft));
      setMessage(t.goldReview.enqueued);
    } catch (error) {
      setMessage(
        error instanceof ApiError && error.status === 409
          ? t.goldReview.revisionConflict
          : error instanceof Error ? error.message : t.goldReview.enqueueFailed,
      );
    } finally {
      setBusy(false);
      setPendingAction(null);
    }
  };

  const submit = async () => {
    if (!item || !reviewDraft) return;
    setBusy(true);
    setMessage("");
    try {
      const response = await submitSpectrumReview(spectrum.id, {
        ...reviewDraft,
        expected_queue_revision: item.queue_revision,
        expected_snapshot_sha256: item.snapshot_sha256,
      });
      setReviewerId(response.reviewer_id);
      setItem(response.item);
      setReviewBaseline(JSON.stringify(reviewDraft));
      setMessage(t.goldReview.submitted);
    } catch (error) {
      setMessage(
        error instanceof ApiError && error.status === 409
          ? t.goldReview.revisionConflict
          : error instanceof Error ? error.message : t.goldReview.submitFailed,
      );
    } finally {
      setBusy(false);
      setPendingAction(null);
    }
  };

  const adjudicate = async () => {
    if (!item) return;
    setBusy(true);
    setMessage("");
    try {
      const resolved = await adjudicateSpectrumReview(spectrum.id, {
        ...adjudication,
        expected_queue_revision: item.queue_revision,
        expected_snapshot_sha256: item.snapshot_sha256,
      });
      setItem(resolved);
      setAdjudicationBaseline(JSON.stringify(adjudication));
      setMessage(t.goldReview.adjudicated);
    } catch (error) {
      setMessage(
        error instanceof ApiError && error.status === 409
          ? t.goldReview.revisionConflict
          : error instanceof Error ? error.message : t.goldReview.adjudicateFailed,
      );
    } finally {
      setBusy(false);
      setPendingAction(null);
    }
  };

  const downloadManifest = async () => {
    setMessage("");
    try {
      await downloadGoldSpectrumManifest();
      setMessage(t.goldReview.manifestDownloaded);
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : t.goldReview.manifestFailed,
      );
    }
  };

  const confirm = () => {
    if (pendingAction === "enqueue") void enqueue();
    if (pendingAction === "submit") void submit();
    if (pendingAction === "adjudicate") void adjudicate();
  };

  const dialog = pendingAction === "enqueue"
    ? {
        title: t.goldReview.confirmEnqueueTitle,
        message: t.goldReview.confirmEnqueueMessage,
        label: t.goldReview.enqueue,
        danger: false,
      }
    : pendingAction === "submit"
      ? {
          title: t.goldReview.confirmSubmitTitle,
          message: t.goldReview.confirmSubmitMessage,
          label: t.goldReview.submit,
          danger: false,
        }
      : {
          title: t.goldReview.confirmAdjudicateTitle,
          message: t.goldReview.confirmAdjudicateMessage,
          label: t.goldReview.adjudicate,
          danger: true,
        };

  return (
    <section className="gold-review-panel" aria-labelledby={`${baseId}-title`}>
      <div className="gold-review-head">
        <div>
          <h3 id={`${baseId}-title`}>{t.goldReview.title}</h3>
          <p>{t.goldReview.description}</p>
        </div>
        {capabilities?.can_export_gold && (
          <button
            type="button"
            className="secondary-btn"
            onClick={() => void downloadManifest()}
          >
            {t.goldReview.exportManifest}
          </button>
        )}
      </div>

      {loading && <p>{t.action.loading}</p>}
      {!loading && reviewerId && (
        <p className="gold-review-identity">
          {t.goldReview.authenticatedAs}: <strong>{reviewerId}</strong>
        </p>
      )}
      {!loading && adminId && (
        <p className="gold-review-identity">
          {t.goldReview.adminAuthenticatedAs}: <strong>{adminId}</strong>
        </p>
      )}

      {!loading && !item && capabilities?.can_manage_queue && (
        <fieldset className="gold-review-form">
          <legend>{t.goldReview.enqueueTitle}</legend>
          <p>{t.goldReview.enqueueHint}</p>
          <div className="gold-review-grid">
            {([
              ["structure_smiles", t.goldReview.structureSmiles],
              ["structure_source", t.goldReview.structureSource],
              ["molecule_id", t.goldReview.moleculeId],
              ["source_collection", t.goldReview.sourceCollection],
              ["source_record_id", t.goldReview.sourceRecordId],
              ["independence_group", t.goldReview.independenceGroup],
              ["license_id", t.goldReview.licenseId],
              ["provenance_uri", t.goldReview.provenanceUri],
            ] as Array<[keyof EnqueueDraft, string]>).map(([field, label]) => (
              <label key={field} htmlFor={`${baseId}-${field}`}>
                <span>{label}</span>
                <input
                  id={`${baseId}-${field}`}
                  value={String(enqueueDraft[field])}
                  onChange={(event) => updateEnqueue(field, event.target.value)}
                  required
                />
              </label>
            ))}
          </div>
          <label
            className="gold-review-rights"
            htmlFor={`${baseId}-rights`}
          >
            <input
              id={`${baseId}-rights`}
              type="checkbox"
              checked={enqueueDraft.rights_confirmed}
              onChange={(event) => updateEnqueue(
                "rights_confirmed",
                event.target.checked,
              )}
            />
            <span>{t.goldReview.rightsConfirmed}</span>
          </label>
          <button
            type="button"
            onClick={() => setPendingAction("enqueue")}
            disabled={busy || !canEnqueue}
          >
            {t.goldReview.enqueue}
          </button>
        </fieldset>
      )}

      {item && (
        <>
          <div className="gold-review-status-row">
            <span className={`gold-review-status ${item.status}`}>
              {t.goldReview.status[item.status]}
            </span>
            <span>
              {t.goldReview.cycle} {item.cycle} · {t.goldReview.reviewCount}{" "}
              {item.review_count}/2
            </span>
          </div>
          <dl className="gold-review-facts">
            <div><dt>{t.goldReview.structureSmiles}</dt><dd><code>{item.structure_smiles}</code></dd></div>
            <div><dt>{t.goldReview.nucleus}</dt><dd>{item.facts.nucleus || "—"}</dd></div>
            <div><dt>{t.goldReview.axis}</dt><dd>{item.facts.axis_direction} · {item.facts.axis_unit || "—"}</dd></div>
            <div><dt>{t.goldReview.peaks}</dt><dd>{item.facts.peak_count}</dd></div>
            <div><dt>{t.goldReview.solvent}</dt><dd>{item.facts.solvent || "—"}</dd></div>
            <div><dt>SHA-256</dt><dd><code title={item.snapshot_sha256}>{item.snapshot_sha256.slice(0, 16)}…</code></dd></div>
          </dl>
        </>
      )}

      {item
        && reviewDraft
        && capabilities?.can_review
        && !item.has_submitted
        && ["pending", "awaiting_second"].includes(item.status)
        && (
          <fieldset className="gold-review-form">
            <legend>{t.goldReview.independentReview}</legend>
            <p>{t.goldReview.independenceNotice}</p>
            <div className="gold-review-checks">
              {checkKeys.map((field) => (
                <label key={field} htmlFor={`${baseId}-check-${field}`}>
                  <span>{t.goldReview.checks[field]}</span>
                  <select
                    id={`${baseId}-check-${field}`}
                    value={reviewDraft.checks[field]}
                    onChange={(event) => updateCheck(
                      field,
                      event.target.value as SpectrumReviewCheck,
                    )}
                  >
                    <option value="uncertain">{t.goldReview.checkValues.uncertain}</option>
                    <option value="pass">{t.goldReview.checkValues.pass}</option>
                    <option value="fail">{t.goldReview.checkValues.fail}</option>
                  </select>
                </label>
              ))}
            </div>
            <details className="gold-review-observations">
              <summary>{t.goldReview.observedValues}</summary>
              <p>{t.goldReview.observedValuesHint}</p>
              <div className="gold-review-grid">
                <label htmlFor={`${baseId}-observed-structure`}>
                  <span>{t.goldReview.structureSmiles}</span>
                  <input
                    id={`${baseId}-observed-structure`}
                    value={reviewDraft.observations.structure_smiles}
                    onChange={(event) => updateObservation(
                      "structure_smiles",
                      event.target.value,
                    )}
                  />
                </label>
                <label htmlFor={`${baseId}-observed-nucleus`}>
                  <span>{t.goldReview.nucleus}</span>
                  <input
                    id={`${baseId}-observed-nucleus`}
                    value={reviewDraft.observations.nucleus}
                    onChange={(event) => updateObservation(
                      "nucleus",
                      event.target.value,
                    )}
                  />
                </label>
                <label htmlFor={`${baseId}-observed-axis-unit`}>
                  <span>{t.goldReview.axisUnit}</span>
                  <input
                    id={`${baseId}-observed-axis-unit`}
                    value={reviewDraft.observations.axis_unit}
                    onChange={(event) => updateObservation(
                      "axis_unit",
                      event.target.value,
                    )}
                  />
                </label>
                <label htmlFor={`${baseId}-observed-axis-direction`}>
                  <span>{t.goldReview.axisDirection}</span>
                  <select
                    id={`${baseId}-observed-axis-direction`}
                    value={reviewDraft.observations.axis_direction}
                    onChange={(event) => updateObservation(
                      "axis_direction",
                      event.target.value as SpectrumReviewObservations[
                        "axis_direction"
                      ],
                    )}
                  >
                    {([
                      "ascending",
                      "descending",
                      "non_monotonic",
                      "unknown",
                    ] as const).map((value) => (
                      <option key={value} value={value}>
                        {t.goldReview.axisDirections[value]}
                      </option>
                    ))}
                  </select>
                </label>
                <label htmlFor={`${baseId}-observed-peak-count`}>
                  <span>{t.goldReview.peaks}</span>
                  <input
                    id={`${baseId}-observed-peak-count`}
                    type="number"
                    min={0}
                    max={100000}
                    value={reviewDraft.observations.peak_count}
                    onChange={(event) => updateObservation(
                      "peak_count",
                      Number(event.target.value) || 0,
                    )}
                  />
                </label>
                <label htmlFor={`${baseId}-observed-solvent`}>
                  <span>{t.goldReview.solvent}</span>
                  <input
                    id={`${baseId}-observed-solvent`}
                    value={reviewDraft.observations.solvent}
                    onChange={(event) => updateObservation(
                      "solvent",
                      event.target.value,
                    )}
                  />
                </label>
              </div>
            </details>
            <label htmlFor={`${baseId}-verdict`}>
              <span>{t.goldReview.verdict}</span>
              <select
                id={`${baseId}-verdict`}
                value={reviewDraft.verdict}
                onChange={(event) => setReviewDraft((previous) => previous ? ({
                  ...previous,
                  verdict: event.target.value as Verdict,
                }) : previous)}
              >
                <option value="accept">{t.goldReview.accept}</option>
                <option value="reject">{t.goldReview.reject}</option>
              </select>
            </label>
            <label htmlFor={`${baseId}-review-notes`}>
              <span>{t.goldReview.notes}</span>
              <textarea
                id={`${baseId}-review-notes`}
                value={reviewDraft.notes}
                onChange={(event) => setReviewDraft((previous) => previous ? ({
                  ...previous,
                  notes: event.target.value,
                }) : previous)}
                maxLength={4000}
              />
            </label>
            <button
              type="button"
              onClick={() => setPendingAction("submit")}
              disabled={busy || !canSubmit}
            >
              {t.goldReview.submit}
            </button>
          </fieldset>
        )}

      {item?.has_submitted && item.status === "awaiting_second" && (
        <p className="gold-review-waiting">{t.goldReview.waitingSecond}</p>
      )}

      {item?.status === "conflict" && capabilities?.can_adjudicate && (
        <fieldset className="gold-review-form gold-review-conflict">
          <legend>{t.goldReview.conflictTitle}</legend>
          <p>{t.goldReview.conflictHint}</p>
          {auditReviews.length > 0 && (
            <ul className="gold-review-audit-list">
              {auditReviews.map((review) => (
                <li key={review.id}>
                  <strong>{review.reviewer_id}</strong>:{" "}
                  {review.verdict === "accept"
                    ? t.goldReview.accept
                    : t.goldReview.reject}
                  {review.notes ? ` — ${review.notes}` : ""}
                </li>
              ))}
            </ul>
          )}
          <div className="gold-review-checks">
            {checkKeys.map((field) => (
              <label key={field} htmlFor={`${baseId}-adjudicate-${field}`}>
                <span>{t.goldReview.checks[field]}</span>
                <select
                  id={`${baseId}-adjudicate-${field}`}
                  value={adjudication.checks[field]}
                  onChange={(event) => updateAdjudicationCheck(
                    field,
                    event.target.value as SpectrumReviewCheck,
                  )}
                >
                  <option value="uncertain">{t.goldReview.checkValues.uncertain}</option>
                  <option value="pass">{t.goldReview.checkValues.pass}</option>
                  <option value="fail">{t.goldReview.checkValues.fail}</option>
                </select>
              </label>
            ))}
          </div>
          <label htmlFor={`${baseId}-adjudication-decision`}>
            <span>{t.goldReview.decision}</span>
            <select
              id={`${baseId}-adjudication-decision`}
              value={adjudication.decision}
              onChange={(event) => setAdjudication((previous) => ({
                ...previous,
                decision: event.target.value as Verdict,
              }))}
            >
              <option value="accept">{t.goldReview.accept}</option>
              <option value="reject">{t.goldReview.reject}</option>
            </select>
          </label>
          <label htmlFor={`${baseId}-adjudication-reason`}>
            <span>{t.goldReview.reason}</span>
            <textarea
              id={`${baseId}-adjudication-reason`}
              value={adjudication.reason}
              onChange={(event) => setAdjudication((previous) => ({
                ...previous,
                reason: event.target.value,
              }))}
              required
              maxLength={4000}
            />
          </label>
          <button
            type="button"
            className="danger-btn"
            onClick={() => setPendingAction("adjudicate")}
            disabled={busy || !canAdjudicate}
          >
            {t.goldReview.adjudicate}
          </button>
        </fieldset>
      )}

      {item?.status === "stale" && (
        <p className="gold-review-warning">{t.goldReview.staleHint}</p>
      )}
      {item?.status === "accepted" && (
        <p className="gold-review-success">{t.goldReview.goldReady}</p>
      )}
      {item?.status === "rejected" && (
        <p className="gold-review-warning">{t.goldReview.rejectedHint}</p>
      )}
      {dirty && <p className="dirty-indicator">{t.manual.unsaved}</p>}
      {message && <p role="status" aria-live="polite">{message}</p>}

      <ConfirmDialog
        open={pendingAction !== null}
        title={dialog.title}
        message={dialog.message}
        confirmLabel={dialog.label}
        cancelLabel={t.action.cancel}
        busyLabel={t.goldReview.saving}
        busy={busy}
        danger={dialog.danger}
        onConfirm={confirm}
        onCancel={() => setPendingAction(null)}
      />
    </section>
  );
}
