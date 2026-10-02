import { useEffect, useId, useRef, useState } from "react";
import { useLang } from "../i18n/LangContext";
import {
  cancelNmr2StructDownload,
  getNmr2StructWeights,
  startNmr2StructDownload,
  type ModelDownloadJob,
  type Nmr2StructAsset,
  type Nmr2StructAssetId,
  type Nmr2StructWeights,
} from "../services/api";

const OFFICIAL_SOURCE = "https://github.com/MarklandGroup/NMR2Struct";
const POLL_INTERVAL_MS = 1000;
const activeJob = (job: ModelDownloadJob | null) => job != null
  && ["queued", "downloading", "verifying", "cancelling"].includes(job.status);

function bytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / (1024 ** index)).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export default function ModelAssetsPanel() {
  const { t } = useLang();
  const copy = t.modelAssets;
  const headingId = useId();
  const [inventory, setInventory] = useState<Nmr2StructWeights | null>(null);
  const [error, setError] = useState("");
  const [statusError, setStatusError] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [pending, setPending] = useState<{ assetId: Nmr2StructAssetId; kind: "start" | "cancel" } | null>(null);
  const mutation = useRef<AbortController | null>(null);
  const read = useRef<AbortController | null>(null);

  useEffect(() => () => mutation.current?.abort(), []);

  // Each effect owns one request chain. Never overlap polls or accept a response
  // from before a start/cancel, navigation, manual refresh, or language change.
  useEffect(() => {
    if (pending) return;
    const controller = new AbortController();
    read.current = controller;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let pollingActive = false;
    let failures = 0;
    const load = async () => {
      try {
        const data = await getNmr2StructWeights(controller.signal);
        if (controller.signal.aborted) return;
        pollingActive = data.assets.some((asset) => activeJob(asset.job));
        setInventory(data);
        setConfirmed(true);
        setStatusError("");
        failures = 0;
      } catch (cause: unknown) {
        if (controller.signal.aborted) return;
        setStatusError(cause instanceof Error ? cause.message : copy.loadFailed);
        setConfirmed(false);
        failures += 1;
        // A network error does not mean the server stopped downloading.
        pollingActive = true;
      }
      if (!controller.signal.aborted && pollingActive) timer = setTimeout(() => void load(), Math.min(15000, POLL_INTERVAL_MS * (2 ** failures)));
    };
    void load();
    return () => {
      controller.abort();
      if (timer !== undefined) clearTimeout(timer);
      if (read.current === controller) read.current = null;
    };
  }, [refresh, pending, copy.loadFailed]);

  const updateJob = (job: ModelDownloadJob) => {
    setInventory((current) => current ? {
      ...current,
      assets: current.assets.map((asset) => asset.id === job.asset_id ? { ...asset, job } : asset),
    } : current);
  };

  const perform = async (asset: Nmr2StructAsset, kind: "start" | "cancel") => {
    if (mutation.current || (kind === "start" && (
      asset.status === "ready" || inventory?.assets.some((item) => activeJob(item.job))
    ))) return;
    if (kind === "cancel" && (!asset.job || !activeJob(asset.job) || asset.job.status === "cancelling")) return;
    const controller = new AbortController();
    mutation.current = controller;
    read.current?.abort();
    setPending({ assetId: asset.id, kind });
    setError("");
    setConfirmed(false);
    try {
      const job = kind === "start"
        ? await startNmr2StructDownload(asset.id, controller.signal)
        : await cancelNmr2StructDownload(asset.job!.id, controller.signal);
      if (!controller.signal.aborted) updateJob(job);
    } catch (cause: unknown) {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : copy.actionFailed);
    } finally {
      if (!controller.signal.aborted && mutation.current === controller) {
        mutation.current = null;
        setPending(null);
        // Reconcile ambiguous network failures after POST as well as success.
        setRefresh((value) => value + 1);
      }
    }
  };

  const names: Record<Nmr2StructAssetId, string> = {
    cnmr_only: copy.carbon, hnmr_only: copy.proton, multitask: copy.combined,
  };
  const downloading = inventory?.assets.some((asset) => activeJob(asset.job)) ?? false;

  return (
    <section className="settings-panel model-assets-panel" aria-labelledby={headingId}>
      <div className="model-assets-heading">
        <h3 id={headingId}>{copy.title}</h3>
        <button type="button" className="secondary-btn" disabled={!!pending} onClick={() => {
          read.current?.abort();
          setConfirmed(false);
          setError("");
          setRefresh((value) => value + 1);
        }}>{copy.refresh}</button>
      </div>
      <p className="settings-desc">{copy.description}</p>
      <p className="settings-hint">{copy.source}: <a href={OFFICIAL_SOURCE} target="_blank" rel="noreferrer">MarklandGroup/NMR2Struct</a></p>
      <p className="settings-hint">{copy.caution}</p>
      {inventory && (
        <>
          <dl className="model-assets-location">
            <div><dt>{copy.revision}</dt><dd>{inventory.revision}</dd></div>
            <div><dt>{copy.storage}</dt><dd>{inventory.storage_dir}</dd></div>
          </dl>
          <div className="model-assets-grid">
            {inventory.assets.map((asset) => {
              const job = asset.job;
              const active = activeJob(job);
              const status = active && job ? job.status : asset.status === "ready" ? "ready"
                : job && ["error", "cancelled"].includes(job.status) ? job.status : asset.status;
              const name = names[asset.id];
              const retry = asset.status === "invalid" || job?.status === "error" || job?.status === "cancelled";
              const starting = pending?.assetId === asset.id && pending.kind === "start";
              const cancelling = pending?.assetId === asset.id && pending.kind === "cancel";
              const progress = job && job.total_bytes > 0 ? Math.max(0, Math.min(100, job.downloaded_bytes / job.total_bytes * 100)) : undefined;
              return (
                <article key={asset.id} className="model-asset-card" aria-label={name}>
                  <div className="model-asset-title"><h4>{name}</h4><span className={`model-asset-status model-asset-status-${status}`} role="status">{copy[status]}</span></div>
                  <p className="model-asset-filename">{asset.filename}</p>
                  <p>{copy.size}: <strong>{bytes(asset.size_bytes)}</strong></p>
                  <details className="model-asset-checksum"><summary>{copy.checksum}</summary><p>{asset.sha256}</p><p>{copy.checksumHint}</p></details>
                  {active && job && (
                    <div className="model-asset-progress">
                      <progress aria-label={`${name} ${copy.downloading}`} max={100} value={progress} />
                      <span>{copy.progress.replace("{downloaded}", bytes(job.downloaded_bytes)).replace("{total}", bytes(job.total_bytes))}</span>
                    </div>
                  )}
                  {job?.status === "error" && job.error && <p className="error" role="alert">{job.error}</p>}
                  {asset.status === "ready" && <p className="settings-hint">{copy.installed}</p>}
                  {active ? (
                    <button type="button" className="secondary-btn" disabled={!!pending || job?.status === "cancelling"} onClick={() => void perform(asset, "cancel")}>
                      {cancelling || job?.status === "cancelling" ? copy.cancelling : copy.cancel.replace("{name}", name)}
                    </button>
                  ) : asset.status !== "ready" && (
                    <button type="button" disabled={!!pending || downloading || !confirmed} onClick={() => void perform(asset, "start")}>
                      {starting ? copy.starting : (retry ? copy.retry : copy.download).replace("{name}", name)}
                    </button>
                  )}
                </article>
              );
            })}
          </div>
          <p className="settings-hint">{copy.limitHint}</p>
        </>
      )}
      {!inventory && !statusError && <p role="status">{copy.loading}</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {statusError && <p className="error" role="alert">{statusError}{inventory && <> {copy.stale}</>}</p>}
      <p className="settings-hint">{copy.adminHint}</p>
    </section>
  );
}
