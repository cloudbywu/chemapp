import { useEffect, useImperativeHandle, useLayoutEffect, useRef, useState, type DragEvent, type ChangeEvent, type Ref } from "react";
import { listExamples, loadExample, uploadFile } from "../services/api";
import { useLang } from "../i18n/LangContext";
import type { ExampleSpectrum, SpectrumListItem } from "../types/spectrum";

export interface FileUploadHandle {
  chooseFiles: () => void;
  showExamples: () => void;
}

interface Props {
  ref?: Ref<FileUploadHandle>;
  onUploaded: (item: SpectrumListItem) => void;
}

export default function FileUpload({ onUploaded, ref }: Props) {
  const { t } = useLang();
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [examples, setExamples] = useState<ExampleSpectrum[]>([]);
  const [loadingExample, setLoadingExample] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const examplesRef = useRef<HTMLDetailsElement>(null);
  const chooseRef = useRef<HTMLButtonElement>(null);
  const onUploadedRef = useRef(onUploaded);
  const activeOperationRef = useRef<object | null>(null);
  const busy = uploading || !!loadingExample;

  useImperativeHandle(ref, () => ({
    chooseFiles: () => {
      if (!activeOperationRef.current) inputRef.current?.click();
    },
    showExamples: () => {
      const section = examplesRef.current;
      if (!section) {
        chooseRef.current?.focus();
        return;
      }
      section.open = true;
      const target = section.querySelector<HTMLElement>("button:not([disabled])")
        ?? section.querySelector<HTMLElement>("summary");
      target?.focus();
      target?.scrollIntoView?.({ block: "nearest" });
    },
  }), []);

  useLayoutEffect(() => {
    // Completion must use the current navigation/unsaved-edit guard, rather
    // than the callback captured before the upload or example request began.
    onUploadedRef.current = onUploaded;
  }, [onUploaded]);

  useEffect(() => {
    let active = true;
    listExamples().then((items) => {
      if (active) setExamples(items);
    }).catch(() => {
      if (active) setExamples([]);
    });
    return () => {
      active = false;
      activeOperationRef.current = null;
    };
  }, []);

  const uploadMultiple = async (files: FileList | File[]) => {
    if (activeOperationRef.current || files.length === 0) return;
    const operation = {};
    activeOperationRef.current = operation;
    // DataTransfer/FileList contents can change once their event has ended.
    const batch = Array.from(files);
    setError("");
    setUploading(true);
    const total = batch.length;
    let succeeded = 0;
    let firstError = "";

    for (let i = 0; i < total; i++) {
      const file = batch[i];
      // Function-form replacements: file names may contain "$&" and friends,
      // which String.replace would otherwise expand in the replacement text.
      setUploadProgress(t.upload.progress
        .replace("{current}", () => String(i + 1))
        .replace("{total}", () => String(total))
        .replace("{name}", () => file.name));
      try {
        const results = await uploadFile(file);
        if (activeOperationRef.current !== operation) return;
        for (const r of results) {
          onUploadedRef.current(r);
        }
        succeeded++;
      } catch (e: unknown) {
        if (activeOperationRef.current !== operation) return;
        const msg = e instanceof Error ? e.message : "Upload failed";
        const fileErr = `${file.name}: ${msg}`;
        if (!firstError) firstError = fileErr;
      }
    }

    activeOperationRef.current = null;
    setUploading(false);
    setUploadProgress("");
    if (inputRef.current) inputRef.current.value = "";
    if (succeeded === 0 && firstError) {
      setError(firstError);
    } else if (succeeded < total) {
      setError(t.upload.partial
        .replace("{succeeded}", () => String(succeeded))
        .replace("{total}", () => String(total))
        .replace("{error}", () => firstError));
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (activeOperationRef.current) return;
    const items = e.dataTransfer.items;
    if (items) {
      let hasDir = false;
      for (let i = 0; i < items.length; i++) {
        const entry = (items[i] as unknown as { webkitGetAsEntry?: () => FileSystemEntry | null }).webkitGetAsEntry?.();
        if (entry?.isDirectory) {
          hasDir = true;
          break;
        }
      }
      if (hasDir) {
        setError(t.upload.dirError);
        return;
      }
    }
    const files = e.dataTransfer.files;
    if (files.length > 0) void uploadMultiple(files);
  };

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) void uploadMultiple(files);
  };

  const handleLoadExample = async (path: string) => {
    if (activeOperationRef.current) return;
    const operation = {};
    activeOperationRef.current = operation;
    setError("");
    setLoadingExample(path);
    try {
      const results = await loadExample(path);
      if (activeOperationRef.current !== operation) return;
      for (const item of results) onUploadedRef.current(item);
    } catch (e: unknown) {
      if (activeOperationRef.current !== operation) return;
      setError(e instanceof Error ? e.message : "Example load failed");
    } finally {
      if (activeOperationRef.current === operation) {
        activeOperationRef.current = null;
        setLoadingExample("");
      }
    }
  };

  return (
    <div className="file-upload">
      <button
        ref={chooseRef}
        type="button"
        className={`dropzone ${dragOver ? "drag-over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); if (!activeOperationRef.current) setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        aria-label={t.upload.chooseAria}
        aria-busy={busy}
        disabled={busy}
      >
        {uploading ? (
          <>
            <p>{t.upload.uploading}</p>
            <p className="progress-text">{uploadProgress}</p>
          </>
        ) : (
          <>
            <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
              <path d="M12 16V4m-4 4 4-4 4 4M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" />
            </svg>
            <p>{t.workspace.import}</p>
            <p className="dropzone-description">{t.upload.multiple}</p>
          </>
        )}
      </button>
      <details className="upload-formats">
        <summary>{t.upload.formats}</summary>
        <p>{t.upload.hint}</p>
      </details>
      <input
        ref={inputRef}
        id="instrument-file-upload"
        type="file"
        accept=".txt,.csv,.tsv,.dx,.jdx,.jcamp,.zip,.jdf,.asc,.ras"
        multiple
        disabled={busy}
        onChange={onChange}
        hidden
      />
      <label className="visually-hidden" htmlFor="instrument-file-upload">
        {t.upload.chooseAria}
      </label>
      <div className="visually-hidden" role="status" aria-live="polite">
        {uploading ? uploadProgress || t.upload.uploading : loadingExample ? t.upload.loadingExample : ""}
      </div>
      {examples.length > 0 && (
        <details ref={examplesRef} className="example-loader" open>
          <summary className="example-title">{t.upload.examples}<span className="section-count">{examples.length}</span></summary>
          <div className="example-grid">
            {examples.map((ex) => (
              <button
                key={ex.path}
                type="button"
                onClick={() => void handleLoadExample(ex.path)}
                disabled={busy}
                title={ex.filename}
              >
                <span className="example-technique">{ex.technique}</span>
                <span className="example-name">{loadingExample === ex.path ? t.upload.loadingExample : ex.label}</span>
                <span className="example-arrow" aria-hidden="true">↗</span>
              </button>
            ))}
          </div>
        </details>
      )}
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}
