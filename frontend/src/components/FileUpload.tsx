import { useEffect, useRef, useState, type DragEvent, type ChangeEvent } from "react";
import { listExamples, loadExample, uploadFile } from "../services/api";
import { useLang } from "../i18n/LangContext";
import type { ExampleSpectrum, SpectrumListItem } from "../types/spectrum";

interface Props {
  onUploaded: (item: SpectrumListItem) => void;
}

export default function FileUpload({ onUploaded }: Props) {
  const { t } = useLang();
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [examples, setExamples] = useState<ExampleSpectrum[]>([]);
  const [loadingExample, setLoadingExample] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listExamples().then(setExamples).catch(() => setExamples([]));
  }, []);

  const uploadMultiple = async (files: FileList | File[]) => {
    setError("");
    setUploading(true);
    const total = files.length;
    let succeeded = 0;
    let firstError = "";

    for (let i = 0; i < total; i++) {
      const file = files[i];
      // Function-form replacements: file names may contain "$&" and friends,
      // which String.replace would otherwise expand in the replacement text.
      setUploadProgress(t.upload.progress
        .replace("{current}", () => String(i + 1))
        .replace("{total}", () => String(total))
        .replace("{name}", () => file.name));
      try {
        const results = await uploadFile(file);
        for (const r of results) {
          onUploaded(r);
        }
        succeeded++;
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Upload failed";
        const fileErr = `${file.name}: ${msg}`;
        if (!firstError) firstError = fileErr;
      }
    }

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
    if (files.length > 0) uploadMultiple(files);
  };

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) uploadMultiple(files);
  };

  const handleLoadExample = async (path: string) => {
    setError("");
    setLoadingExample(path);
    try {
      const results = await loadExample(path);
      for (const item of results) onUploaded(item);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Example load failed");
    } finally {
      setLoadingExample("");
    }
  };

  return (
    <div className="file-upload">
      <button
        type="button"
        className={`dropzone ${dragOver ? "drag-over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        aria-label={t.upload.chooseAria}
        disabled={uploading}
      >
        {uploading ? (
          <>
            <p>{t.upload.uploading}</p>
            <p className="progress-text">{uploadProgress}</p>
          </>
        ) : (
          <>
            <p>{t.upload.drop}</p>
            <p className="hint">{t.upload.hint}</p>
          </>
        )}
      </button>
      <input
        ref={inputRef}
        id="instrument-file-upload"
        type="file"
        accept=".txt,.csv,.tsv,.dx,.jdx,.jcamp,.zip,.jdf,.asc,.ras"
        multiple
        onChange={onChange}
        hidden
      />
      <label className="visually-hidden" htmlFor="instrument-file-upload">
        {t.upload.chooseAria}
      </label>
      <div className="visually-hidden" role="status" aria-live="polite">
        {uploading ? uploadProgress || t.upload.uploading : ""}
      </div>
      {examples.length > 0 && (
        <div className="example-loader">
          <div className="example-title">{t.upload.examples}</div>
          <div className="example-grid">
            {examples.map((ex) => (
              <button
                key={ex.path}
                type="button"
                onClick={() => handleLoadExample(ex.path)}
                disabled={!!loadingExample}
                title={ex.filename}
              >
                <span>{ex.technique}</span>
                {loadingExample === ex.path ? t.upload.loadingExample : ex.label}
              </button>
            ))}
          </div>
        </div>
      )}
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}
