import { useEffect, useId, useLayoutEffect, useRef } from "react";
import { createPortal } from "react-dom";

interface OpenDialog {
  element: HTMLDivElement;
  returnFocus: HTMLElement | null;
}

// Only the topmost modal owns keyboard input. Retain the original inert state
// so closing a dialog cannot accidentally unlock an already protected surface.
const openDialogs: OpenDialog[] = [];
const inertTargets = new Map<HTMLElement, boolean>();
const FOCUSABLE_SELECTOR = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

function synchronizeModals() {
  if (openDialogs.length > 0 && inertTargets.size === 0) {
    document.querySelectorAll<HTMLElement>(".app-header, .app-main").forEach((element) => {
      inertTargets.set(element, element.hasAttribute("inert"));
      element.setAttribute("inert", "");
    });
  } else if (openDialogs.length === 0) {
    inertTargets.forEach((wasInert, element) => {
      if (!wasInert) element.removeAttribute("inert");
    });
    inertTargets.clear();
  }
  openDialogs.forEach(({ element }, index) => {
    element.toggleAttribute("inert", index !== openDialogs.length - 1);
  });
}

interface Props {
  open: boolean;
  title: string;
  message: string;
  confirmLabel: string;
  cancelLabel: string;
  busyLabel?: string;
  busy?: boolean;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel,
  cancelLabel,
  busyLabel,
  busy = false,
  danger = false,
  onConfirm,
  onCancel,
}: Props) {
  const cancelRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const messageId = useId();

  const latest = useRef({ busy, onCancel });
  useLayoutEffect(() => {
    latest.current = { busy, onCancel };
    if (open && busy && dialogRef.current?.contains(document.activeElement)) {
      dialogRef.current.focus();
    }
  }, [busy, onCancel, open]);

  useEffect(() => {
    if (!open || !dialogRef.current) return;
    const element = dialogRef.current;
    const entry: OpenDialog = {
      element,
      returnFocus: document.activeElement instanceof HTMLElement ? document.activeElement : null,
    };
    openDialogs.push(entry);
    synchronizeModals();
    if (latest.current.busy) element.focus();
    else cancelRef.current?.focus();
    const isTopmost = () => openDialogs.at(-1) === entry;

    const onKeyDown = (event: KeyboardEvent) => {
      if (!isTopmost()) return;
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        if (!latest.current.busy) latest.current.onCancel();
      }
      if (event.key !== "Tab") return;
      const focusable = Array.from(element.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
      const first = focusable[0];
      const last = focusable.at(-1);
      if (!first || !last) {
        event.preventDefault();
        element.focus();
      } else if (!element.contains(document.activeElement) || document.activeElement === element) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      const wasTopmost = isTopmost();
      const index = openDialogs.indexOf(entry);
      // If a lower dialog is removed first, pass its opener to the dialog
      // above it rather than restoring focus to a now-detached button later.
      openDialogs.forEach((other) => {
        if (other !== entry && other.returnFocus && element.contains(other.returnFocus)) {
          other.returnFocus = entry.returnFocus;
        }
      });
      if (index !== -1) openDialogs.splice(index, 1);
      synchronizeModals();
      if (!wasTopmost) return;
      const target = entry.returnFocus;
      if (target?.isConnected && !target.closest("[inert]")) target.focus();
      else {
        const remaining = openDialogs.at(-1)?.element;
        (remaining?.querySelector<HTMLElement>(FOCUSABLE_SELECTOR) ?? remaining)?.focus();
      }
    };
  }, [open]);

  if (!open) return null;

  return createPortal((
    <div className="confirm-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget && !busy) onCancel();
    }}>
      <div
        ref={dialogRef}
        className="confirm-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={messageId}
        tabIndex={-1}
      >
        <h2 id={titleId}>{title}</h2>
        <p id={messageId}>{message}</p>
        <div className="confirm-actions">
          <button ref={cancelRef} type="button" className="secondary-btn" onClick={onCancel} disabled={busy}>
            {cancelLabel}
          </button>
          <button type="button" className={danger ? "danger-btn" : ""} onClick={onConfirm} disabled={busy}>
            {busy ? (busyLabel || confirmLabel) : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  ), document.body);
}
