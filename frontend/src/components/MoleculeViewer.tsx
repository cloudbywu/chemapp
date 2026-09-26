import { useEffect, useId, useRef } from "react";

interface Props {
  smiles: string;
  width?: number;
  height?: number;
}

export default function MoleculeViewer({ smiles, width = 120, height = 100 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  // useId is StrictMode-safe (no render-time module counter); sanitised
  // because the value is also reused as a CSS selector for smiles-drawer.
  const canvasId = `mol-canvas-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;

  useEffect(() => {
    if (!containerRef.current || !smiles) {
      if (containerRef.current) containerRef.current.innerHTML = "";
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.id = canvasId;
    canvas.setAttribute("data-smiles", smiles);
    canvas.width = width * 2;
    canvas.height = height * 2;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    containerRef.current.innerHTML = "";
    containerRef.current.appendChild(canvas);

    let cancelled = false;
    import("smiles-drawer").then((SmilesDrawer) => {
      if (cancelled) return;
      SmilesDrawer.default.apply(
        { width: width * 2, height: height * 2, bondThickness: 1.5, bondLength: 18, shortBondLength: 14, fontSize: 12, terminalCarbons: false, compactDrawing: true },
        `#${canvasId}`,
        "dark",
      );
    }).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [smiles, width, height, canvasId]);

  if (!smiles) return null;
  return <div ref={containerRef} style={{ width, height, flexShrink: 0 }} />;
}
