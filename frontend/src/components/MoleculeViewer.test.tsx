import { render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { applyMock } = vi.hoisted(() => ({ applyMock: vi.fn() }));

vi.mock("smiles-drawer", () => ({
  default: { apply: applyMock },
}));

import MoleculeViewer from "./MoleculeViewer";

describe("MoleculeViewer", () => {
  beforeEach(() => {
    applyMock.mockClear();
  });

  it("assigns unique, selector-safe canvas ids", () => {
    const { container } = render(
      <>
        <MoleculeViewer smiles="CC" />
        <MoleculeViewer smiles="CO" />
      </>,
    );
    const canvases = [...container.querySelectorAll("canvas")];
    expect(canvases).toHaveLength(2);
    const ids = canvases.map((canvas) => canvas.id);
    expect(new Set(ids).size).toBe(2);
    for (const id of ids) {
      expect(id).toMatch(/^[a-zA-Z0-9_-]+$/);
    }
  });

  it("draws through a CSS selector that resolves to its own canvas", async () => {
    const { container } = render(<MoleculeViewer smiles="CC" />);
    await waitFor(() => expect(applyMock).toHaveBeenCalledTimes(1));
    const selector = applyMock.mock.calls[0][1] as string;
    expect(container.querySelector(selector)).toBe(container.querySelector("canvas"));
  });

  it("ignores the async draw result after unmount", async () => {
    const view = render(<MoleculeViewer smiles="CC" />);
    view.unmount();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(applyMock).not.toHaveBeenCalled();
  });

  it("renders nothing without a SMILES string", () => {
    const { container } = render(<MoleculeViewer smiles="" />);
    expect(container.firstChild).toBeNull();
  });
});
