import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LangProvider } from "../i18n/LangContext";
import SpectrumList from "./SpectrumList";

const items = [
  {
    id: "s1",
    technique: "NMR",
    points: 1024,
    name: "Sample A",
    has_result: true,
    summary: "",
  },
  {
    id: "s2",
    technique: "HPLC",
    points: 900,
    name: "Sample B",
    has_result: false,
    summary: "",
  },
];

describe("SpectrumList", () => {
  beforeEach(() => {
    localStorage.setItem("chemapp-lang", "en");
  });

  it("renders items and fires select", () => {
    const onSelect = vi.fn();
    const onDelete = vi.fn();
    render(
      <LangProvider>
        <SpectrumList spectra={items} selectedId="s1" onSelect={onSelect} onDelete={onDelete} />
      </LangProvider>,
    );
    expect(screen.getByText("Sample A")).toBeInTheDocument();
    expect(screen.getByText("Sample B")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Select spectrum Sample B" }));
    expect(onSelect).toHaveBeenCalledWith("s2");
  });

  it("fires delete", () => {
    const onDelete = vi.fn();
    render(
      <LangProvider>
        <SpectrumList spectra={items} selectedId={null} onSelect={vi.fn()} onDelete={onDelete} />
      </LangProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Delete spectrum Sample A" }));
    expect(onDelete).toHaveBeenCalledWith("s1");
  });

  it("shows empty hint", () => {
    render(
      <LangProvider>
        <SpectrumList spectra={[]} selectedId={null} onSelect={vi.fn()} onDelete={vi.fn()} />
      </LangProvider>,
    );
    expect(
      screen.getByText("No spectra loaded yet. Upload a file to get started."),
    ).toBeInTheDocument();
  });

  it("keeps replacement-pattern characters in spectrum names literal", () => {
    render(
      <LangProvider>
        <SpectrumList
          spectra={[{
            id: "s9",
            technique: "NMR",
            points: 4,
            name: "Batch $& run",
            has_result: false,
            summary: "",
          }]}
          selectedId={null}
          onSelect={vi.fn()}
          onDelete={vi.fn()}
        />
      </LangProvider>,
    );
    expect(screen.getByRole("button", { name: "Select spectrum Batch $& run" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete spectrum Batch $& run" })).toBeInTheDocument();
  });
});
