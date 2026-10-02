import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LangProvider } from "../i18n/LangContext";
import { compareSpectra } from "../services/api";
import type { ComparisonResult, SpectrumListItem } from "../types/spectrum";
import CompareView from "./CompareView";

vi.mock("../services/api", () => ({ compareSpectra: vi.fn() }));

const spectra: SpectrumListItem[] = ["a", "b", "c"].map((id) => ({
  id, name: id, technique: "Fluorescence", points: 10, has_result: true,
  summary: "", spectrum_revision: 1, result_revision: 1,
}));

const comparison = (id1 = "a", id2 = "b", points1 = 10): ComparisonResult => ({
  id1, id2, technique1: "Fluorescence", technique2: "Fluorescence",
  shared: { points1, points2: 20 },
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

const view = (items = spectra) => <LangProvider><CompareView spectra={items} /></LangProvider>;
const select = (first = "a", second = "b") => {
  fireEvent.change(screen.getByRole("combobox", { name: "Select first spectrum..." }), { target: { value: first } });
  fireEvent.change(screen.getByRole("combobox", { name: "Select second spectrum..." }), { target: { value: second } });
};
const compare = () => fireEvent.click(screen.getByRole("button", { name: "Compare" }));

describe("comparison selection ownership", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    localStorage.setItem("chemapp-lang", "en");
  });

  it("labels both selectors and announces current failures", async () => {
    vi.mocked(compareSpectra).mockRejectedValue(new Error("Comparison unavailable"));
    render(view());
    select();
    compare();
    expect(await screen.findByRole("alert")).toHaveTextContent("Comparison unavailable");
    expect(screen.getByRole("button", { name: "Compare" })).toBeEnabled();
  });

  it("clears completed results as soon as a different pair is selected", async () => {
    vi.mocked(compareSpectra).mockResolvedValue(comparison());
    render(view());
    select();
    compare();
    expect(await screen.findByText("Points: 10 vs 20")).toBeInTheDocument();
    select("a", "c");
    expect(screen.queryByText("Points: 10 vs 20")).not.toBeInTheDocument();
  });

  it("ignores an older result without ending the newer request", async () => {
    const oldRequest = deferred<ComparisonResult>();
    const newRequest = deferred<ComparisonResult>();
    vi.mocked(compareSpectra).mockReturnValueOnce(oldRequest.promise).mockReturnValueOnce(newRequest.promise);
    render(view());
    select();
    compare();
    expect(screen.getByRole("status")).toHaveTextContent("Comparing...");
    select("a", "c");
    compare();
    await act(async () => { oldRequest.resolve(comparison()); await oldRequest.promise; });
    expect(screen.queryByText("Points: 10 vs 20")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Comparing..." })).toBeDisabled();
    await act(async () => { newRequest.resolve(comparison("a", "c", 30)); await newRequest.promise; });
    expect(screen.getByText("Points: 30 vs 20")).toBeInTheDocument();
  });

  it("does not show a late failure after leaving and returning to the same pair", async () => {
    const oldRequest = deferred<ComparisonResult>();
    vi.mocked(compareSpectra).mockReturnValueOnce(oldRequest.promise).mockResolvedValueOnce(comparison("a", "b", 40));
    render(view());
    select();
    compare();
    select("a", "c");
    select();
    compare();
    expect(await screen.findByText("Points: 40 vs 20")).toBeInTheDocument();
    await act(async () => { oldRequest.reject(new Error("Old failure")); await oldRequest.promise.catch(() => undefined); });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByText("Points: 40 vs 20")).toBeInTheDocument();
  });

  it("clears an unavailable selection and hides its outstanding result", async () => {
    const pending = deferred<ComparisonResult>();
    vi.mocked(compareSpectra).mockReturnValue(pending.promise);
    const mounted = render(view());
    select();
    compare();
    mounted.rerender(view(spectra.filter((item) => item.id !== "b")));
    expect(screen.getByRole("combobox", { name: "Select second spectrum..." })).toHaveValue("");
    expect(screen.getByRole("button", { name: "Compare" })).toBeDisabled();
    await act(async () => { pending.resolve(comparison()); await pending.promise; });
    expect(screen.queryByText("Points: 10 vs 20")).not.toBeInTheDocument();
  });

  it("invalidates a comparison when the same spectrum receives a new revision", async () => {
    vi.mocked(compareSpectra).mockResolvedValue(comparison());
    const mounted = render(view());
    select();
    compare();
    expect(await screen.findByText("Points: 10 vs 20")).toBeInTheDocument();
    mounted.rerender(view(spectra.map((item) => item.id === "a" ? { ...item, result_revision: 2 } : item)));
    expect(screen.queryByText("Points: 10 vs 20")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Compare" })).toBeEnabled();
  });
});
