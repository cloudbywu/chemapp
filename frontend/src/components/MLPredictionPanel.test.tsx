import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LangProvider } from "../i18n/LangContext";
import type { AnalysisResult } from "../types/spectrum";
import MLPredictionPanel from "./MLPredictionPanel";

const elucidateStructure = vi.fn();

vi.mock("../services/api", () => ({
  elucidateStructure: (...args: unknown[]) => elucidateStructure(...args),
  elucidateCombined: vi.fn(),
}));

vi.mock("./MoleculeViewer", () => ({
  default: ({ smiles }: { smiles: string }) => <span>{smiles}</span>,
}));

const result: AnalysisResult = {
  technique: "NMR",
  result_revision: 3,
  peaks: [{
    position: 1.234,
    intensity: 100,
    area: 6,
    width: 0.01,
    assignment: "analyte",
    multiplicity: "s",
    coupling_constant: null,
  }],
  metrics: {},
  summary: "test",
};

describe("MLPredictionPanel", () => {
  beforeEach(() => {
    localStorage.setItem("chemapp-lang", "en");
    elucidateStructure.mockResolvedValue({
      method: "formula-constrained retrieval",
      candidates: [{
        rank: 1,
        smiles: "CC",
        molecular_formula: "C2H6",
        ranking_score: 0.42,
        evidence_level: "weak",
        matched_1h: 1,
      }],
      generated_candidates: [{ rank: 1, smiles: "CCC" }],
      query: {
        formula: "C2H6",
        preprocessing: { "1h": "clustered" },
      },
      warnings: ["Ranking score is not a calibrated probability."],
      forward_model: {
        status: "unsupported_modality",
        model_called: false,
        used_for_ranking: false,
        calibrated_probability: false,
        quantile_enabled: false,
      },
    });
  });

  it("passes NMR evidence fields and presents ranking evidence without confidence wording", async () => {
    render(
      <LangProvider>
        <MLPredictionPanel
          spectrumId="spectrum-1"
          hasResult
          technique="NMR"
          nucleus="1H"
          solvent="CDCl3"
          result={result}
        />
      </LangProvider>,
    );

    fireEvent.change(screen.getByLabelText("Molecular formula"), { target: { value: "C2H6" } });
    fireEvent.click(screen.getByText("Advanced candidate constraints"));
    fireEvent.change(screen.getByLabelText("Candidate SMILES"), {
      target: { value: "CC\nCCC\n" },
    });
    fireEvent.change(screen.getByLabelText("Required SMARTS"), {
      target: { value: "C\n[N,O]" },
    });
    fireEvent.change(screen.getByLabelText("Forbidden SMARTS"), {
      target: { value: "[Si]" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Rank candidates" }));

    await waitFor(() => expect(elucidateStructure).toHaveBeenCalledTimes(1));
    expect(elucidateStructure.mock.calls[0][0]).toMatchObject({
      formula: "C2H6",
      solvent: "CDCl3",
      generate_experimental: false,
      candidate_smiles: ["CC", "CCC"],
      required_smarts: ["C", "[N,O]"],
      forbidden_smarts: ["[Si]"],
      peaks_1h: [{
        shift: 1.234,
        integral: 6,
        multiplicity: "s",
        assignment: "analyte",
      }],
    });
    expect(await screen.findByText("0.4200")).toBeInTheDocument();
    expect(screen.getByText("Ranking score is not a calibrated probability.")).toBeInTheDocument();
    expect(screen.getByText("Experimental generated hypotheses")).toBeInTheDocument();
    expect(screen.queryByText(/confidence/i)).not.toBeInTheDocument();
    expect(screen.getByText(
      /Not run: this DP5q adapter requires observed ¹³C resonances\./,
    )).toBeInTheDocument();
  });

  it("keeps experimental generation off unless the user explicitly opts in", async () => {
    render(
      <LangProvider>
        <MLPredictionPanel
          spectrumId="spectrum-1"
          hasResult
          technique="NMR"
          nucleus="1H"
          result={result}
        />
      </LangProvider>,
    );

    fireEvent.click(screen.getByLabelText("Enable experimental T5 generation"));
    fireEvent.click(screen.getByRole("button", { name: "Rank candidates" }));

    await waitFor(() => expect(elucidateStructure).toHaveBeenCalledTimes(1));
    expect(elucidateStructure.mock.calls[0][0]).toMatchObject({
      generate_experimental: true,
    });
  });

  it("renders mandatory abstention and the complete hybrid pipeline audit trail", async () => {
    const stages = [
      "input_validation",
      "candidate_generation",
      "reference_scoring",
      "forward_scoring",
      "assignment",
      "ranking",
      "decision",
    ];
    elucidateStructure.mockResolvedValueOnce({
      schema_version: "chemapp.nmr.hybrid-prediction.v1",
      pipeline_version: "1.0.0",
      method: "hybrid-v1 candidate ranking",
      evidence_level: "weak",
      calibrated_probability: false,
      decision: {
        action: "abstain",
        reason_code: "correctness_probability_not_calibrated",
        selected_candidate_id: null,
        leading_hypothesis_candidate_id: "candidate-1",
      },
      pipeline: {
        stages: stages.map((stage) => ({
          stage,
          status: "completed",
          reason_code: stage === "decision"
            ? "correctness_probability_not_calibrated"
            : undefined,
        })),
      },
      candidates: [{
        rank: 1,
        candidate_id: "candidate-1",
        smiles: "CC",
        molecular_formula: "C2H6",
        ranking_score: 0.42,
        evidence_level: "weak",
      }],
      query: { formula: "C2H6" },
      forward_model: {
        status: "unsupported_modality",
        model_called: false,
        used_for_ranking: false,
        calibrated_probability: false,
      },
    });
    render(
      <LangProvider>
        <MLPredictionPanel
          spectrumId="spectrum-1"
          hasResult
          technique="NMR"
          nucleus="1H"
          result={result}
        />
      </LangProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Rank candidates" }));

    expect(await screen.findByText(
      "Automatic structure selection was refused",
    )).toBeInTheDocument();
    expect(screen.getByText(
      /Reason: correctness_probability_not_calibrated/,
    )).toBeInTheDocument();
    expect(screen.getByText(
      "Candidate order is relative evidence, not a probability of structural correctness.",
    )).toBeInTheDocument();

    fireEvent.click(screen.getByText("Inspect hybrid prediction pipeline"));
    const pipeline = screen.getByText("Inspect hybrid prediction pipeline")
      .closest("details");
    expect(pipeline).not.toBeNull();
    for (const stage of stages) {
      expect(within(pipeline as HTMLElement).getByText(stage)).toBeInTheDocument();
    }
    expect(within(pipeline as HTMLElement).getAllByText("completed")).toHaveLength(7);
    expect(screen.getByText("0.4200")).toBeInTheDocument();
  });

  it("uses reviewed 1H multiplet centers instead of raw peak lines", async () => {
    const resultWithMultiplets: AnalysisResult = {
      ...result,
      peaks: [
        { ...result.peaks[0], position: 1.22 },
        { ...result.peaks[0], position: 1.24 },
        { ...result.peaks[0], position: 7.28 },
      ],
      integrals: [{
        center_ppm: 1.23,
        start_ppm: 1.1,
        end_ppm: 1.3,
        raw_area: 10,
        relative_area: 6,
        intensity: 100,
      }],
      multiplets: [
        {
          center_ppm: 1.23,
          range_ppm: [1.2, 1.26],
          component_positions: [1.22, 1.24],
          n_peaks: 2,
          n_components: 2,
          multiplicity: "d",
          estimated_j_hz: 7,
          intensity_max: 100,
        },
        {
          center_ppm: 7.28,
          range_ppm: [7.27, 7.29],
          component_positions: [7.28],
          n_peaks: 1,
          n_components: 1,
          multiplicity: "s",
          estimated_j_hz: null,
          intensity_max: 20,
        },
      ],
    };
    render(
      <LangProvider>
        <MLPredictionPanel
          spectrumId="spectrum-multiplets"
          hasResult
          technique="NMR"
          nucleus="1H"
          result={resultWithMultiplets}
        />
      </LangProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Rank candidates" }));

    await waitFor(() => expect(elucidateStructure).toHaveBeenCalledTimes(1));
    expect(elucidateStructure.mock.calls[0][0].peaks_1h).toEqual([
      {
        shift: 1.23,
        intensity: 100,
        integral: 6,
        multiplicity: "d",
      },
      {
        shift: 7.28,
        intensity: 20,
        integral: null,
        multiplicity: "s",
      },
    ]);
  });

  it("shows 13C shadow diagnostics without changing candidate order", async () => {
    elucidateStructure.mockResolvedValueOnce({
      method: "formula-constrained retrieval",
      candidates: [
        {
          rank: 1,
          smiles: "CC",
          molecular_formula: "C2H6",
          ranking_score: 0.42,
          evidence_level: "weak",
          matched_13c: 1,
          forward_evidence: {
            status: "evaluated",
            relative_fit_rank: 2,
            mae_ppm: 1.234,
            rmse_ppm: 1.5,
            observed_coverage: 1,
            predicted_coverage: 0.5,
            prediction: {
              atom_predictions: [{ atom_index: 0, shift_ppm: 12.345 }],
            },
          },
        },
        {
          rank: 2,
          smiles: "CCC",
          molecular_formula: "C3H8",
          ranking_score: 0.3,
          evidence_level: "weak",
          matched_13c: 1,
          forward_evidence: {
            status: "evaluated",
            relative_fit_rank: 1,
            mae_ppm: 0.8,
            rmse_ppm: 1,
            observed_coverage: 1,
            predicted_coverage: 1,
            prediction: { atom_predictions: [] },
          },
        },
      ],
      query: { formula: "C2H6" },
      forward_model: {
        status: "completed",
        model_called: true,
        used_for_ranking: false,
        calibrated_probability: false,
        quantile_enabled: false,
        elapsed_ms: 42,
      },
    });

    render(
      <LangProvider>
        <MLPredictionPanel
          spectrumId="spectrum-13c"
          hasResult
          technique="NMR"
          nucleus="13C"
          result={result}
        />
      </LangProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Rank candidates" }));

    expect(await screen.findByText("¹³C forward fit (shadow evaluation)")).toBeInTheDocument();
    expect(screen.getByText(
      "Compares candidates with observed ¹³C shifts only; it does not affect ranking and is not a probability of structural correctness.",
    )).toBeInTheDocument();
    const cards = screen.getAllByRole("article");
    expect(within(cards[0]).getByText("#1")).toBeInTheDocument();
    expect(within(cards[0]).getByText("Relative fit rank: 2")).toBeInTheDocument();
    expect(within(cards[0]).getByText("MAE: 1.234 ppm")).toBeInTheDocument();
    expect(within(cards[0]).getByText("Observed coverage: 100%")).toBeInTheDocument();
    expect(within(cards[0]).getByText("Predicted-carbon coverage: 50%")).toBeInTheDocument();
    expect(within(cards[1]).getByText("#2")).toBeInTheDocument();
    expect(within(cards[1]).getByText("Relative fit rank: 1")).toBeInTheDocument();
    expect(screen.queryByText(/^Confidence$/i)).not.toBeInTheDocument();
  });

  it("renders matched-peak isotope labels as superscripts and keys cards by stable source ids", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    elucidateStructure.mockResolvedValueOnce({
      method: "formula-constrained retrieval",
      candidates: [
        {
          rank: 1,
          source_id: "src-1",
          smiles: "CC",
          molecular_formula: "C2H6",
          ranking_score: 0.5,
          evidence_level: "weak",
          matched_13c: 2,
          matched_1h: 1,
        },
        {
          rank: 1,
          source_id: "src-2",
          smiles: "CC",
          molecular_formula: "C2H6",
          ranking_score: 0.5,
          evidence_level: "weak",
          matched_13c: 1,
          matched_1h: 1,
        },
        {
          rank: 3,
          candidate_id: "cand-3",
          smiles: "CCC",
          molecular_formula: "C3H8",
          ranking_score: 0.4,
          evidence_level: "weak",
          matched_13c: 0,
          matched_1h: 3,
        },
      ],
      query: { formula: "C2H6" },
    });

    render(
      <LangProvider>
        <MLPredictionPanel
          spectrumId="spectrum-keys"
          hasResult
          technique="NMR"
          nucleus="13C"
          result={result}
        />
      </LangProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Rank candidates" }));

    const cards = await screen.findAllByRole("article");
    expect(cards).toHaveLength(3);
    // Isotope labels render as superscripts, not literal "^13C"/"^1H".
    expect(within(cards[0]).getByText("13", { selector: "sup" })).toBeInTheDocument();
    expect(within(cards[0]).getByText("1", { selector: "sup" })).toBeInTheDocument();
    expect(screen.queryByText(/\^13C/)).not.toBeInTheDocument();
    expect(screen.queryByText(/\^1H/)).not.toBeInTheDocument();
    // candidate_id falls back into the stable source id shown on the card.
    expect(within(cards[2]).getByText(/#cand-3/)).toBeInTheDocument();
    // Identical rank+smiles pairs stay distinct because the key uses sourceId.
    const duplicateKeyWarnings = errorSpy.mock.calls.filter((args) =>
      String(args[0]).includes("same key"),
    );
    expect(duplicateKeyWarnings).toHaveLength(0);
    errorSpy.mockRestore();
  });
});
