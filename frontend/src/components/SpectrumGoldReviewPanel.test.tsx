import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LangProvider } from "../i18n/LangContext";
import type * as api from "../services/api";
import type {
  SpectrumData,
  SpectrumReviewItem,
} from "../types/spectrum";
import SpectrumGoldReviewPanel from "./SpectrumGoldReviewPanel";
import { SpectrumReviewQueuePanel } from "./SpectrumGoldReviewPanel";

const getSpectrumReviewCapabilities = vi.fn<typeof api.getSpectrumReviewCapabilities>();
const getSpectrumReview = vi.fn<typeof api.getSpectrumReview>();
const getSpectrumReviewAudit = vi.fn<typeof api.getSpectrumReviewAudit>();
const enqueueSpectrumReview = vi.fn<typeof api.enqueueSpectrumReview>();
const submitSpectrumReview = vi.fn<typeof api.submitSpectrumReview>();
const adjudicateSpectrumReview = vi.fn<typeof api.adjudicateSpectrumReview>();
const downloadGoldSpectrumManifest = vi.fn<typeof api.downloadGoldSpectrumManifest>();
const listSpectrumReviewQueue = vi.fn<typeof api.listSpectrumReviewQueue>();

vi.mock("../services/api", () => ({
  ApiError: class ApiError extends Error {
    status: number;

    constructor(message: string, status: number) {
      super(message);
      this.status = status;
    }
  },
  getSpectrumReviewCapabilities: (
    ...args: Parameters<typeof api.getSpectrumReviewCapabilities>
  ) => getSpectrumReviewCapabilities(...args),
  getSpectrumReview: (...args: Parameters<typeof api.getSpectrumReview>) => (
    getSpectrumReview(...args)
  ),
  getSpectrumReviewAudit: (
    ...args: Parameters<typeof api.getSpectrumReviewAudit>
  ) => getSpectrumReviewAudit(...args),
  enqueueSpectrumReview: (
    ...args: Parameters<typeof api.enqueueSpectrumReview>
  ) => enqueueSpectrumReview(...args),
  submitSpectrumReview: (
    ...args: Parameters<typeof api.submitSpectrumReview>
  ) => submitSpectrumReview(...args),
  adjudicateSpectrumReview: (
    ...args: Parameters<typeof api.adjudicateSpectrumReview>
  ) => adjudicateSpectrumReview(...args),
  downloadGoldSpectrumManifest: (
    ...args: Parameters<typeof api.downloadGoldSpectrumManifest>
  ) => downloadGoldSpectrumManifest(...args),
  listSpectrumReviewQueue: (
    ...args: Parameters<typeof api.listSpectrumReviewQueue>
  ) => listSpectrumReviewQueue(...args),
}));

const spectrum: SpectrumData = {
  id: "nmr-review-1",
  spectrum_revision: 4,
  result_revision: 3,
  technique: "NMR",
  x_data: [8, 7, 6],
  y_data: [0, 1, 0],
  x_label: "Chemical shift",
  y_label: "Intensity",
  x_unit: "ppm",
  y_unit: "a.u.",
  parameters: { nucleus: "1H" },
  metadata: {
    smiles: "CCO",
    inchi_key: "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    solvent: "CDCl3",
  },
  peaks: [],
  source_file: "review.jdx",
};

const queuedItem: SpectrumReviewItem = {
  schema_version: "chemapp.spectrum-review.v1",
  id: "queue-1",
  spectrum_id: spectrum.id,
  cycle: 1,
  status: "awaiting_second",
  queue_revision: 2,
  structure_smiles: "CCO",
  structure_source: "signed source record",
  molecule_id: "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
  source_collection: "independent collection",
  source_record_id: "record-1",
  independence_group: "lab-batch-1",
  license_id: "CC-BY-4.0",
  provenance_uri: "",
  rights_confirmed: true,
  snapshot_sha256: "a".repeat(64),
  spectrum_sha256: "b".repeat(64),
  result_sha256: "c".repeat(64),
  spectrum_revision: 4,
  result_revision: 3,
  facts: {
    technique: "NMR",
    source_file: "review.jdx",
    source_checksum_sha256: null,
    nucleus: "1H",
    solvent: "CDCl3",
    axis_unit: "ppm",
    axis_direction: "descending",
    point_count: 3,
    peak_count: 2,
    axis_min: 6,
    axis_max: 8,
  },
  review_count: 1,
  has_submitted: false,
  own_review: null,
  final_decision: null,
  final_reason: "",
  created_at: "2026-07-26T00:00:00Z",
  updated_at: "2026-07-26T00:01:00Z",
};

function renderPanel(onDirtyChange = vi.fn()) {
  return {
    ...render(
      <LangProvider>
        <SpectrumGoldReviewPanel
          spectrum={spectrum}
          onDirtyChange={onDirtyChange}
        />
      </LangProvider>,
    ),
    onDirtyChange,
  };
}

describe("SpectrumGoldReviewPanel", () => {
  beforeEach(() => {
    localStorage.setItem("chemapp-lang", "zh");
    getSpectrumReviewCapabilities.mockReset();
    getSpectrumReview.mockReset();
    getSpectrumReviewAudit.mockReset();
    enqueueSpectrumReview.mockReset();
    submitSpectrumReview.mockReset();
    adjudicateSpectrumReview.mockReset();
    downloadGoldSpectrumManifest.mockReset();
    listSpectrumReviewQueue.mockReset();
    getSpectrumReviewCapabilities.mockResolvedValue({
      reviewer: {
        configured: true,
        authenticated: true,
        subject: "bob",
      },
      admin: {
        configured: true,
        authenticated: true,
        subject: "curation-admin",
      },
      separation_ok: true,
      can_review: true,
      can_manage_queue: true,
      can_view_audit: true,
      can_adjudicate: true,
      can_export_gold: true,
    });
  });

  it("submits all five checks without accepting a client reviewer identity", async () => {
    getSpectrumReview.mockResolvedValue({
      reviewer_id: "bob",
      item: queuedItem,
    });
    submitSpectrumReview.mockResolvedValue({
      reviewer_id: "bob",
      item: {
        ...queuedItem,
        status: "accepted",
        queue_revision: 3,
        review_count: 2,
        has_submitted: true,
        final_decision: "accept",
      },
    });

    const { container, onDirtyChange } = renderPanel();
    expect(await screen.findByText("bob")).toBeInTheDocument();
    expect(
      container.querySelector(".gold-review-status-row")?.textContent,
    ).toContain("1/2");
    expect(screen.queryByText("alice")).not.toBeInTheDocument();

    for (const label of [
      "结构",
      "核种",
      "轴方向与单位",
      "峰表",
      "溶剂",
    ]) {
      fireEvent.change(screen.getByLabelText(label, { selector: "select" }), {
        target: { value: "pass" },
      });
    }
    await waitFor(() => expect(onDirtyChange).toHaveBeenCalledWith(true));

    fireEvent.click(screen.getByRole("button", {
      name: "提交不可变复核",
    }));
    const dialog = screen.getByRole("alertdialog");
    fireEvent.click(within(dialog).getByRole("button", {
      name: "提交不可变复核",
    }));

    await waitFor(() => expect(submitSpectrumReview).toHaveBeenCalledOnce());
    const [, payload] = submitSpectrumReview.mock.calls[0];
    expect(payload).toMatchObject({
      verdict: "accept",
      expected_queue_revision: 2,
      expected_snapshot_sha256: "a".repeat(64),
      checks: {
        structure: "pass",
        nucleus: "pass",
        axis: "pass",
        peaks: "pass",
        solvent: "pass",
      },
    });
    expect(payload).not.toHaveProperty("reviewer_id");
    expect(await screen.findByText(/可进入 Gold/)).toBeInTheDocument();
  });

  it("requires governance fields, rights confirmation, and confirmation to enqueue", async () => {
    const { ApiError } = await import("../services/api");
    getSpectrumReview.mockRejectedValue(
      new ApiError("not queued", 404),
    );
    enqueueSpectrumReview.mockResolvedValue({
      ...queuedItem,
      status: "pending",
      queue_revision: 1,
      review_count: 0,
    });

    renderPanel();
    await screen.findByText("加入人工复核队列");
    fireEvent.change(screen.getByLabelText("结构依据"), {
      target: { value: "signed source record" },
    });
    fireEvent.change(screen.getByLabelText("来源集合"), {
      target: { value: "independent collection" },
    });
    fireEvent.change(screen.getByLabelText("独立分组"), {
      target: { value: "lab-batch-1" },
    });
    fireEvent.change(screen.getByLabelText("许可证 / 权利标识"), {
      target: { value: "CC-BY-4.0" },
    });
    fireEvent.change(screen.getByLabelText("HTTPS 来源链接"), {
      target: { value: "https://example.test/record-1" },
    });
    fireEvent.click(screen.getByLabelText(/我确认该数据的权利/));

    const enqueueButton = screen.getByRole("button", { name: "确认入队" });
    expect(enqueueButton).toBeEnabled();
    fireEvent.click(enqueueButton);
    fireEvent.click(within(screen.getByRole("alertdialog")).getByRole(
      "button",
      { name: "确认入队" },
    ));

    await waitFor(() => expect(enqueueSpectrumReview).toHaveBeenCalledWith(
      expect.objectContaining({
        spectrum_id: spectrum.id,
        rights_confirmed: true,
        expected_spectrum_revision: 4,
        expected_result_revision: 3,
      }),
    ));
  });

  it("lists the authenticated review queue and opens a selected spectrum", async () => {
    listSpectrumReviewQueue.mockResolvedValue({
      reviewer_id: "bob",
      items: [queuedItem],
      count: 1,
    });
    const onOpenSpectrum = vi.fn();
    render(
      <LangProvider>
        <SpectrumReviewQueuePanel onOpenSpectrum={onOpenSpectrum} />
      </LangProvider>,
    );

    expect(await screen.findByText("review.jdx")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", {
      name: `打开谱图 ${spectrum.id} 进行复核`,
    }));
    expect(onOpenSpectrum).toHaveBeenCalledWith(spectrum.id);
  });

  it("hides review-management controls from a reviewer-only capability", async () => {
    getSpectrumReviewCapabilities.mockResolvedValue({
      reviewer: {
        configured: true,
        authenticated: true,
        subject: "bob",
      },
      admin: {
        configured: true,
        authenticated: false,
        subject: "",
      },
      separation_ok: true,
      can_review: true,
      can_manage_queue: false,
      can_view_audit: false,
      can_adjudicate: false,
      can_export_gold: false,
    });
    getSpectrumReview.mockResolvedValue({
      reviewer_id: "bob",
      item: queuedItem,
    });

    renderPanel();
    expect(await screen.findByText("bob")).toBeInTheDocument();
    expect(screen.queryByRole("button", {
      name: "导出 Gold manifest",
    })).not.toBeInTheDocument();
    expect(screen.getByRole("button", {
      name: "提交不可变复核",
    })).toBeInTheDocument();
  });

  it("uses the protected audit view and hides review submission for admin-only access", async () => {
    getSpectrumReviewCapabilities.mockResolvedValue({
      reviewer: {
        configured: true,
        authenticated: false,
        subject: "",
      },
      admin: {
        configured: true,
        authenticated: true,
        subject: "curation-admin",
      },
      separation_ok: true,
      can_review: false,
      can_manage_queue: true,
      can_view_audit: true,
      can_adjudicate: true,
      can_export_gold: true,
    });
    getSpectrumReviewAudit.mockResolvedValue({
      queue: queuedItem,
      reviews: [],
      events: [],
    });

    renderPanel();
    expect(await screen.findByText("curation-admin")).toBeInTheDocument();
    expect(getSpectrumReview).not.toHaveBeenCalled();
    expect(getSpectrumReviewAudit).toHaveBeenCalledWith(spectrum.id);
    expect(screen.getByRole("button", {
      name: "导出 Gold manifest",
    })).toBeInTheDocument();
    expect(screen.queryByRole("button", {
      name: "提交不可变复核",
    })).not.toBeInTheDocument();
  });
});
