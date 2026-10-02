import { beforeEach, describe, expect, it, vi } from "vitest";

const { mockApi } = vi.hoisted(() => ({
  mockApi: {
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
    get: vi.fn(),
    post: vi.fn(),
  },
}));

vi.mock("axios", () => ({
  default: {
    create: vi.fn(() => mockApi),
    isCancel: vi.fn(() => false),
  },
}));

import {
  ApiError,
  REPORT_DOWNLOAD_TIMEOUT_MS,
  UPLOAD_TIMEOUT_MS,
  downloadBatchCsvZip,
  downloadBlob,
  downloadMarkdownReport,
  downloadSpectrumCsv,
  uploadFile,
  elucidateStructure,
  elucidateCombined,
  processNmrSpectrum,
  resetNmrSpectrum,
} from "./api";

// Capture the installed interceptor before individual tests clear mock calls.
const rejectApiResponse = mockApi.interceptors.response.use.mock.calls[0][1] as (
  error: unknown,
) => Promise<never>;

describe("API error details", () => {
  it.each(["later_result_edit", "unverifiable_history"])("preserves the structured %s protection reason for localization", async (reason) => {
    const detail = {
      code: "ai_undo_blocked",
      reason,
      message: "AI undo was blocked to preserve the current result",
      current_revision: 7,
    };

    await expect(rejectApiResponse({
      response: { status: 409, data: { detail } },
    })).rejects.toMatchObject({
      name: "ApiError",
      message: detail.message,
      status: 409,
      detail,
    });
  });

  it("keeps string errors and existing two-argument construction compatible", async () => {
    await expect(rejectApiResponse({
      response: { status: 404, data: { detail: "Spectrum not found" } },
    })).rejects.toMatchObject({
      name: "ApiError",
      message: "Spectrum not found",
      status: 404,
      detail: "Spectrum not found",
    });
    expect(new ApiError("offline", 503).detail).toBeUndefined();
  });
});

describe("NMR processing revision binding", () => {
  beforeEach(() => mockApi.post.mockReset());

  it("sends both reviewed revisions for preview and apply", async () => {
    mockApi.post.mockResolvedValue({ data: {} });
    for (const preview_only of [true, false]) {
      const payload = { expected_revision: 4, expected_result_revision: 7, preview_only, invert: true };
      await processNmrSpectrum("nmr-1", payload);
      expect(mockApi.post).toHaveBeenLastCalledWith("/api/nmr/nmr-1/process", payload);
    }
  });

  it("sends both reviewed revisions for reset", async () => {
    mockApi.post.mockResolvedValue({ data: {} });
    await resetNmrSpectrum("nmr-1", 4, 7);
    expect(mockApi.post).toHaveBeenCalledExactlyOnceWith("/api/nmr/nmr-1/reset", {
      expected_revision: 4, expected_result_revision: 7,
    });
  });
});

describe("downloadBlob", () => {
  let createdAnchor: HTMLAnchorElement | null;

  beforeEach(() => {
    vi.clearAllMocks();
    createdAnchor = null;
    URL.createObjectURL = vi.fn(() => "blob:mock-url");
    URL.revokeObjectURL = vi.fn();
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    const realCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((
      tagName: string,
      options?: ElementCreationOptions,
    ) => {
      const element = realCreateElement(tagName, options);
      if (tagName === "a") createdAnchor = element as HTMLAnchorElement;
      return element;
    });
  });

  it("creates an object URL, clicks a temporary anchor, and revokes the URL", () => {
    const blob = new Blob(["payload"]);
    downloadBlob(blob, "report.md");

    expect(URL.createObjectURL).toHaveBeenCalledWith(blob);
    expect(createdAnchor).not.toBeNull();
    expect(createdAnchor?.download).toBe("report.md");
    expect(createdAnchor?.href).toBe("blob:mock-url");
    expect(createdAnchor?.isConnected).toBe(false);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:mock-url");
  });

  it("uploads with the 10-minute timeout aligned with the nginx 600s ceiling", async () => {
    mockApi.post.mockResolvedValue({ data: [] });
    const file = new File(["x"], "x.txt");

    await uploadFile(file);

    expect(UPLOAD_TIMEOUT_MS).toBe(600_000);
    expect(mockApi.post).toHaveBeenCalledWith("/api/upload", expect.any(FormData), {
      timeout: UPLOAD_TIMEOUT_MS,
    });
  });

  it("downloads the markdown report as a blob with the 5-minute timeout", async () => {
    const blob = new Blob(["# report"]);
    mockApi.post.mockResolvedValue({ data: blob });

    await downloadMarkdownReport(["s1"], "Title");

    expect(REPORT_DOWNLOAD_TIMEOUT_MS).toBe(300_000);
    expect(mockApi.post).toHaveBeenCalledWith(
      "/api/reports/markdown",
      { ids: ["s1"], title: "Title", include_peaks: true },
      { responseType: "blob", timeout: REPORT_DOWNLOAD_TIMEOUT_MS },
    );
    expect(URL.createObjectURL).toHaveBeenCalledWith(blob);
    expect(createdAnchor?.download).toBe("chemapp-report.md");
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:mock-url");
  });

  it("downloads the batch CSV zip with the 5-minute timeout", async () => {
    mockApi.post.mockResolvedValue({ data: new Blob(["zip"]) });

    await downloadBatchCsvZip(["s1", "s2"]);

    expect(mockApi.post).toHaveBeenCalledWith(
      "/api/spectra/export/csv.zip",
      { ids: ["s1", "s2"] },
      { responseType: "blob", timeout: REPORT_DOWNLOAD_TIMEOUT_MS },
    );
    expect(createdAnchor?.download).toBe("chemapp-export.zip");
  });

  it("downloads a single-spectrum CSV with the 5-minute timeout and a custom filename", async () => {
    mockApi.get.mockResolvedValue({ data: new Blob(["csv"]) });

    await downloadSpectrumCsv("s1", "custom.csv");

    expect(mockApi.get).toHaveBeenCalledWith("/api/spectra/s1/csv", {
      responseType: "blob",
      timeout: REPORT_DOWNLOAD_TIMEOUT_MS,
    });
    expect(createdAnchor?.download).toBe("custom.csv");
  });
});


describe("elucidation evidence request wiring", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockApi.post.mockResolvedValue({ data: { candidates: [] } });
  });

  it("preserves the selected continuous proton source and cancellation signal", async () => {
    const controller = new AbortController();
    const payload = { peaks_13c: [], peaks_1h: [{ shift: 1.2 }], spectrum_1h_id: "proton-2", generate_experimental: true };
    await elucidateStructure(payload, controller.signal);
    expect(mockApi.post).toHaveBeenCalledWith("/api/ml/elucidate/predict", payload, {
      signal: controller.signal, timeout: 120000,
    });
  });

  it("preserves the exact pair for server-side nucleus and source resolution", async () => {
    const controller = new AbortController();
    await elucidateCombined("carbon-1", "proton-2", "C2H6O", true, controller.signal);
    expect(mockApi.post).toHaveBeenCalledWith("/api/ml/elucidate/predict/combined", {
      id1: "carbon-1", id2: "proton-2", formula: "C2H6O", generate_experimental: true,
    }, { signal: controller.signal, timeout: 120000 });
  });
});
