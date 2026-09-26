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
  REPORT_DOWNLOAD_TIMEOUT_MS,
  UPLOAD_TIMEOUT_MS,
  downloadBatchCsvZip,
  downloadBlob,
  downloadMarkdownReport,
  downloadSpectrumCsv,
  uploadFile,
} from "./api";

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
