import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LangProvider } from "../i18n/LangContext";
import type * as api from "../services/api";
import FileUpload from "./FileUpload";

const uploadFile = vi.fn<typeof api.uploadFile>();
const listExamples = vi.fn<typeof api.listExamples>();
const loadExample = vi.fn<typeof api.loadExample>();

vi.mock("../services/api", () => ({
  uploadFile: (...args: Parameters<typeof api.uploadFile>) => uploadFile(...args),
  listExamples: (...args: Parameters<typeof api.listExamples>) => listExamples(...args),
  loadExample: (...args: Parameters<typeof api.loadExample>) => loadExample(...args),
}));

describe("FileUpload", () => {
  beforeEach(() => {
    listExamples.mockResolvedValue([]);
    uploadFile.mockReset();
  });

  it("uploads a file and reports the stored spectrum", async () => {
    const onUploaded = vi.fn();
    uploadFile.mockResolvedValue([{ id: "new-1", technique: "NMR", points: 10, name: "", has_result: false, summary: "" }]);
    const { container } = render(
      <LangProvider>
        <FileUpload onUploaded={onUploaded} />
      </LangProvider>,
    );
    const input = container.querySelector("input[type='file']") as HTMLInputElement;
    expect(input).not.toBeNull();
    fireEvent.change(input, {
      target: { files: [new File(["x"], "sample.txt")] },
    });
    await waitFor(() => expect(uploadFile).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(onUploaded).toHaveBeenCalledWith({
      id: "new-1",
      technique: "NMR",
      points: 10,
      name: "",
      has_result: false,
      summary: "",
    }));
  });

  it("shows an error when the upload fails", async () => {
    uploadFile.mockRejectedValue(new Error("boom"));
    const { container } = render(
      <LangProvider>
        <FileUpload onUploaded={vi.fn()} />
      </LangProvider>,
    );
    const input = container.querySelector("input[type='file']") as HTMLInputElement;
    fireEvent.change(input, {
      target: { files: [new File(["x"], "bad.txt")] },
    });
    await waitFor(() => expect(screen.getByText(/bad\.txt: boom/)).toBeInTheDocument());
  });

  it("keeps replacement-pattern characters in file names literal", async () => {
    uploadFile.mockRejectedValue(new Error("boom"));
    const { container } = render(
      <LangProvider>
        <FileUpload onUploaded={vi.fn()} />
      </LangProvider>,
    );
    const input = container.querySelector("input[type='file']") as HTMLInputElement;
    fireEvent.change(input, {
      target: { files: [new File(["x"], "A$&B.txt")] },
    });
    await waitFor(() => expect(screen.getByText(/A\$&B\.txt: boom/)).toBeInTheDocument());
  });
});
