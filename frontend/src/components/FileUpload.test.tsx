import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LangProvider } from "../i18n/LangContext";
import FileUpload from "./FileUpload";

const uploadFile = vi.fn();
const listExamples = vi.fn();
const loadExample = vi.fn();

vi.mock("../services/api", () => ({
  uploadFile: (...args: unknown[]) => uploadFile(...args),
  listExamples: (...args: unknown[]) => listExamples(...args),
  loadExample: (...args: unknown[]) => loadExample(...args),
}));

describe("FileUpload", () => {
  beforeEach(() => {
    listExamples.mockResolvedValue([]);
    uploadFile.mockReset();
  });

  it("uploads a file and reports the stored spectrum", async () => {
    const onUploaded = vi.fn();
    uploadFile.mockResolvedValue([{ id: "new-1", technique: "NMR", points: 10, summary: "" }]);
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
