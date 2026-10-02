import { createRef } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LangProvider } from "../i18n/LangContext";
import type * as api from "../services/api";
import type { SpectrumListItem } from "../types/spectrum";
import FileUpload, { type FileUploadHandle } from "./FileUpload";

const uploadFile = vi.fn<typeof api.uploadFile>();
const listExamples = vi.fn<typeof api.listExamples>();
const loadExample = vi.fn<typeof api.loadExample>();
const storedSpectrum: SpectrumListItem = {
  id: "new-1", technique: "NMR", points: 10, name: "New spectrum", has_result: false, summary: "",
};
const example = { technique: "NMR", label: "Example spectrum", path: "example.txt", filename: "example.txt", size: 1 };

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => { resolve = resolvePromise; });
  return { promise, resolve };
}

vi.mock("../services/api", () => ({
  uploadFile: (...args: Parameters<typeof api.uploadFile>) => uploadFile(...args),
  listExamples: (...args: Parameters<typeof api.listExamples>) => listExamples(...args),
  loadExample: (...args: Parameters<typeof api.loadExample>) => loadExample(...args),
}));

describe("FileUpload", () => {
  beforeEach(() => {
    localStorage.setItem("chemapp-lang", "en");
    listExamples.mockReset();
    listExamples.mockResolvedValue([]);
    uploadFile.mockReset();
    loadExample.mockReset();
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

  it.each(["upload", "example"] as const)("uses the latest callback when a pending %s completes", async (operation) => {
    const pending = deferred<SpectrumListItem[]>();
    const oldCallback = vi.fn();
    const currentCallback = vi.fn();
    listExamples.mockResolvedValue([example]);
    uploadFile.mockReturnValue(pending.promise);
    loadExample.mockReturnValue(pending.promise);
    const view = render(<LangProvider><FileUpload onUploaded={oldCallback} /></LangProvider>);
    const exampleButton = await screen.findByRole("button", { name: /Example spectrum/ });
    if (operation === "upload") {
      fireEvent.change(view.container.querySelector("input")!, {
        target: { files: [new File(["x"], "sample.txt")] },
      });
    } else fireEvent.click(exampleButton);

    view.rerender(<LangProvider><FileUpload onUploaded={currentCallback} /></LangProvider>);
    await act(async () => {
      pending.resolve([storedSpectrum]);
      await pending.promise;
    });
    expect(oldCallback).not.toHaveBeenCalled();
    expect(currentCallback).toHaveBeenCalledExactlyOnceWith(storedSpectrum);
  });

  it("blocks duplicate drops and example loads throughout an upload, then allows another operation", async () => {
    const pending = deferred<SpectrumListItem[]>();
    uploadFile.mockReturnValue(pending.promise);
    listExamples.mockResolvedValue([example]);
    loadExample.mockResolvedValue([storedSpectrum]);
    const onUploaded = vi.fn();
    const view = render(<LangProvider><FileUpload onUploaded={onUploaded} /></LangProvider>);
    const exampleButton = await screen.findByRole("button", { name: /Example spectrum/ });
    const dropzone = view.container.querySelector(".dropzone")!;
    const input = view.container.querySelector("input")!;
    const drop = { dataTransfer: { files: [new File(["x"], "sample.txt")] } };
    act(() => {
      fireEvent.drop(dropzone, drop);
      fireEvent.drop(dropzone, drop);
      fireEvent.click(exampleButton);
    });
    expect(uploadFile).toHaveBeenCalledTimes(1);
    expect(loadExample).not.toHaveBeenCalled();
    expect(dropzone).toBeDisabled();
    expect(input).toBeDisabled();
    expect(exampleButton).toBeDisabled();

    await act(async () => {
      pending.resolve([storedSpectrum]);
      await pending.promise;
    });
    expect(dropzone).toBeEnabled();
    expect(input).toBeEnabled();
    fireEvent.click(exampleButton);
    await waitFor(() => expect(loadExample).toHaveBeenCalledExactlyOnceWith(example.path));
    await waitFor(() => expect(onUploaded).toHaveBeenCalledTimes(2));
  });

  it("blocks duplicate examples and file gestures while an example is loading", async () => {
    const pending = deferred<SpectrumListItem[]>();
    listExamples.mockResolvedValue([example]);
    loadExample.mockReturnValue(pending.promise);
    const view = render(<LangProvider><FileUpload onUploaded={vi.fn()} /></LangProvider>);
    const exampleButton = await screen.findByRole("button", { name: /Example spectrum/ });
    const dropzone = view.container.querySelector(".dropzone")!;
    const input = view.container.querySelector("input")!;
    const files = [new File(["x"], "sample.txt")];
    act(() => {
      fireEvent.click(exampleButton);
      fireEvent.click(exampleButton);
      fireEvent.drop(dropzone, { dataTransfer: { files } });
      fireEvent.change(input, { target: { files } });
    });
    expect(loadExample).toHaveBeenCalledTimes(1);
    expect(uploadFile).not.toHaveBeenCalled();
    expect(dropzone).toBeDisabled();
    expect(input).toBeDisabled();
    await act(async () => {
      pending.resolve([storedSpectrum]);
      await pending.promise;
    });
    expect(dropzone).toBeEnabled();
  });

  it("snapshots the whole file batch before waiting for the first upload", async () => {
    const pending = deferred<SpectrumListItem[]>();
    const first = new File(["first"], "first.txt");
    const second = new File(["second"], "second.txt");
    const files = [first, second];
    uploadFile.mockReturnValueOnce(pending.promise).mockResolvedValueOnce([storedSpectrum]);
    const onUploaded = vi.fn();
    const view = render(<LangProvider><FileUpload onUploaded={onUploaded} /></LangProvider>);
    fireEvent.drop(view.container.querySelector(".dropzone")!, { dataTransfer: { files } });
    files.length = 0;
    await act(async () => {
      pending.resolve([storedSpectrum]);
      await pending.promise;
    });
    expect(uploadFile).toHaveBeenNthCalledWith(1, first);
    expect(uploadFile).toHaveBeenNthCalledWith(2, second);
    expect(onUploaded).toHaveBeenCalledTimes(2);
  });

  it.each(["upload", "example"] as const)("releases the operation lock after a failed %s so it can be retried", async (operation) => {
    listExamples.mockResolvedValue([example]);
    uploadFile.mockRejectedValueOnce(new Error("boom")).mockResolvedValueOnce([storedSpectrum]);
    loadExample.mockRejectedValueOnce(new Error("boom")).mockResolvedValueOnce([storedSpectrum]);
    const onUploaded = vi.fn();
    const view = render(<LangProvider><FileUpload onUploaded={onUploaded} /></LangProvider>);
    const exampleButton = await screen.findByRole("button", { name: /Example spectrum/ });
    const input = view.container.querySelector("input")!;
    const start = () => {
      if (operation === "upload") {
        fireEvent.change(input, { target: { files: [new File(["x"], "sample.txt")] } });
      } else fireEvent.click(exampleButton);
    };
    start();
    expect(await screen.findByRole("alert")).toHaveTextContent("boom");
    expect(exampleButton).toBeEnabled();
    expect(input).toBeEnabled();
    start();
    await waitFor(() => expect(onUploaded).toHaveBeenCalledExactlyOnceWith(storedSpectrum));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it.each(["upload", "example"] as const)("ignores a pending %s after unmount and never starts queued files", async (operation) => {
    const pending = deferred<SpectrumListItem[]>();
    listExamples.mockResolvedValue([example]);
    uploadFile.mockReturnValue(pending.promise);
    loadExample.mockReturnValue(pending.promise);
    const onUploaded = vi.fn();
    const view = render(<LangProvider><FileUpload onUploaded={onUploaded} /></LangProvider>);
    const exampleButton = await screen.findByRole("button", { name: /Example spectrum/ });
    if (operation === "upload") {
      fireEvent.change(view.container.querySelector("input")!, {
        target: { files: [new File(["a"], "a.txt"), new File(["b"], "b.txt")] },
      });
    } else fireEvent.click(exampleButton);
    view.unmount();
    await act(async () => {
      pending.resolve([storedSpectrum]);
      await pending.promise;
    });
    expect(onUploaded).not.toHaveBeenCalled();
    expect(uploadFile).toHaveBeenCalledTimes(operation === "upload" ? 1 : 0);
  });
});


it("opens the real file chooser and expands/focuses examples from workspace shortcuts", async () => {
  localStorage.setItem("chemapp-lang", "en");
  listExamples.mockResolvedValue([example]);
  const ref = createRef<FileUploadHandle>();
  const view = render(<LangProvider><FileUpload ref={ref} onUploaded={vi.fn()} /></LangProvider>);
  const button = await screen.findByRole("button", { name: /Example spectrum/ });
  const input = view.container.querySelector("input")!;
  const choose = vi.spyOn(input, "click");
  act(() => ref.current?.chooseFiles());
  expect(choose).toHaveBeenCalledTimes(1);
  const section = view.container.querySelector(".example-loader") as HTMLDetailsElement;
  section.open = false;
  act(() => ref.current?.showExamples());
  expect(section.open).toBe(true);
  expect(button).toHaveFocus();
});

it("keeps long technique badges and names in separate sample-card fields", async () => {
  localStorage.setItem("chemapp-lang", "en");
  listExamples.mockResolvedValue([{ ...example, technique: "Fluorescence", label: "A very long JEOL-like sample label with multiple experiment details" }]);
  render(<LangProvider><FileUpload onUploaded={vi.fn()} /></LangProvider>);
  const button = await screen.findByRole("button", { name: /A very long/ });
  expect(button.querySelector(".example-technique")).toHaveTextContent("Fluorescence");
  expect(button.querySelector(".example-name")).toHaveTextContent("A very long JEOL-like sample label");
  expect(button.querySelector(".example-arrow")).toHaveAttribute("aria-hidden", "true");
});
