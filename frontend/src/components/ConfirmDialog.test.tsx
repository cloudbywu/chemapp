import type { ReactElement } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import ConfirmDialog from "./ConfirmDialog";

describe("ConfirmDialog", () => {
  it("has a labelled modal, traps focus, supports Escape, and restores focus", () => {
    const onCancel = vi.fn();
    const { rerender } = render(
      <>
        <button type="button">Opener</button>
        <ConfirmDialog
          open
          title="Delete spectrum?"
          message="This cannot be undone."
          confirmLabel="Delete"
          cancelLabel="Cancel"
          onConfirm={vi.fn()}
          onCancel={onCancel}
        />
      </>,
    );

    const dialog = screen.getByRole("alertdialog");
    expect(dialog).toHaveAccessibleName("Delete spectrum?");
    expect(dialog).toHaveAccessibleDescription("This cannot be undone.");
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onCancel).toHaveBeenCalledTimes(1);

    rerender(
      <>
        <button type="button">Opener</button>
        <ConfirmDialog
          open={false}
          title="Delete spectrum?"
          message="This cannot be undone."
          confirmLabel="Delete"
          cancelLabel="Cancel"
          onConfirm={vi.fn()}
          onCancel={onCancel}
        />
      </>,
    );
    expect(screen.getByRole("button", { name: "Opener" })).toBeInTheDocument();
  });

  it("keeps the page inert until the last of several open dialogs closes", () => {
    const shell = (dialogs: ReactElement[]) => (
      <div>
        <header className="app-header">header</header>
        <main className="app-main">main</main>
        {dialogs}
      </div>
    );
    const dialog = (key: string, title: string) => (
      <ConfirmDialog
        key={key}
        open
        title={title}
        message="message"
        confirmLabel="Confirm"
        cancelLabel="Cancel"
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    const { rerender } = render(shell([dialog("a", "First"), dialog("b", "Second")]));
    const header = document.querySelector(".app-header");
    const main = document.querySelector(".app-main");
    expect(header).toHaveAttribute("inert");
    expect(main).toHaveAttribute("inert");

    // Close the first dialog; the second is still open, so the page stays inert.
    rerender(shell([dialog("b", "Second")]));
    expect(header).toHaveAttribute("inert");
    expect(main).toHaveAttribute("inert");

    // Close the last dialog; only now is the inert flag cleared.
    rerender(shell([]));
    expect(header).not.toHaveAttribute("inert");
    expect(main).not.toHaveAttribute("inert");
  });
});

it("preserves focus and the original opener through rerenders and a busy retry", () => {
  const opener = document.createElement("button");
  document.body.append(opener);
  opener.focus();
  const firstCancel = vi.fn();
  const latestCancel = vi.fn();
  const props = { title: "Delete?", message: "Confirm deletion", confirmLabel: "Delete", cancelLabel: "Cancel", onConfirm: vi.fn() };
  const view = render(<ConfirmDialog {...props} open onCancel={firstCancel} />);
  const confirm = screen.getByRole("button", { name: "Delete" });
  confirm.focus();
  view.rerender(<ConfirmDialog {...props} open onCancel={latestCancel} />);
  expect(confirm).toHaveFocus();
  fireEvent.keyDown(document, { key: "Escape" });
  expect(firstCancel).not.toHaveBeenCalled();
  expect(latestCancel).toHaveBeenCalledTimes(1);

  view.rerender(<ConfirmDialog {...props} open busy onCancel={latestCancel} />);
  expect(screen.getByRole("alertdialog")).toHaveFocus();
  fireEvent.keyDown(document, { key: "Tab" });
  fireEvent.keyDown(document, { key: "Escape" });
  expect(latestCancel).toHaveBeenCalledTimes(1);
  expect(screen.getByRole("alertdialog")).toHaveFocus();

  view.rerender(<ConfirmDialog {...props} open onCancel={latestCancel} />);
  fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
  expect(screen.getByRole("button", { name: "Delete" })).toHaveFocus();
  fireEvent.keyDown(document, { key: "Tab" });
  expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();
  fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
  expect(screen.getByRole("button", { name: "Delete" })).toHaveFocus();

  view.rerender(<ConfirmDialog {...props} open={false} onCancel={latestCancel} />);
  expect(opener).toHaveFocus();
  opener.remove();
});

it("only dismisses the topmost dialog and keeps its opener valid if the lower modal closes first", () => {
  const opener = document.createElement("button");
  document.body.append(opener);
  opener.focus();
  const firstCancel = vi.fn();
  const secondCancel = vi.fn();
  const props = { message: "message", confirmLabel: "Confirm", cancelLabel: "Cancel", onConfirm: vi.fn() };
  const dialogs = (first: boolean, second: boolean) => <>
    <ConfirmDialog {...props} title="First" open={first} onCancel={firstCancel} />
    <ConfirmDialog {...props} title="Second" open={second} onCancel={secondCancel} />
  </>;
  const view = render(dialogs(true, true));
  fireEvent.keyDown(document, { key: "Escape" });
  expect(secondCancel).toHaveBeenCalledTimes(1);
  expect(firstCancel).not.toHaveBeenCalled();
  view.rerender(dialogs(false, true));
  expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();
  view.rerender(dialogs(false, false));
  expect(opener).toHaveFocus();
  opener.remove();
});

it("does not remove inert state that existed before the dialog opened", () => {
  const props = { title: "Confirm?", message: "message", confirmLabel: "Confirm", cancelLabel: "Cancel", onConfirm: vi.fn(), onCancel: vi.fn() };
  const view = render(<><main className="app-main" inert>Protected page</main><ConfirmDialog {...props} open /></>);
  view.rerender(<><main className="app-main" inert>Protected page</main><ConfirmDialog {...props} open={false} /></>);
  expect(document.querySelector(".app-main")).toHaveAttribute("inert");
});
