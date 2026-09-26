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
