import { render, screen } from "@testing-library/react";
import SafeMarkdown from "./SafeMarkdown";

describe("SafeMarkdown", () => {
  it("preserves useful Markdown while removing executable content", () => {
    const { container } = render(
      <SafeMarkdown markdown={'**result** <img src="x" onerror="window.pwned=1"><script>alert(1)</script><form><input autofocus></form>'} />,
    );

    expect(screen.getByText("result")).toBeInTheDocument();
    expect(container.querySelector("strong")).toBeInTheDocument();
    expect(container.querySelector("script")).not.toBeInTheDocument();
    expect(container.querySelector("form")).not.toBeInTheDocument();
    expect(container.querySelector("input")).not.toBeInTheDocument();
    expect(container.querySelector("img")).not.toHaveAttribute("onerror");
  });
});
