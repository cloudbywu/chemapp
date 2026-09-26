import { useMemo } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";

interface Props {
  markdown: string;
  className?: string;
}

/**
 * Render untrusted model output without allowing it to execute HTML, scripts,
 * event handlers, embedded documents, forms, or style-based UI spoofing.
 */
export default function SafeMarkdown({ markdown, className = "" }: Props) {
  const safeHtml = useMemo(() => {
    const parsed = marked.parse(markdown, {
      breaks: true,
      async: false,
    }) as string;

    return DOMPurify.sanitize(parsed, {
      USE_PROFILES: { html: true },
      FORBID_TAGS: ["style", "form", "input", "button", "textarea", "select", "option", "iframe", "object", "embed", "svg", "math"],
      FORBID_ATTR: ["style", "srcdoc"],
      ALLOW_DATA_ATTR: false,
    });
  }, [markdown]);

  return <div className={className} dangerouslySetInnerHTML={{ __html: safeHtml }} />;
}
