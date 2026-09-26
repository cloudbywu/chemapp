import { describe, expect, it } from "vitest";
import { translations } from "./translations";

type Dict = Record<string, unknown>;

function flatten(obj: Dict, prefix = ""): string[] {
  return Object.entries(obj).flatMap(([key, value]) =>
    value !== null && typeof value === "object"
      ? flatten(value as Dict, `${prefix}${key}.`)
      : [`${prefix}${key}`],
  );
}

function get(obj: Dict, path: string): unknown {
  return path.split(".").reduce<unknown>(
    (acc, part) => (acc == null ? acc : (acc as Dict)[part]),
    obj,
  );
}

function placeholders(value: unknown): string[] {
  return typeof value === "string"
    ? [...value.matchAll(/\{[^{}]+\}/g)].map((match) => match[0]).sort()
    : [];
}

describe("translations", () => {
  const zh = translations.zh as Dict;
  const en = translations.en as Dict;

  it("has identical key structures in zh and en", () => {
    expect(flatten(en).sort()).toEqual(flatten(zh).sort());
  });

  it("uses the same placeholder tokens in zh and en strings", () => {
    for (const key of flatten(zh)) {
      expect(placeholders(get(en, key)), `placeholder mismatch for ${key}`).toEqual(
        placeholders(get(zh, key)),
      );
    }
  });

  it("defines the frontend-polish keys on both sides", () => {
    const required = [
      "training.legacyNotice",
      "compare.points",
      "viewer.selectRangeKeyboard",
      "action.exportCsv",
      "inference.totalArea",
      "inference.refTR",
      "inference.refType",
      "inference.exportMarkdown",
      "inference.exportHtml",
      "inference.exportWord",
      "inference.hplcCsv",
    ];
    for (const key of required) {
      expect(get(zh, key), `zh.${key}`).toEqual(expect.any(String));
      expect(get(en, key), `en.${key}`).toEqual(expect.any(String));
    }
  });
});
