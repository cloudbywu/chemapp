import { describe, expect, it } from "vitest";
import {
  MISSING_VALUE_PLACEHOLDER,
  finiteNumber,
  formatExponential,
  formatFixed,
  numberArray,
  record,
  text,
} from "./number";

describe("finiteNumber", () => {
  it("accepts numbers, zero, and numeric strings", () => {
    expect(finiteNumber(3)).toBe(3);
    expect(finiteNumber(0)).toBe(0);
    expect(finiteNumber("2.5")).toBe(2.5);
  });

  it("rejects missing, empty, and non-finite values", () => {
    expect(finiteNumber(undefined)).toBeNull();
    expect(finiteNumber(null)).toBeNull();
    expect(finiteNumber("")).toBeNull();
    expect(finiteNumber("   ")).toBeNull();
    expect(finiteNumber("abc")).toBeNull();
    expect(finiteNumber(NaN)).toBeNull();
    expect(finiteNumber(Infinity)).toBeNull();
    expect(finiteNumber({})).toBeNull();
  });
});

describe("numberArray", () => {
  it("keeps finite entries and drops unusable ones", () => {
    expect(numberArray([1, "2", "x", null, NaN, undefined])).toEqual([1, 2]);
  });

  it("returns an empty list for non-array input", () => {
    expect(numberArray(undefined)).toEqual([]);
    expect(numberArray("1,2")).toEqual([]);
    expect(numberArray({ 0: 1 })).toEqual([]);
  });
});

describe("text", () => {
  it("returns non-empty strings only", () => {
    expect(text("1H")).toBe("1H");
    expect(text("")).toBeNull();
    expect(text(42)).toBeNull();
    expect(text(null)).toBeNull();
    expect(text(["a"])).toBeNull();
  });
});

describe("record", () => {
  it("returns plain objects and rejects arrays and primitives", () => {
    expect(record({ a: 1 })).toEqual({ a: 1 });
    expect(record([1])).toBeNull();
    expect(record("x")).toBeNull();
    expect(record(null)).toBeNull();
    expect(record(0)).toBeNull();
  });
});

describe("formatters", () => {
  it("formatFixed formats finite values and falls back to the placeholder", () => {
    expect(formatFixed(1.234, 2)).toBe("1.23");
    expect(formatFixed(0, 2)).toBe("0.00");
    expect(formatFixed(null, 2)).toBe(MISSING_VALUE_PLACEHOLDER);
  });

  it("formatExponential formats finite values and falls back to the placeholder", () => {
    expect(formatExponential(1234, 2)).toBe("1.23e+3");
    expect(formatExponential(null, 2)).toBe(MISSING_VALUE_PLACEHOLDER);
  });
});
