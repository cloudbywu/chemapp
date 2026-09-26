/**
 * Defensive parsing helpers for untrusted backend payloads.
 *
 * Analysis metrics arrive as 'Record<string, unknown>'; single 'as number'
 * assertions compile under strict mode but crash the UI at runtime the moment
 * the backend changes shape. These helpers coerce only values that are
 * actually usable and return 'null' (or an empty list) otherwise, so panels
 * can render the MISSING_VALUE_PLACEHOLDER instead of throwing.
 */

/** Placeholder rendered when a metric is missing or not a finite number. */
export const MISSING_VALUE_PLACEHOLDER = "—";

/**
 * Parse an unknown value into a finite number.
 * Returns 'null' for missing values, empty strings, and non-numeric input.
 */
export function finiteNumber(value: unknown): number | null {
  if (typeof value === "string" && value.trim() === "") return null;
  const parsed = value == null ? null : Number(value);
  return parsed != null && Number.isFinite(parsed) ? parsed : null;
}

/**
 * Parse an unknown value into an array of finite numbers.
 * Non-array input yields an empty list; unusable entries are dropped.
 */
export function numberArray(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  const result: number[] = [];
  for (const item of value) {
    const parsed = finiteNumber(item);
    if (parsed != null) result.push(parsed);
  }
  return result;
}

/** Return the value when it is a non-empty string, otherwise 'null'. */
export function text(value: unknown): string | null {
  return typeof value === "string" && value !== "" ? value : null;
}

/** Return the value as a plain record when it is a non-array object, else null. */
export function record(value: unknown): Record<string, unknown> | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

/** 'toFixed' that renders the placeholder for missing values. */
export function formatFixed(value: number | null, digits: number): string {
  return value == null ? MISSING_VALUE_PLACEHOLDER : value.toFixed(digits);
}

/** 'toExponential' that renders the placeholder for missing values. */
export function formatExponential(value: number | null, digits: number): string {
  return value == null ? MISSING_VALUE_PLACEHOLDER : value.toExponential(digits);
}
