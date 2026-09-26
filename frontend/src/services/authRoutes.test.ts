import { describe, expect, it } from "vitest";
import { needsAdmin, needsReviewer } from "./authRoutes";

describe("auth route classification", () => {
  it("marks privileged endpoints as admin", () => {
    expect(needsAdmin("DELETE", "/api/spectra/abc")).toBe(true);
    expect(needsAdmin("GET", "/api/reviews/gold-manifest")).toBe(true);
    expect(needsAdmin("GET", "/api/reviews/capabilities")).toBe(true);
    expect(needsAdmin("POST", "/api/reviews/queue")).toBe(true);
    expect(needsAdmin("POST", "/api/standards")).toBe(true);
    expect(needsAdmin("GET", "/api/spectra")).toBe(false);
    expect(needsAdmin("POST", "/api/upload")).toBe(false);
  });

  it("marks review routes as reviewer unless admin", () => {
    expect(needsReviewer("GET", "/api/reviews/me")).toBe(true);
    expect(needsReviewer("GET", "/api/reviews/queue")).toBe(true);
    expect(needsReviewer("GET", "/api/reviews/abc")).toBe(true);
    expect(needsReviewer("POST", "/api/reviews/abc/submit")).toBe(true);
    expect(needsReviewer("GET", "/api/reviews/capabilities")).toBe(true);
    expect(needsReviewer("POST", "/api/reviews/abc/adjudicate")).toBe(false);
    expect(needsReviewer("GET", "/api/spectra")).toBe(false);
  });

  it("ignores query strings", () => {
    expect(needsAdmin("DELETE", "/api/spectra/abc?force=1")).toBe(true);
  });
});
