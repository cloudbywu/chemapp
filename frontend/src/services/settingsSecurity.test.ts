import {
  clearAIKey,
  loadAISettings,
  saveAISettings,
} from "./aiSettings";
import {
  requestNeedsAdmin,
  requestNeedsReviewer,
} from "./api";
import {
  chemAppAuthHeaders,
  clearChemAppTokens,
  saveChemAppTokens,
} from "./authTokens";

describe("credential storage", () => {
  afterEach(() => {
    clearAIKey();
    clearChemAppTokens();
  });

  it("never persists external AI API keys", () => {
    saveAISettings({
      preset: "custom",
      apiKey: "secret-ai-key",
      baseUrl: "https://example.test",
      model: "test-model",
      allowDataSharing: true,
    });

    expect(loadAISettings().apiKey).toBe("secret-ai-key");
    expect(localStorage.getItem("chemapp-ai-settings")).not.toContain("secret-ai-key");
  });

  it("scopes admin and reviewer credentials independently", () => {
    saveChemAppTokens({
      accessToken: "access",
      adminToken: "admin",
      reviewerToken: "reviewer",
    });

    expect(chemAppAuthHeaders(false)).toEqual({
      "X-ChemApp-Access-Token": "access",
    });
    expect(chemAppAuthHeaders(true)).toEqual({
      "X-ChemApp-Access-Token": "access",
      "X-ChemApp-Admin-Token": "admin",
    });
    expect(chemAppAuthHeaders(false, true)).toEqual({
      "X-ChemApp-Access-Token": "access",
      "X-ChemApp-Reviewer-Token": "reviewer",
    });
    expect(chemAppAuthHeaders(true, true)).toEqual({
      "X-ChemApp-Access-Token": "access",
      "X-ChemApp-Admin-Token": "admin",
      "X-ChemApp-Reviewer-Token": "reviewer",
    });
  });

  it("sends the admin credential only to privileged endpoints", () => {
    expect(requestNeedsAdmin("delete", "/api/spectra/example-id")).toBe(true);
    expect(requestNeedsAdmin("POST", "/api/ml/train")).toBe(false);
    expect(requestNeedsAdmin("get", "/api/ml/download?format=json")).toBe(false);
    expect(requestNeedsAdmin("POST", "/api/ml/elucidate/index/import")).toBe(true);
    expect(requestNeedsAdmin("post", "/api/analyze/example-id")).toBe(false);
    expect(requestNeedsAdmin("post", "/api/ai/stream")).toBe(false);
    expect(requestNeedsAdmin("get", "/api/spectra")).toBe(false);
    expect(requestNeedsAdmin("post", "/api/reviews/queue")).toBe(true);
    expect(requestNeedsAdmin("post", "/api/reviews/s1/submit")).toBe(false);
    expect(requestNeedsAdmin("post", "/api/reviews/s1/adjudicate")).toBe(true);
    expect(requestNeedsAdmin("get", "/api/reviews/gold-manifest")).toBe(true);
  });

  it("sends reviewer credentials only to reviewer-scoped routes", () => {
    expect(requestNeedsReviewer("get", "/api/reviews/capabilities")).toBe(true);
    expect(requestNeedsReviewer("get", "/api/reviews/me")).toBe(true);
    expect(requestNeedsReviewer("get", "/api/reviews/queue")).toBe(true);
    expect(requestNeedsReviewer("get", "/api/reviews/s1")).toBe(true);
    expect(requestNeedsReviewer("post", "/api/reviews/s1/submit")).toBe(true);
    expect(requestNeedsReviewer("post", "/api/reviews/queue")).toBe(false);
    expect(
      requestNeedsReviewer("post", "/api/reviews/s1/adjudicate"),
    ).toBe(false);
    expect(
      requestNeedsReviewer("get", "/api/reviews/gold-manifest"),
    ).toBe(false);
    expect(requestNeedsReviewer("get", "/api/spectra")).toBe(false);
  });
});
