export interface ChemAppTokens {
  accessToken: string;
  adminToken: string;
  reviewerToken: string;
}

let tokens: ChemAppTokens = {
  accessToken: "",
  adminToken: "",
  reviewerToken: "",
};

export function loadChemAppTokens(): ChemAppTokens {
  return { ...tokens };
}

export function saveChemAppTokens(next: ChemAppTokens): void {
  tokens = {
    accessToken: next.accessToken.trim(),
    adminToken: next.adminToken.trim(),
    reviewerToken: next.reviewerToken.trim(),
  };
}

export function clearChemAppTokens(): void {
  tokens = { accessToken: "", adminToken: "", reviewerToken: "" };
}

export function chemAppAuthHeaders(
  includeAdmin = false,
  includeReviewer = false,
): Record<string, string> {
  const current = loadChemAppTokens();
  return {
    ...(current.accessToken ? { "X-ChemApp-Access-Token": current.accessToken } : {}),
    ...(includeAdmin && current.adminToken ? { "X-ChemApp-Admin-Token": current.adminToken } : {}),
    ...(includeReviewer && current.reviewerToken
      ? { "X-ChemApp-Reviewer-Token": current.reviewerToken }
      : {}),
  };
}
