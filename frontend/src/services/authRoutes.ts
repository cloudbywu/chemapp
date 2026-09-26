/** Centralized endpoint -> credential classification for the API client. */

interface RouteRule {
  methods: string[];
  regex: RegExp;
}

const ADMIN_EXACT_GET = new Set([
  "/api/reviews/capabilities",
  "/api/reviews/admin/queue",
]);

const ADMIN_EXACT_POST = new Set([
  "/api/ml/elucidate/index/import",
  "/api/ml/elucidate/ranker/train",
  "/api/spectra/examples/load",
  "/api/standards",
  "/api/reviews/queue",
]);

const ADMIN_RULES: RouteRule[] = [
  { methods: ["delete"], regex: /^\/api\/spectra\/[^/]+$/ },
  { methods: ["get"], regex: /^\/api\/reviews\/gold-manifest$/ },
  { methods: ["get"], regex: /^\/api\/reviews\/[^/]+\/audit$/ },
  { methods: ["post"], regex: /^\/api\/reviews\/[^/]+\/adjudicate$/ },
];

const REVIEWER_RULES: RouteRule[] = [
  { methods: ["get"], regex: /^\/api\/reviews\/me$/ },
  { methods: ["get"], regex: /^\/api\/reviews\/queue$/ },
  { methods: ["get"], regex: /^\/api\/reviews\/[^/]+$/ },
  { methods: ["post"], regex: /^\/api\/reviews\/[^/]+\/submit$/ },
];

function pathOnly(requestUrl: string): string {
  return requestUrl.split(/[?#]/, 1)[0];
}

function matchesAny(rules: RouteRule[], method: string, path: string): boolean {
  const normalized = method.toLowerCase();
  return rules.some(
    (rule) =>
      rule.methods.includes(normalized) && rule.regex.test(path),
  );
}

export function needsAdmin(method: string, requestUrl: string): boolean {
  const path = pathOnly(requestUrl);
  const normalized = method.toLowerCase();
  if (
    (normalized === "get" && ADMIN_EXACT_GET.has(path))
    || (normalized === "post" && ADMIN_EXACT_POST.has(path))
  ) {
    return true;
  }
  return matchesAny(ADMIN_RULES, normalized, path);
}

export function needsReviewer(method: string, requestUrl: string): boolean {
  const path = pathOnly(requestUrl);
  if (path === "/api/reviews/capabilities") return true;
  if (!path.startsWith("/api/reviews/")) return false;
  if (needsAdmin(method, path)) return false;
  return matchesAny(REVIEWER_RULES, method, path);
}
