"""Access-control primitives for the ChemApp API.

Recorded trust boundaries and design intent:

* `path_is_access_exempt` exempts `/api/reviews` from the global access
gate on purpose. Every review route re-authenticates its caller with the
dedicated reviewer or review-admin dependencies (see `app.api.deps`), so
reviewer tokens are never treated as general API access tokens. This design
presumes reviewer tokens are issued out of band and remain independent of
`CHEMAPP_ACCESS_TOKEN` / `CHEMAPP_ADMIN_TOKEN`.
* `is_local_request` decides local origin from the transport peer address
only. The `Host` header is client- or proxy-controlled and is at most an
auxiliary hint; it can never grant the local fallback on its own.
"""

from __future__ import annotations

import hmac
import json
import os
import re

from fastapi import Request


LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost", "testclient"}
REVIEW_SUBJECT_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{1,63}")


def truthy_env(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def is_local_request(request: Request) -> bool:
    """Return True only when the transport peer is a loopback client.

    The peer address (`request.client.host`) is the authoritative signal of
    where a request entered this process. The `Host` header is auxiliary at
    best: a remote client can trivially send `Host: localhost`, and a
    same-host reverse proxy may rewrite the header in either direction, so it
    must never be the factor that *grants* local fallback.

    Trust boundary: a same-host reverse proxy makes its own remote clients
    indistinguishable from loopback peers. Deployments that expose such a
    proxy must disable the local fallback (`CHEMAPP_LOCAL_ACCESS_BYPASS=0`)
    and configure real tokens instead of relying on this check.
    """

    client_host = request.client.host if request.client else ""
    return client_host in LOCAL_CLIENTS


def bearer_token(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    scheme, separator, value = authorization.partition(" ")
    if separator and scheme.lower() == "bearer":
        return value.strip()
    return ""


def supplied_access_tokens(request: Request) -> tuple[str, ...]:
    return tuple(
        value
        for value in (
            request.headers.get("x-chemapp-access-token", ""),
            request.headers.get("x-chemapp-admin-token", ""),
            bearer_token(request),
        )
        if value
    )


def configured_reviewer_tokens() -> dict[str, str]:
    """Return the validated server-side reviewer identity mapping.

    The mapping is deliberately configuration-only. Reviewer identifiers sent
    in request bodies are never trusted as identities.
    """

    raw = os.environ.get("CHEMAPP_REVIEWER_TOKENS", "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict):
        return {}

    tokens: dict[str, str] = {}
    seen_secrets: set[str] = set()
    reserved_secrets = {
        secret
        for secret in (
            os.environ.get("CHEMAPP_ACCESS_TOKEN", ""),
            os.environ.get("CHEMAPP_ADMIN_TOKEN", ""),
        )
        if secret
    }
    for subject, token in value.items():
        if (
            isinstance(subject, str)
            and REVIEW_SUBJECT_PATTERN.fullmatch(subject)
            and isinstance(token, str)
            and len(token) >= 16
            and token not in seen_secrets
            and token not in reserved_secrets
        ):
            tokens[subject] = token
            seen_secrets.add(token)
    return tokens


def configured_review_admin_subject() -> str:
    subject = os.environ.get("CHEMAPP_REVIEW_ADMIN_SUBJECT", "").strip()
    if REVIEW_SUBJECT_PATTERN.fullmatch(subject):
        return subject
    return ""


def access_allowed(request: Request) -> tuple[bool, str]:
    configured = os.environ.get("CHEMAPP_ACCESS_TOKEN", "")
    admin = os.environ.get("CHEMAPP_ADMIN_TOKEN", "")
    if configured or admin:
        valid_tokens = tuple(token for token in (configured, admin) if token)
        for supplied in supplied_access_tokens(request):
            if any(hmac.compare_digest(supplied, valid) for valid in valid_tokens):
                return True, ""
        return False, "A valid ChemApp access token is required"

    if truthy_env("CHEMAPP_LOCAL_ACCESS_BYPASS", "1") and is_local_request(request):
        return True, ""
    return False, "API access is disabled until CHEMAPP_ACCESS_TOKEN is configured"


def path_is_access_exempt(request: Request) -> bool:
    if request.method == "OPTIONS":
        return True
    if request.url.path == "/api/live":
        return True
    if request.url.path == "/api/ready" and not truthy_env("CHEMAPP_PROTECT_READY", "0"):
        return True
    # Design intent (do not remove): /api/reviews is exempt from the global
    # access gate because every review route authenticates its caller
    # independently via the reviewer or review-admin dependencies. Reviewer
    # credentials are deliberately NOT accepted as general API access tokens,
    # and exempting the prefix preserves the route-specific WWW-Authenticate
    # challenges. Prerequisite: reviewer tokens are issued out of band and
    # authenticated independently of CHEMAPP_ACCESS_TOKEN /
    # CHEMAPP_ADMIN_TOKEN.
    if request.url.path.startswith("/api/reviews"):
        return True
    return False
