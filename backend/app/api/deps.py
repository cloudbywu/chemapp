from __future__ import annotations

import hmac
import os
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status

from app.api.security import (
    bearer_token,
    configured_review_admin_subject,
    configured_reviewer_tokens,
    is_local_request,
)
from app.api.store import get_store as _get_store


def get_store():
    return _get_store()


@dataclass(frozen=True)
class ReviewerPrincipal:
    subject: str


@dataclass(frozen=True)
class ReviewAdminPrincipal:
    subject: str


def _review_subjects_overlap() -> bool:
    admin_subject = configured_review_admin_subject()
    if not admin_subject:
        return False
    reviewer_subjects = {
        subject.casefold()
        for subject in configured_reviewer_tokens()
    }
    return admin_subject.casefold() in reviewer_subjects


def require_reviewer(
    x_chemapp_reviewer_token: str | None = Header(default=None),
) -> ReviewerPrincipal:
    """Bind a review write to a server-configured reviewer identity."""

    configured = configured_reviewer_tokens()
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Spectrum review is disabled until CHEMAPP_REVIEWER_TOKENS "
                "contains a valid reviewer-to-token JSON mapping"
            ),
        )
    if _review_subjects_overlap():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Spectrum review identities are disabled because the review "
                "administrator subject overlaps a reviewer subject"
            ),
        )
    supplied = x_chemapp_reviewer_token or ""
    for subject, token in configured.items():
        if hmac.compare_digest(supplied, token):
            return ReviewerPrincipal(subject=subject)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="A valid ChemApp reviewer token is required",
        headers={"WWW-Authenticate": "ChemAppReviewer"},
    )


def require_admin(
    request: Request,
    x_chemapp_admin_token: str | None = Header(default=None),
) -> None:
    """Protect destructive and resource-intensive endpoints.

    Local development remains usable without configuration. Remote callers must
    provide ``X-ChemApp-Admin-Token`` whenever ``CHEMAPP_ADMIN_TOKEN`` is set,
    and are denied by default when it is not set.
    """
    configured = os.environ.get("CHEMAPP_ADMIN_TOKEN", "")
    supplied_tokens = tuple(
        value
        for value in (
            x_chemapp_admin_token or "",
            bearer_token(request),
            request.headers.get("x-chemapp-access-token", ""),
        )
        if value
    )
    if configured:
        if not any(hmac.compare_digest(supplied, configured) for supplied in supplied_tokens):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A valid ChemApp admin token is required",
                headers={"WWW-Authenticate": "ChemAppAdmin"},
            )
        return

    local_bypass = os.environ.get("CHEMAPP_LOCAL_ADMIN_BYPASS", "1").strip().lower()
    if local_bypass in {"1", "true", "yes", "on"} and is_local_request(request):
        return

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Admin operations are disabled until CHEMAPP_ADMIN_TOKEN is configured",
    )


def require_review_admin(
    request: Request,
    x_chemapp_admin_token: str | None = Header(default=None),
) -> ReviewAdminPrincipal:
    """Authorize a review-management action and bind it to a stable subject."""

    require_admin(request, x_chemapp_admin_token)
    subject = configured_review_admin_subject()
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Review administration is disabled until "
                "CHEMAPP_REVIEW_ADMIN_SUBJECT is a stable lowercase pseudonym"
            ),
        )
    if _review_subjects_overlap():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Review administration is disabled because its subject "
                "overlaps a reviewer subject"
            ),
        )
    return ReviewAdminPrincipal(subject=subject)
