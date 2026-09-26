from __future__ import annotations

import json
import hmac
import os
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status as http_status,
)
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.api.deps import (
    ReviewAdminPrincipal,
    ReviewerPrincipal,
    get_store,
    require_review_admin,
    require_reviewer,
)
from app.api.security import (
    bearer_token,
    configured_review_admin_subject,
    configured_reviewer_tokens,
    is_local_request,
)
from app.review.spectrum_review import (
    QUEUE_STATES,
    ReviewConflict,
    ReviewRepository,
    ReviewStateError,
)


router = APIRouter(prefix="/api/reviews", tags=["spectrum-review"])
SpectrumId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$"),
]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=512)]
ReviewCheck = Literal["pass", "fail", "uncertain"]


class EnqueueReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spectrum_id: SpectrumId
    structure_smiles: Annotated[
        str,
        StringConstraints(min_length=1, max_length=4096),
    ]
    structure_source: ShortText
    molecule_id: ShortText
    source_collection: ShortText
    source_record_id: ShortText
    independence_group: ShortText
    license_id: ShortText
    provenance_uri: Annotated[
        str,
        StringConstraints(min_length=9, max_length=2048),
    ]
    rights_confirmed: bool
    expected_spectrum_revision: int = Field(ge=1)
    expected_result_revision: int = Field(ge=0)


class ReviewChecks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure: ReviewCheck
    nucleus: ReviewCheck
    axis: ReviewCheck
    peaks: ReviewCheck
    solvent: ReviewCheck


class ReviewObservations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure_smiles: Annotated[str, StringConstraints(max_length=4096)] = ""
    nucleus: Annotated[str, StringConstraints(max_length=64)] = ""
    axis_unit: Annotated[str, StringConstraints(max_length=64)] = ""
    axis_direction: Literal[
        "ascending",
        "descending",
        "non_monotonic",
        "unknown",
    ] = "unknown"
    solvent: Annotated[str, StringConstraints(max_length=256)] = ""
    peak_count: int = Field(default=0, ge=0, le=100000)


class SubmitReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["accept", "reject"]
    checks: ReviewChecks
    observations: ReviewObservations
    notes: Annotated[str, StringConstraints(max_length=4000)] = ""
    expected_queue_revision: int = Field(ge=1)
    expected_snapshot_sha256: Annotated[
        str,
        StringConstraints(pattern=r"^[a-f0-9]{64}$"),
    ]


class AdjudicateReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept", "reject"]
    checks: ReviewChecks
    reason: Annotated[str, StringConstraints(min_length=3, max_length=4000)]
    expected_queue_revision: int = Field(ge=1)
    expected_snapshot_sha256: Annotated[
        str,
        StringConstraints(pattern=r"^[a-f0-9]{64}$"),
    ]


def _repository() -> ReviewRepository:
    return ReviewRepository(get_store())


def _translate_error(error: Exception) -> HTTPException:
    if isinstance(error, ReviewConflict):
        return HTTPException(
            409,
            {
                "code": "review_revision_conflict",
                "message": str(error),
                "current_revision": error.current_revision,
            },
        )
    if isinstance(error, ReviewStateError):
        return HTTPException(
            409,
            {"code": "review_state_error", "message": str(error)},
        )
    if isinstance(error, KeyError):
        return HTTPException(404, str(error).strip("'"))
    return HTTPException(500, "Spectrum review operation failed")


@router.get("/capabilities")
def review_capabilities(
    request: Request,
    x_chemapp_reviewer_token: str | None = Header(default=None),
    x_chemapp_admin_token: str | None = Header(default=None),
):
    reviewer_tokens = configured_reviewer_tokens()
    admin_subject = configured_review_admin_subject()
    subjects_overlap = bool(
        admin_subject
        and admin_subject.casefold()
        in {subject.casefold() for subject in reviewer_tokens}
    )

    reviewer_subject = ""
    if x_chemapp_reviewer_token:
        for subject, token in reviewer_tokens.items():
            if hmac.compare_digest(x_chemapp_reviewer_token, token):
                reviewer_subject = subject
                break
        if not reviewer_subject:
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="A valid ChemApp reviewer token is required",
                headers={"WWW-Authenticate": "ChemAppReviewer"},
            )

    configured_admin_token = os.environ.get("CHEMAPP_ADMIN_TOKEN", "")
    admin_supplied = tuple(
        value
        for value in (
            x_chemapp_admin_token or "",
            bearer_token(request),
            request.headers.get("x-chemapp-access-token", ""),
        )
        if value
    )
    admin_authenticated = False
    if configured_admin_token:
        admin_authenticated = any(
            hmac.compare_digest(value, configured_admin_token)
            for value in admin_supplied
        )
        if x_chemapp_admin_token and not admin_authenticated:
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="A valid ChemApp admin token is required",
                headers={"WWW-Authenticate": "ChemAppAdmin"},
            )
    else:
        local_bypass = os.environ.get(
            "CHEMAPP_LOCAL_ADMIN_BYPASS",
            "1",
        ).strip().lower()
        admin_authenticated = (
            local_bypass in {"1", "true", "yes", "on"}
            and is_local_request(request)
        )

    reviewer_ready = bool(reviewer_tokens) and not subjects_overlap
    admin_ready = bool(admin_subject) and not subjects_overlap
    can_admin = admin_ready and admin_authenticated
    return {
        "reviewer": {
            "configured": reviewer_ready,
            "authenticated": bool(reviewer_subject) and reviewer_ready,
            "subject": reviewer_subject if reviewer_ready else "",
        },
        "admin": {
            "configured": admin_ready,
            "authenticated": can_admin,
            "subject": admin_subject if can_admin else "",
        },
        "separation_ok": not subjects_overlap,
        "can_review": bool(reviewer_subject) and reviewer_ready,
        "can_manage_queue": can_admin,
        "can_view_audit": can_admin,
        "can_adjudicate": can_admin,
        "can_export_gold": can_admin,
    }


@router.get("/me")
def review_identity(
    principal: ReviewerPrincipal = Depends(require_reviewer),
):
    return {"reviewer_id": principal.subject}


@router.get("/queue")
def list_review_queue(
    status: Annotated[str | None, Query()] = None,
    include_history: bool = False,
    principal: ReviewerPrincipal = Depends(require_reviewer),
):
    if status is not None and status not in QUEUE_STATES:
        raise HTTPException(422, "Unsupported review status")
    try:
        items = _repository().list_queue(
            status=status,
            reviewer_id=principal.subject,
            include_history=include_history,
        )
    except Exception as error:
        raise _translate_error(error) from error
    return {
        "reviewer_id": principal.subject,
        "items": items,
        "count": len(items),
    }


@router.post("/queue")
def enqueue_review(
    payload: EnqueueReviewRequest,
    principal: ReviewAdminPrincipal = Depends(require_review_admin),
):
    try:
        return _repository().enqueue(
            **payload.model_dump(),
            actor_id=principal.subject,
        )
    except Exception as error:
        raise _translate_error(error) from error


@router.get("/admin/queue")
def list_review_queue_as_admin(
    status: Annotated[str | None, Query()] = None,
    include_history: bool = False,
    _principal: ReviewAdminPrincipal = Depends(require_review_admin),
):
    if status is not None and status not in QUEUE_STATES:
        raise HTTPException(422, "Unsupported review status")
    try:
        items = _repository().list_queue(
            status=status,
            include_history=include_history,
        )
    except Exception as error:
        raise _translate_error(error) from error
    return {"items": items, "count": len(items)}


@router.get("/gold-manifest")
def export_gold_manifest(
    _principal: ReviewAdminPrincipal = Depends(require_review_admin),
):
    try:
        manifest = _repository().gold_manifest()
    except Exception as error:
        raise _translate_error(error) from error
    content = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return Response(
        content=content,
        media_type="application/json",
        headers={
            "Content-Disposition": (
                "attachment; filename=chemapp-nmr-gold-manifest-v1.json"
            ),
            "X-Content-SHA256": manifest["manifest_sha256"],
        },
    )


@router.get("/{spectrum_id}")
def get_review(
    spectrum_id: SpectrumId,
    principal: ReviewerPrincipal = Depends(require_reviewer),
):
    try:
        item = _repository().get_latest(
            spectrum_id,
            reviewer_id=principal.subject,
        )
    except Exception as error:
        raise _translate_error(error) from error
    if item is None:
        raise HTTPException(404, f"Spectrum {spectrum_id} is not queued")
    return {"reviewer_id": principal.subject, "item": item}


@router.get("/{spectrum_id}/audit")
def get_review_audit(
    spectrum_id: SpectrumId,
    _principal: ReviewAdminPrincipal = Depends(require_review_admin),
):
    try:
        audit = _repository().audit(spectrum_id)
    except Exception as error:
        raise _translate_error(error) from error
    if audit is None:
        raise HTTPException(404, f"Spectrum {spectrum_id} is not queued")
    return audit


@router.post("/{spectrum_id}/submit")
def submit_review(
    spectrum_id: SpectrumId,
    payload: SubmitReviewRequest,
    principal: ReviewerPrincipal = Depends(require_reviewer),
):
    try:
        item = _repository().submit(
            spectrum_id=spectrum_id,
            reviewer_id=principal.subject,
            verdict=payload.verdict,
            checks=payload.checks.model_dump(),
            observations=payload.observations.model_dump(),
            notes=payload.notes,
            expected_queue_revision=payload.expected_queue_revision,
            expected_snapshot_sha256=payload.expected_snapshot_sha256,
        )
    except Exception as error:
        raise _translate_error(error) from error
    return {"reviewer_id": principal.subject, "item": item}


@router.post("/{spectrum_id}/adjudicate")
def adjudicate_review(
    spectrum_id: SpectrumId,
    payload: AdjudicateReviewRequest,
    principal: ReviewAdminPrincipal = Depends(require_review_admin),
):
    try:
        return _repository().adjudicate(
            spectrum_id=spectrum_id,
            decision=payload.decision,
            checks=payload.checks.model_dump(),
            reason=payload.reason,
            expected_queue_revision=payload.expected_queue_revision,
            expected_snapshot_sha256=payload.expected_snapshot_sha256,
            actor_id=principal.subject,
        )
    except Exception as error:
        raise _translate_error(error) from error
