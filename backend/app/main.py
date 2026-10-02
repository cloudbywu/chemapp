from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from app.paths import nmr_index_v2_path

from app.api.routes.upload import router as upload_router
from app.api.routes.spectra import router as spectra_router
from app.api.routes.analysis import router as analysis_router
from app.api.routes.inference import router as inference_router
from app.api.routes.ai import router as ai_router
from app.api.routes.ml import router as ml_router
from app.api.routes.model_weights import router as model_weights_router
from app.api.routes.elucidate import router as elucidate_router
from app.api.routes.reports import router as reports_router
from app.api.routes.nmr import router as nmr_router
from app.api.routes.standards import router as standards_router
from app.api.routes.experiment_agent import router as experiment_agent_router
from app.api.routes.reviews import router as reviews_router
from app.api.security import access_allowed, path_is_access_exempt

logger = logging.getLogger("chemapp.main")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    from app.ml.deployment_check import verify_deployment

    check = verify_deployment()
    logger.info("deployment model check: %s", check)
    yield


app = FastAPI(title="ChemApp API", version="0.2.0", lifespan=lifespan)

_origins = os.environ.get("CHEMAPP_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
# Fail-closed CORS: only the HTTP methods this API actually serves, and only
# the request headers used by the backend auth scheme and its frontend (see
# app.api.security, app.api.deps, and frontend/src/services/authTokens.py).
# Wildcards are intentionally not used: credentialed cross-origin access must
# stay enumerable and reviewable.
_CORS_ALLOWED_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
_CORS_ALLOWED_HEADERS = [
    "Authorization",
    "Content-Type",
    "X-Request-ID",
    "X-ChemApp-Access-Token",
    "X-ChemApp-Admin-Token",
    "X-ChemApp-Reviewer-Token",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=_CORS_ALLOWED_METHODS,
    allow_headers=_CORS_ALLOWED_HEADERS,
)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)

_trusted_hosts = [
    host.strip()
    for host in os.environ.get("CHEMAPP_TRUSTED_HOSTS", "").split(",")
    if host.strip()
]
if _trusted_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_trusted_hosts)


# Baseline security response headers applied to every API response,
# including the early access-gate rejection produced below.
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


@app.middleware("http")
async def request_metadata(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    request_id = (
        supplied
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", supplied)
        else uuid.uuid4().hex
    )
    request.state.request_id = request_id
    if request.url.path.startswith("/api/") and not path_is_access_exempt(request):
        allowed, detail = access_allowed(request)
        if not allowed:
            status_code = (
                401
                if os.environ.get("CHEMAPP_ACCESS_TOKEN") or os.environ.get("CHEMAPP_ADMIN_TOKEN")
                else 503
            )
            return JSONResponse(
                status_code=status_code,
                content={"detail": detail, "request_id": request_id},
                headers={
                    "WWW-Authenticate": "Bearer",
                    "Cache-Control": "no-store",
                    "X-Request-ID": request_id,
                    **_SECURITY_HEADERS,
                },
            )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    for header_name, header_value in _SECURITY_HEADERS.items():
        response.headers[header_name] = header_value
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault(
            "Vary",
            "Authorization, X-ChemApp-Access-Token, X-ChemApp-Admin-Token",
        )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fail closed on uncaught errors: a sanitized 500 with the request id.

    The original exception text is logged server-side together with the
    request id and is never reflected to the client.
    """

    request_id = getattr(request.state, "request_id", "") or uuid.uuid4().hex
    logger.exception(
        "unhandled error path=%s request_id=%s",
        request.url.path,
        request_id,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
        headers={"X-Request-ID": request_id, **_SECURITY_HEADERS},
    )

app.include_router(upload_router)
app.include_router(spectra_router)
app.include_router(analysis_router)
app.include_router(inference_router)
app.include_router(ai_router)
app.include_router(ml_router)
app.include_router(model_weights_router)
app.include_router(elucidate_router)
app.include_router(reports_router)
app.include_router(nmr_router)
app.include_router(standards_router)
app.include_router(experiment_agent_router)
app.include_router(reviews_router)


@app.get("/api/live")
def live():
    return {"status": "ok"}


@app.get("/api/ready")
def ready():
    from app.api.deps import get_store
    store = get_store()
    return {"status": "ok", "spectra_count": store.count()}


@app.get("/api/health")
def health():
    """Backward-compatible readiness endpoint."""

    base = ready()
    assets: dict[str, Any] = {}
    try:
        from app.ml.deployment_check import verify_csp5_weights

        assets["csp5_weights"] = verify_csp5_weights()
    except Exception as exc:
        assets["csp5_weights"] = {
            "status": "error",
            "reason": str(exc),
        }
    try:
        from app.ml.calibration.applicability_v1 import load_signature

        signature = load_signature()
        assets["calibration_applicability"] = {
            "schema_version": signature.get("schema_version"),
            "status": signature.get("status"),
        }
    except Exception as exc:
        assets["calibration_applicability"] = {
            "status": "error",
            "reason": str(exc),
        }
    try:
        from app.ml.calibration.calibrator import policy

        gate = policy()
        assets["calibration_policy"] = {
            "probability_claim_allowed": gate.get(
                "probability_claim_allowed", False
            ),
            "external_holder_pending": gate.get(
                "external_holder_pending", True
            ),
        }
    except Exception as exc:
        assets["calibration_policy"] = {
            "status": "error",
            "reason": str(exc),
        }
    index_v2 = nmr_index_v2_path()
    assets["nmr_index_v2"] = {
        "path": str(index_v2),
        "exists": index_v2.is_file(),
    }
    return {**base, "assets": assets}
