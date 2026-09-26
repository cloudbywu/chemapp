FROM ghcr.io/astral-sh/uv:0.11.7 AS uv

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    CHEMAPP_LOCAL_ADMIN_BYPASS=0 \
    CHEMAPP_DB_PATH=/var/lib/chemapp/chemapp.db \
    CHEMAPP_NMR_INDEX=/var/lib/chemapp-models/nmr_spectral_index.sqlite \
    CHEMAPP_NMR_INDEX_V2=/var/lib/chemapp-models/nmr_spectral_index_v2.sqlite \
    CHEMAPP_NMR_RANKER=/var/lib/chemapp-models/nmr_joint_ranker.joblib \
    CHEMAPP_T5_MODEL_DIR=/var/lib/chemapp-models/t5_nmr \
    CHEMAPP_CSP5_MODE=on

WORKDIR /app
COPY --from=uv /uv /uvx /bin/
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --locked --no-dev

COPY backend/app ./app
COPY backend/vendor ./vendor
COPY backend/scripts/fetch_csp5_weights.py ./scripts/fetch_csp5_weights.py
RUN python scripts/fetch_csp5_weights.py --target /app/vendor/csp5
RUN addgroup --system chemapp \
    && adduser --system --ingroup chemapp chemapp \
    && mkdir -p /var/lib/chemapp /var/lib/chemapp-models \
    && chown -R chemapp:chemapp /app /var/lib/chemapp /var/lib/chemapp-models

USER chemapp
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ready', timeout=3)"]
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
