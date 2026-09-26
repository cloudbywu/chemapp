FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

LABEL org.opencontainers.image.title="ChemApp DP5q overlap audit"
LABEL org.opencontainers.image.description="Minimal one-shot pickle-to-JSON isolation environment"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m pip install --no-cache-dir \
    pandas==2.2.3 \
    rdkit==2026.3.4

WORKDIR /audit
