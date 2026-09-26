"""Verify the legacy ML endpoints are removed and broken modules fail closed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/ml/predict/anything"),
        ("post", "/api/ml/predict/dual/anything"),
        ("post", "/api/ml/train"),
        ("get", "/api/ml/download"),
    ],
)
def test_legacy_ml_endpoints_are_gone(client: TestClient, method: str, path: str) -> None:
    response = getattr(client, method)(path)
    assert response.status_code == 410
    assert "removed" in response.json()["detail"]


def test_hose_verify_fails_closed() -> None:
    import app.ml.hose_verify as hose_verify

    with pytest.raises(NotImplementedError):
        getattr(hose_verify, "build_enriched_database")


def test_retrieval_requires_real_smiles() -> None:
    from types import SimpleNamespace

    from app.ml.retrieval import canonical_smiles_from_shard

    assert canonical_smiles_from_shard(SimpleNamespace(smiles="CCO")) == "CCO"
    with pytest.raises(NotImplementedError):
        canonical_smiles_from_shard(SimpleNamespace())
    with pytest.raises(ValueError):
        canonical_smiles_from_shard(SimpleNamespace(smiles="not-a-smiles"))
