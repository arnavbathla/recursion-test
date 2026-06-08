"""API health test using FastAPI TestClient."""

from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import create_app


def test_healthz():
    app = create_app()
    with TestClient(app) as client:
        res = client.get("/healthz")
        assert res.status_code == 200
        assert res.json() == {"ok": True}
