from starlette.testclient import TestClient

from sdb.ui.app import app
import sdb.ui.app as app_module
from starlette.websockets import WebSocketDisconnect
import time
import pytest


def test_websocket_chat():
    client = TestClient(app)
    res = client.post(
        "/login",
        json={"username": "physician", "password": "secret"},
    )
    token = res.json()["token"]
    with client.websocket_connect(f"/ws?token={token}") as ws:
        ws.send_json({"action": "question", "content": "cough"})
        parts = []
        while True:
            data = ws.receive_json()
            parts.append(data["reply"])
            if data["done"]:
                assert data["total_spent"] == 0.0
                assert data["ordered_tests"] == []
                break
        assert len(parts) > 1

        ws.send_json({"action": "test", "content": "complete blood count"})
        parts = []
        while True:
            data = ws.receive_json()
            parts.append(data["reply"])
            if data["done"]:
                assert data["cost"] == 10.0
                assert data["total_spent"] == 10.0
                assert data["ordered_tests"] == ["complete blood count"]
                break
        assert len(parts) > 1


def test_index_layout():
    client = TestClient(app)
    res = client.get("/")
    html = res.text
    assert "summary-panel" in html
    assert "tests-panel" in html
    assert "flow-panel" in html
    assert "grid" in html  # check styling


def test_case_summary():
    """Case summary endpoint returns the demo case info."""
    client = TestClient(app)
    res = client.get("/case")
    assert res.status_code == 200
    assert res.json() == {"summary": "A 30 year old with cough"}


def test_token_expiry(monkeypatch):
    """Expired tokens are rejected for websocket connections."""

    client = TestClient(app)
    monkeypatch.setattr(app_module, "TOKEN_TIMEOUT", 0.1)
    res = client.post(
        "/login",
        json={"username": "physician", "password": "secret"},
    )
    token = res.json()["token"]
    time.sleep(0.2)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws?token={token}"):
            pass


def test_token_purged_on_login(monkeypatch):
    """Login removes expired tokens from the store."""

    client = TestClient(app)
    monkeypatch.setattr(app_module, "TOKEN_TIMEOUT", 0.1)
    res = client.post(
        "/login",
        json={"username": "physician", "password": "secret"},
    )
    token1 = res.json()["token"]
    time.sleep(0.2)
    res = client.post(
        "/login",
        json={"username": "physician", "password": "secret"},
    )
    token2 = res.json()["token"]
    assert token1 != token2
    assert len(app_module.TOKENS) == 1
