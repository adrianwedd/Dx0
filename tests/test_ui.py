import pytest
import httpx
import threading
import uvicorn
import asyncio
import time
from httpx_ws import aconnect_ws, WebSocketUpgradeError, WebSocketDisconnect
from starlette.testclient import TestClient

import sdb.ui.app as ui_app

app = ui_app.app
SessionDB = ui_app.SessionDB
SESSION_DB = ui_app.SESSION_DB


@pytest.mark.asyncio
async def test_websocket_chat():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        await asyncio.sleep(0.01)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        res = await client.post(
            "/api/v1/login",
            json={"username": "physician", "password": "secret"},
        )
        assert res.status_code == 200
        token = res.json()["token"]
        async with aconnect_ws(
            f"ws://127.0.0.1:8000/api/v1/ws?token={token}", client
        ) as ws:
            await ws.send_json({"action": "question", "content": "cough"})
            parts = []
            while True:
                data = await ws.receive_json()
                parts.append(data["reply"])
                if data["done"]:
                    assert data["total_spent"] == 0.0
                    assert data["ordered_tests"] == []
                    break
            assert len(parts) > 1

            await ws.send_json({"action": "test", "content": "complete blood count"})
            parts = []
            while True:
                data = await ws.receive_json()
                parts.append(data["reply"])
                if data["done"]:
                    assert data["cost"] == 10.0
                    assert data["total_spent"] == 10.0
                    assert data["ordered_tests"] == ["complete blood count"]
                    break
            assert len(parts) > 1

    server.should_exit = True
    thread.join()


def test_index_layout():
    client = TestClient(app)
    res = client.get("/api/v1")
    html = res.text
    assert "summary-panel" in html
    assert "tests-panel" in html
    assert "flow-panel" in html
    assert "grid" in html  # check styling


def test_case_summary():
    """Case summary endpoint returns the demo case info."""
    client = TestClient(app)
    res = client.get("/api/v1/case")
    assert res.status_code == 200
    assert res.json() == {"summary": "A 30 year old with cough"}


def test_unknown_version():
    """Requests to an unsupported API version return 404."""
    client = TestClient(app)
    res = client.get("/api/v2/case")
    assert res.status_code == 404


def test_login_success():
    client = TestClient(app)
    res = client.post(
        "/api/v1/login",
        json={"username": "physician", "password": "secret"},
    )
    assert res.status_code == 200
    assert "token" in res.json()


def test_login_failure():
    client = TestClient(app)
    res = client.post(
        "/api/v1/login",
        json={"username": "physician", "password": "wrong"},
    )
    assert res.status_code == 401


def test_login_missing_field():
    """Missing fields should trigger validation error."""
    client = TestClient(app)
    res = client.post("/api/v1/login", json={"username": "physician"})
    assert res.status_code == 422


def test_login_lockout(monkeypatch):
    monkeypatch.setattr(ui_app, "FAILED_LOGIN_LIMIT", 2)
    monkeypatch.setattr(ui_app, "FAILED_LOGIN_COOLDOWN", 1)
    ui_app.FAILED_LOGINS.clear()
    client = TestClient(app)
    for _ in range(2):
        res = client.post(
            "/api/v1/login",
            json={"username": "physician", "password": "wrong"},
        )
        assert res.status_code == 401

    res = client.post(
        "/api/v1/login",
        json={"username": "physician", "password": "wrong"},
    )
    assert res.status_code == 429

    time.sleep(1.1)
    res = client.post(
        "/api/v1/login",
        json={"username": "physician", "password": "secret"},
    )
    assert res.status_code == 200


@pytest.mark.asyncio
async def test_ws_requires_token():
    config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        await asyncio.sleep(0.01)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8001") as client:
        with pytest.raises(WebSocketUpgradeError):
            async with aconnect_ws("ws://127.0.0.1:8001/api/v1/ws", client):
                pass

    server.should_exit = True
    thread.join()


@pytest.mark.asyncio
async def test_ws_schema_validation():
    """Invalid websocket payload closes the connection."""
    config = uvicorn.Config(app, host="127.0.0.1", port=8002, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        await asyncio.sleep(0.01)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8002") as client:
        res = await client.post(
            "/api/v1/login",
            json={"username": "physician", "password": "secret"},
        )
        token = res.json()["token"]
        async with aconnect_ws(
            f"ws://127.0.0.1:8002/api/v1/ws?token={token}", client
        ) as ws:
            await ws.send_json({"bad": "data"})
            with pytest.raises(WebSocketDisconnect):
                await ws.receive_json()

    server.should_exit = True
    thread.join()


@pytest.mark.asyncio
async def test_ws_missing_field():
    """WebSocket closes when required fields are missing."""
    config = uvicorn.Config(app, host="127.0.0.1", port=8003, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        await asyncio.sleep(0.01)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8003") as client:
        res = await client.post(
            "/api/v1/login",
            json={"username": "physician", "password": "secret"},
        )
        token = res.json()["token"]
        async with aconnect_ws(
            f"ws://127.0.0.1:8003/api/v1/ws?token={token}", client
        ) as ws:
            await ws.send_json({"action": "question"})
            with pytest.raises(WebSocketDisconnect):
                await ws.receive_json()

    server.should_exit = True
    thread.join()


@pytest.mark.asyncio
async def test_ws_invalid_action():
    """WebSocket closes when an unknown action is provided."""
    config = uvicorn.Config(app, host="127.0.0.1", port=8004, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        await asyncio.sleep(0.01)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8004") as client:
        res = await client.post(
            "/api/v1/login",
            json={"username": "physician", "password": "secret"},
        )
        token = res.json()["token"]
        async with aconnect_ws(
            f"ws://127.0.0.1:8004/api/v1/ws?token={token}", client
        ) as ws:
            await ws.send_json({"action": "invalid", "content": "hi"})
            with pytest.raises(WebSocketDisconnect):
                await ws.receive_json()

    server.should_exit = True
    thread.join()


@pytest.mark.asyncio
async def test_token_persistence(tmp_path):
    path = tmp_path / "sessions.db"
    ui_app.SESSION_DB.path = str(path)
    ui_app.SESSION_DB._init_db()
    ui_app.SESSION_DB.ttl = 10

    config = uvicorn.Config(app, host="127.0.0.1", port=8010, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        await asyncio.sleep(0.01)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8010") as client:
        res = await client.post(
            "/api/v1/login",
            json={"username": "physician", "password": "secret"},
        )
        token = res.json()["token"]

    server.should_exit = True
    thread.join()

    ui_app.SESSION_DB = SessionDB(str(path), ttl=10)
    # restart server
    config = uvicorn.Config(app, host="127.0.0.1", port=8011, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        await asyncio.sleep(0.01)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8011") as client:
        async with aconnect_ws(
            f"ws://127.0.0.1:8011/api/v1/ws?token={token}", client
        ) as ws:
            await ws.send_json({"action": "question", "content": "hi"})
            while True:
                data = await ws.receive_json()
                if data["done"]:
                    break

    server.should_exit = True
    thread.join()


@pytest.mark.asyncio
async def test_token_cleanup(tmp_path):
    path = tmp_path / "sessions.db"
    store = SessionDB(str(path), ttl=1)
    store.add("tok", "user")
    await asyncio.sleep(1.2)
    store.cleanup()
    assert store.get("tok") is None


def test_fhir_transcript_endpoint():
    client = TestClient(app)
    data = [["panel", "hi"], ["gatekeeper", "hello"]]
    res = client.post(
        "/api/v1/fhir/transcript",
        json={"transcript": data, "patient_id": "p1"},
    )
    assert res.status_code == 200
    bundle = res.json()
    assert bundle["resourceType"] == "Bundle"
    assert bundle["entry"][0]["resource"]["sender"]["display"] == "panel"


def test_fhir_tests_endpoint():
    client = TestClient(app)
    res = client.post(
        "/api/v1/fhir/tests",
        json={"tests": ["cbc"], "patient_id": "p2"},
    )
    assert res.status_code == 200
    bundle = res.json()
    assert bundle["entry"][0]["resource"]["code"]["text"] == "cbc"
