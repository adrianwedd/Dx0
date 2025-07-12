import pytest
import httpx
import threading
import uvicorn
import asyncio
import time
from httpx_ws import aconnect_ws, WebSocketUpgradeError
from starlette.testclient import TestClient
from pathlib import Path

import sdb.ui.app as ui_app

app = ui_app.app
SessionStore = ui_app.SessionStore
SESSION_STORE = ui_app.SESSION_STORE


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
        access = res.json()["access_token"]
        async with aconnect_ws(
            f"ws://127.0.0.1:8000/api/v1/ws?token={access}", client
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


def test_refresh_token():
    client = TestClient(app)
    res = client.post(
        "/api/v1/login",
        json={"username": "physician", "password": "secret"},
    )
    refresh = res.json()["refresh_token"]
    res2 = client.post("/api/v1/refresh", json={"refresh_token": refresh})
    assert res2.status_code == 200
    data = res2.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["refresh_token"] != refresh
    assert "access_token" in res.json()
    assert "refresh_token" in res.json()


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
        token = res.json()["access_token"]
        async with aconnect_ws(
            f"ws://127.0.0.1:8002/api/v1/ws?token={token}", client
        ) as ws:
            await ws.send_json({"bad": "data"})
            data = await ws.receive_json()
            assert "error" in data

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
        token = res.json()["access_token"]
        async with aconnect_ws(
            f"ws://127.0.0.1:8003/api/v1/ws?token={token}", client
        ) as ws:
            await ws.send_json({"action": "question"})
            data = await ws.receive_json()
            assert "error" in data

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
        token = res.json()["access_token"]
        async with aconnect_ws(
            f"ws://127.0.0.1:8004/api/v1/ws?token={token}", client
        ) as ws:
            await ws.send_json({"action": "invalid", "content": "hi"})
            data = await ws.receive_json()
            assert "error" in data

    server.should_exit = True
    thread.join()


@pytest.mark.asyncio
async def test_token_persistence(tmp_path):
    path = tmp_path / "sessions.db"
    ui_app.SESSION_STORE.path = str(path)
    ui_app.SESSION_STORE.migrate()
    ui_app.SESSION_STORE.ttl = 10

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
        token = res.json()["access_token"]

    server.should_exit = True
    thread.join()

    ui_app.SESSION_STORE = SessionStore(str(path), ttl=10)
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
async def test_budget_persistence(tmp_path):
    path = tmp_path / "sessions.db"
    ui_app.SESSION_STORE.path = str(path)
    ui_app.SESSION_STORE.migrate()
    ui_app.SESSION_STORE.ttl = 10

    config = uvicorn.Config(app, host="127.0.0.1", port=8012, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        await asyncio.sleep(0.01)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8012") as client:
        res = await client.post(
            "/api/v1/login",
            json={"username": "physician", "password": "secret"},
        )
        token = res.json()["access_token"]
        async with aconnect_ws(
            f"ws://127.0.0.1:8012/api/v1/ws?token={token}", client
        ) as ws:
            await ws.send_json({"action": "test", "content": "complete blood count"})
            while True:
                data = await ws.receive_json()
                if data["done"]:
                    break

    server.should_exit = True
    thread.join()

    ui_app.SESSION_STORE = SessionStore(str(path), ttl=10)
    config = uvicorn.Config(app, host="127.0.0.1", port=8013, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        await asyncio.sleep(0.01)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8013") as client:
        async with aconnect_ws(
            f"ws://127.0.0.1:8013/api/v1/ws?token={token}", client
        ) as ws:
            await ws.send_json({"action": "question", "content": "hi"})
            total = None
            while True:
                data = await ws.receive_json()
                if data["done"]:
                    total = data["total_spent"]
                    break
            assert total == 10.0

    server.should_exit = True
    thread.join()


@pytest.mark.asyncio
async def test_token_cleanup(tmp_path):
    path = tmp_path / "sessions.db"
    store = SessionStore(str(path), ttl=1)
    store.add("tok", "user", "refresh")
    await asyncio.sleep(1.2)
    store.cleanup()
    assert store.get("tok") is None


def test_fhir_transcript_endpoint():
    client = TestClient(app)
    token = client.post(
        "/api/v1/login",
        json={"username": "physician", "password": "secret"},
    ).json()["access_token"]
    data = [["panel", "hi"], ["gatekeeper", "hello"]]
    res = client.post(
        "/api/v1/fhir/transcript",
        headers={"Authorization": f"Bearer {token}"},
        json={"transcript": data, "patient_id": "p1"},
    )
    assert res.status_code == 200
    bundle = res.json()
    assert bundle["resourceType"] == "Bundle"
    assert bundle["entry"][0]["resource"]["sender"]["display"] == "panel"


def test_fhir_tests_endpoint():
    client = TestClient(app)
    token = client.post(
        "/api/v1/login",
        json={"username": "physician", "password": "secret"},
    ).json()["access_token"]
    res = client.post(
        "/api/v1/fhir/tests",
        headers={"Authorization": f"Bearer {token}"},
        json={"tests": ["cbc"], "patient_id": "p2"},
    )
    assert res.status_code == 200
    bundle = res.json()
    assert bundle["entry"][0]["resource"]["code"]["text"] == "cbc"


def test_cost_chart_updates():
    """Chart.js dataset is updated in the websocket handler."""
    js = Path("sdb/ui/static/main.js").read_text()
    assert "datasets[0].data.push" in js


@pytest.mark.asyncio
async def test_ws_budget_query_param(monkeypatch):
    captured: dict[str, float | None] = {}

    class DummyBudgetManager:
        def __init__(self, *args, **kwargs):
            captured["limit"] = kwargs.get("budget")
            self.budget = kwargs.get("budget")
            self.spent = 0.0

        def add_test(self, *_args, **_kwargs):
            return None

        def over_budget(self) -> bool:
            return False

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            self.budget_manager = kwargs.get("budget_manager")
            self.spent = 0.0
            self.ordered_tests = []

        def run_turn(self, *_args, **_kwargs):
            return "ok"

    monkeypatch.setattr(ui_app, "BudgetManager", DummyBudgetManager)
    monkeypatch.setattr(ui_app, "Orchestrator", DummyOrchestrator)

    config = uvicorn.Config(app, host="127.0.0.1", port=8014, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        await asyncio.sleep(0.01)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8014") as client:
        res = await client.post(
            "/api/v1/login",
            json={"username": "physician", "password": "secret"},
        )
        token = res.json()["access_token"]
        async with aconnect_ws(
            f"ws://127.0.0.1:8014/api/v1/ws?token={token}&budget=42",
            client,
        ) as ws:
            await ws.send_json({"action": "question", "content": "hi"})
            await ws.receive_json()

    server.should_exit = True
    thread.join()

    assert captured["limit"] == 42.0


@pytest.mark.asyncio
async def test_ws_rate_limit(monkeypatch):
    monkeypatch.setattr(ui_app, "MESSAGE_RATE_LIMIT", 2)
    monkeypatch.setattr(ui_app, "MESSAGE_RATE_WINDOW", 1)
    ui_app.MESSAGE_HISTORY.clear()

    class DummyBudgetManager:
        def __init__(self, *_, **__):
            self.budget = None
            self.spent = 0.0

        def add_test(self, *_args, **_kwargs):
            return None

        def over_budget(self) -> bool:
            return False

    class DummyOrchestrator:
        def __init__(self, *_, **kwargs):
            self.budget_manager = kwargs.get("budget_manager")
            self.spent = 0.0
            self.ordered_tests = []

        def run_turn(self, *_args, **_kwargs):
            return "ok"

    monkeypatch.setattr(ui_app, "BudgetManager", DummyBudgetManager)
    monkeypatch.setattr(ui_app, "Orchestrator", DummyOrchestrator)

    config = uvicorn.Config(app, host="127.0.0.1", port=8015, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        await asyncio.sleep(0.01)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8015") as client:
        res = await client.post(
            "/api/v1/login",
            json={"username": "physician", "password": "secret"},
        )
        token = res.json()["access_token"]
        async with aconnect_ws(
            f"ws://127.0.0.1:8015/api/v1/ws?token={token}",
            client,
        ) as ws:
            for _ in range(2):
                await ws.send_json({"action": "question", "content": "hi"})
                await ws.receive_json()
            await ws.send_json({"action": "question", "content": "hi"})
            data = await ws.receive_json()
            assert "error" in data

    server.should_exit = True
    thread.join()
