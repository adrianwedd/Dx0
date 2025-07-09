import pytest
import httpx
import threading
import uvicorn
import asyncio
from httpx_ws import aconnect_ws, WebSocketUpgradeError, WebSocketDisconnect
from starlette.testclient import TestClient

from sdb.ui.app import app


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
    res = client.get("/")
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
