from starlette.testclient import TestClient

from sdb.ui.app import app


def test_websocket_chat():
    client = TestClient(app)
    res = client.post(
        "/login",
        json={"username": "physician", "password": "secret"},
    )
    token = res.json()["token"]
    with client.websocket_connect(f"/ws?token={token}") as ws:
        ws.send_json({"action": "question", "content": "cough"})
        data = ws.receive_json()
        assert "reply" in data
        assert data["total_spent"] == 0.0
        ws.send_json({"action": "test", "content": "complete blood count"})
        data = ws.receive_json()
        assert data["cost"] == 10.0
        assert data["total_spent"] == 10.0
