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
