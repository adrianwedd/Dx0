import json
import time

import httpx
import pytest
from starlette.testclient import TestClient

import sdb.ui.app as ui_app
from sdb import token

app = ui_app.app


def fake_post_factory(client):
    def _post(url, json=None, timeout=30):
        path = url.replace("http://testserver/api/v1", "/api/v1")
        return client.post(path, json=json)
    return _post


def test_login_and_refresh(tmp_path, monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(httpx, "post", fake_post_factory(client))
    tok_file = tmp_path / "tok.json"
    monkeypatch.setattr(token, "TOKEN_PATH", tok_file)

    token.login("http://testserver/api/v1", "physician", "secret")
    assert tok_file.exists()
    data = json.load(open(tok_file))
    assert "access_token" in data
    old_refresh = data["refresh_token"]

    # expire token
    data["expires"] = int(time.time()) - 1
    with open(tok_file, "w") as fh:
        json.dump(data, fh)

    new_token = token.get_access_token("http://testserver/api/v1")
    assert new_token != ""
    data2 = json.load(open(tok_file))
    assert data2["refresh_token"] != old_refresh
