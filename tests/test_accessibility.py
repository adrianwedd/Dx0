"""Accessibility tests for the web UI."""

from pathlib import Path
from starlette.testclient import TestClient

from sdb.ui.app import app


def test_roles_and_focus_styles():
    """Static assets include ARIA roles and focus styling."""

    client = TestClient(app)
    res = client.get("/api/v1")
    html = res.text
    assert ":focus" in html
    js_path = Path("sdb/ui/static/main.js")
    content = js_path.read_text(encoding="utf-8")
    assert "role='region'" in content
    assert "form-control" in content


def test_keyboard_navigation_classes():
    """Interactive elements expose Bootstrap classes for focusable controls."""

    js_path = Path("sdb/ui/static/main.js")
    content = js_path.read_text(encoding="utf-8")
    assert "btn btn-primary" in content
    assert "form-select" in content
