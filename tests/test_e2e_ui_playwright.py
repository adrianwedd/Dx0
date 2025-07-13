import threading
import time
import socket
import uvicorn

import pytest

from playwright.sync_api import sync_playwright

import sdb.ui.app as ui_app

app = ui_app.app


def _get_free_port() -> int:
    """Return an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_server() -> tuple[uvicorn.Server, threading.Thread, int]:
    """Start the FastAPI server on a free port."""
    port = _get_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    return server, thread, port


def _stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join()


@pytest.mark.parametrize("browser_name", ["chromium", "firefox", "webkit"])
def test_accessibility_roles(browser_name: str):
    """UI exposes regions and focus outlines across browsers."""
    server, thread, port = _start_server()
    with sync_playwright() as pw:
        browser = getattr(pw, browser_name).launch()
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{port}/static/react/index.html")
        regions = page.locator("[role=region]")
        assert regions.count() >= 4
        page.locator("button").first.focus()
        outline = page.evaluate("getComputedStyle(document.activeElement).outlineStyle")
        assert outline != "none"
        browser.close()
    _stop_server(server, thread)


@pytest.mark.parametrize("browser_name", ["chromium", "firefox", "webkit"])
def test_dark_mode_colors(browser_name: str):
    """Dark theme sets expected background color across browsers."""
    server, thread, port = _start_server()
    with sync_playwright() as pw:
        browser = getattr(pw, browser_name).launch()
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{port}/static/react/index.html")
        page.evaluate("document.documentElement.setAttribute('data-theme','dark')")
        bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
        assert bg == "rgb(36, 36, 36)"
        browser.close()
    _stop_server(server, thread)


@pytest.mark.parametrize("browser_name", ["chromium", "firefox", "webkit"])
def test_keyboard_navigation_and_aria_labels(browser_name: str):
    """Keyboard tab order matches ARIA labels on login form."""
    server, thread, port = _start_server()
    with sync_playwright() as pw:
        browser = getattr(pw, browser_name).launch()
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{port}/static/react/index.html")
        page.wait_for_selector("input[name='user']")
        page.keyboard.press("Tab")
        label = page.evaluate(
            "document.activeElement.getAttribute('aria-label')"
        )
        assert label == "Username"
        page.keyboard.press("Tab")
        label = page.evaluate(
            "document.activeElement.getAttribute('aria-label')"
        )
        assert label == "Password"
        page.keyboard.press("Tab")
        tag = page.evaluate("document.activeElement.tagName")
        assert tag == "BUTTON"
        browser.close()
    _stop_server(server, thread)


@pytest.mark.parametrize("browser_name", ["chromium", "firefox"])
def test_responsive_layout(browser_name: str):
    """Layout adapts to mobile viewport without horizontal scroll."""
    server, thread, port = _start_server()
    with sync_playwright() as pw:
        browser = getattr(pw, browser_name).launch()
        page = browser.new_page(viewport={"width": 375, "height": 812})
        page.goto(f"http://127.0.0.1:{port}/static/react/index.html")
        width = page.evaluate("document.body.scrollWidth")
        assert width <= 375
        browser.close()
    _stop_server(server, thread)
