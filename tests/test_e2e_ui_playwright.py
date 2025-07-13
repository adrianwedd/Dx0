import threading
import time
import uvicorn

from playwright.sync_api import sync_playwright

import sdb.ui.app as ui_app

app = ui_app.app


def _start_server(port: int):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    return server, thread


def _stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join()


def test_accessibility_roles():
    """UI exposes regions and focus outlines."""
    server, thread = _start_server(8020)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto("http://127.0.0.1:8020/api/v1")
        regions = page.locator("[role=region]")
        assert regions.count() >= 4
        page.locator("button").first.focus()
        outline = page.evaluate("getComputedStyle(document.activeElement).outlineStyle")
        assert outline != "none"
        browser.close()
    _stop_server(server, thread)


def test_dark_mode_colors():
    """Dark theme sets expected background color."""
    server, thread = _start_server(8021)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto("http://127.0.0.1:8021/api/v1")
        page.evaluate("document.documentElement.setAttribute('data-theme','dark')")
        bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
        assert bg == "rgb(36, 36, 36)"
        browser.close()
    _stop_server(server, thread)
