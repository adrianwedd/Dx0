"""Tests for dark theme color contrast."""

import re
from pathlib import Path

HEX = r"#[0-9a-fA-F]{6}"


def hex_to_rgb(hex_color: str):
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return r, g, b


def relative_luminance(rgb):
    def channel(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(c1: str, c2: str) -> float:
    l1 = relative_luminance(hex_to_rgb(c1))
    l2 = relative_luminance(hex_to_rgb(c2))
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def test_theme_contrast():
    css = Path("webui/src/index.css").read_text()
    themes = {m.group(1): m.group(2) for m in re.finditer(r"--(foreground|background):\s*(%s)" % HEX, css)}
    assert contrast(themes['foreground'], themes['background']) >= 4.5
