"""Tests for markdown_math.sanitize_obsidian_math."""

from __future__ import annotations

from obsidian_llm_wiki.markdown_math import sanitize_obsidian_math


def test_sanitize_converts_display_math():
    body = sanitize_obsidian_math("Before \\[\nE = mc^2\n\\] after")
    assert body == "Before $$\nE = mc^2\n$$ after"


def test_sanitize_wraps_bare_latex_line():
    body = sanitize_obsidian_math("\\frac{a}{b}\n")
    assert body == "$$ \\frac{a}{b} $$\n"


def test_sanitize_keeps_code_and_existing_math_unchanged():
    body = sanitize_obsidian_math("`\\frac{a}{b}`\n$$\nE = mc^2\n$$\n")
    assert body == "`\\frac{a}{b}`\n$$\nE = mc^2\n$$\n"
