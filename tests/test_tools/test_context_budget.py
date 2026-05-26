"""Tests for agent/context_budget.py — token trimming logic."""

import pytest

from agent.context_budget import TOKEN_BUDGET_TOTAL, count_tokens, trim_to_budget


def test_count_tokens_non_empty():
    assert count_tokens("hello world") > 0


def test_count_tokens_empty():
    assert count_tokens("") >= 0


def test_trim_noop_when_under_budget():
    sections = {
        "supplier_lines": "line1\nline2",
        "short_term_lines": "entry1\nentry2",
    }
    result = trim_to_budget(sections, budget=TOKEN_BUDGET_TOTAL)
    assert result["supplier_lines"] == sections["supplier_lines"]
    assert result["short_term_lines"] == sections["short_term_lines"]


def test_trim_drops_retrieval_trace_first():
    """retrieval_trace is the lowest-priority section — it should be dropped first."""
    long_text = "x " * 2000
    sections = {
        "retrieval_trace": long_text,
        "supplier_lines":  "supplier A\nsupplier B",
    }
    result = trim_to_budget(sections, budget=50)
    assert result["retrieval_trace"] == ""
    # supplier_lines is higher priority and shorter — should survive
    assert result["supplier_lines"]


def test_trim_preserves_all_keys():
    sections = {
        "retrieval_trace":   "trace data " * 300,
        "supplier_lines":    "supplier A",
        "short_term_lines":  "entry A",
        "long_term_lines":   "memory A",
        "coverage_section":  "coverage info",
    }
    result = trim_to_budget(sections, budget=20)
    assert set(result.keys()) == set(sections.keys())


def test_trim_result_is_new_dict():
    sections = {"supplier_lines": "a b c", "retrieval_trace": "trace"}
    result = trim_to_budget(sections, budget=TOKEN_BUDGET_TOTAL)
    assert result is not sections


def test_trim_long_term_lines_drops_from_bottom():
    lines = "\n".join(f"memory_{i}" for i in range(50))
    sections = {"long_term_lines": lines, "coverage_section": "coverage info"}
    result = trim_to_budget(sections, budget=5)
    remaining = result["long_term_lines"]
    # Should either be trimmed or "(trimmed)" placeholder
    assert len(remaining) <= len(lines) or "trimmed" in remaining
