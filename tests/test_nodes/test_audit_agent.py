"""Tests for audit_agent validation logic and route_after_audit routing."""

import asyncio
from unittest.mock import patch

import pytest

from agent.tools import validate_recommendation


class TestValidateRecommendation:
    def _state(self, gap=155):
        return {"coverage_gap": gap}

    def test_valid_recommendation_returns_no_errors(self):
        rec = {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      500,
            "rationale":     "Stock critically low; ordering 500 units.",
            "confidence":    "high",
        }
        errors = validate_recommendation(rec, self._state())
        assert errors == []

    def test_missing_required_field_returns_error(self):
        rec = {
            "supplier_name": "MedSupply Co",
            "quantity":      500,
            "rationale":     "Valid rationale text.",
            "confidence":    "high",
        }
        errors = validate_recommendation(rec, self._state())
        assert any("supplier_id" in e for e in errors)

    def test_invalid_confidence_returns_error(self):
        rec = {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      100,
            "rationale":     "Valid rationale here.",
            "confidence":    "super-high",
        }
        errors = validate_recommendation(rec, self._state())
        assert any("confidence" in e for e in errors)

    def test_short_rationale_returns_error(self):
        rec = {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      100,
            "rationale":     "Too short",
            "confidence":    "medium",
        }
        errors = validate_recommendation(rec, self._state())
        assert any("rationale" in e for e in errors)

    def test_zero_gap_with_nonzero_quantity_returns_error(self):
        rec = {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      100,
            "rationale":     "Ordering more stock just in case the gap reappears.",
            "confidence":    "medium",
        }
        errors = validate_recommendation(rec, self._state(gap=0))
        assert any("quantity" in e or "coverage_gap" in e for e in errors)

    def test_zero_quantity_for_zero_gap_is_valid(self):
        rec = {
            "supplier_id":   None,
            "supplier_name": "N/A",
            "quantity":      0,
            "rationale":     "Existing orders already cover the reorder point requirement.",
            "confidence":    "high",
        }
        errors = validate_recommendation(rec, self._state(gap=0))
        assert errors == []

    def test_negative_quantity_returns_error(self):
        rec = {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      -5,
            "rationale":     "This should not have a negative quantity value.",
            "confidence":    "low",
        }
        errors = validate_recommendation(rec, self._state())
        assert any("quantity" in e for e in errors)


def _route_after_audit(state: dict) -> str:
    """Pure replica of agent.graph.route_after_audit for unit testing without
    importing the full graph module (which requires langgraph-checkpoint-mongodb)."""
    if state.get("audit_result", {}).get("valid"):
        return "save_and_notify"
    if state.get("audit_retries", 0) >= 2:
        return "escalate"
    return "recommendation_agent"


class TestRouteAfterAudit:
    def test_valid_routes_to_save(self):
        state = {"audit_result": {"valid": True, "errors": []}, "audit_retries": 0,
                 "alert": {"sku": "MED-3017"}}
        assert _route_after_audit(state) == "save_and_notify"

    def test_invalid_first_retry_routes_to_recommendation(self):
        state = {"audit_result": {"valid": False, "errors": ["bad"]}, "audit_retries": 1,
                 "alert": {"sku": "MED-3017"}}
        assert _route_after_audit(state) == "recommendation_agent"

    def test_invalid_max_retries_routes_to_escalate(self):
        state = {"audit_result": {"valid": False, "errors": ["bad"]}, "audit_retries": 2,
                 "alert": {"sku": "MED-3017"}}
        assert _route_after_audit(state) == "escalate"
