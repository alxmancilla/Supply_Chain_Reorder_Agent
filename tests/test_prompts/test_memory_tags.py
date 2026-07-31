"""Tests for HUMAN REJECTED / APPROVED tag rendering in prompts."""

import pytest

from agent.prompts import build_recommendation_prompt


class TestMemoryTags:
    def _state_with_memories(self, memories):
        return {
            "alert":    {"sku": "MED-3017", "location": "DC-Texas",
                         "_id": "000000000000000000000001",
                         "on_hand": 45, "days_of_stock_remaining": 1.8,
                         "reorder_point": 200},
            "inventory": {"name": "Surgical Gloves", "category": "surgical",
                          "on_hand": 45, "reorder_point": 200},
            "coverage_gap": 155,
            "existing_order_qty": 0,
            "expedite": True,
            "suppliers": [],
            "consumption": {"avg_daily": 25.0, "trend": "stable"},
            "supplier_search_results": [],
            "similar_orders": [],
            "short_term_memories": memories,
            "long_term_memories": [],
            "retrieval_results": {},
            "retrieval_trace": [],
            "analysis": {
                "best_supplier_id":   "SUP-001",
                "best_supplier_name": "MedSupply Co",
                "confidence":         "high",
                "risk_flags":         [],
                "reasoning_trace":    "Good fit.",
            },
            "recommendation": {},
            "retry_count": 0,
            "prior_errors": [],
            "escalate_flag": False,
        }

    def test_human_approved_tag_present(self):
        memories = [{
            "decided_by":    "human",
            "human_decision": "approved",
            "supplier_name": "MedSupply Co",
            "quantity":      200,
            "confidence":    "high",
        }]
        state = self._state_with_memories(memories)
        prompt = build_recommendation_prompt(state)
        assert "HUMAN APPROVED" in prompt or "approved" in prompt.lower()

    def test_human_rejected_tag_present(self):
        memories = [{
            "decided_by":    "human",
            "human_decision": "rejected",
            "supplier_name": "QuickMed LLC",
            "quantity":      100,
            "confidence":    "low",
        }]
        state = self._state_with_memories(memories)
        prompt = build_recommendation_prompt(state)
        assert "HUMAN REJECTED" in prompt or "rejected" in prompt.lower()

    def test_agent_decision_shows_supplier_not_human_override(self):
        """An agent auto-approval entry should surface the supplier name,
        and the overall human_decision count should be zero (no human tag in entries)."""
        memories = [{
            "decided_by":    "agent",
            "supplier_name": "MedSupply Co",
            "quantity":      300,
            "confidence":    "high",
            "auto_approved": True,
        }]
        state = self._state_with_memories(memories)
        prompt = build_recommendation_prompt(state)
        # Supplier name should appear from the entry
        assert "MedSupply Co" in prompt
        # Count occurrences of the ⚠ tag — legend adds one; a per-entry rejection
        # would add a second. Only one means no individual entry was tagged rejected.
        assert prompt.count("⚠") <= 1
