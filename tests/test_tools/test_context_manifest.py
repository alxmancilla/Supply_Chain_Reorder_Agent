"""Tests for context manifest construction."""

from agent.context_manifest import build_context_manifest


def test_context_manifest_summarizes_sources_and_guardrails():
    state = {
        "alert": {"sku": "MED-3017", "location": "DC-Texas"},
        "inventory": {"on_hand": 600, "reorder_point": 1000},
        "coverage_gap": 400,
        "expedite": False,
        "suppliers": [{"supplier_id": "SUP-003"}, {"error": "bad"}],
        "consumption": {"trend": "increasing"},
        "supplier_search_results": [{"supplier_id": "SUP-003"}],
        "similar_orders": [{"sku": "MED-3017"}],
        "short_term_memories": [{"human_decision": "rejected"}],
        "long_term_memories": [{"content": "Prefer BioPharm"}],
        "retrieval_results": {
            "get_applicable_procedures": [{"preferred_supplier_id": "SUP-003"}],
            "get_episode_history": [{"events": []}],
        },
        "retrieval_trace": [
            {"tool": "get_supplier_options"},
            {"tool": "get_consumption_trend"},
        ],
        "context_budget_report": {"budget": 6000, "total_after": 1200},
    }

    manifest = build_context_manifest(state, {"supplier_id": "SUP-003", "quantity": 400})

    assert manifest["sku"] == "MED-3017"
    assert manifest["coverage_gap"] == 400
    assert manifest["tool_sequence"] == ["get_supplier_options", "get_consumption_trend"]
    assert manifest["token_budget"]["total_after"] == 1200
    assert any(s["name"] == "Short-term memory" and s["records"] == 1 for s in manifest["context_sources"])
    assert any("Supplier ID" in rule for rule in manifest["validation_rules"])
