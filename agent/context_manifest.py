"""Build a UI-friendly manifest of the context used for a recommendation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _valid_count(value: Any) -> int:
    if not isinstance(value, list):
        return 0
    return len([v for v in value if isinstance(v, dict) and "error" not in v and "info" not in v])


def _tool_names(trace: list) -> list[str]:
    names: list[str] = []
    for item in trace or []:
        if isinstance(item, dict) and item.get("tool"):
            names.append(str(item["tool"]))
    return names


def build_context_manifest(state: dict, recommendation: dict | None = None) -> dict:
    """Summarize what context was selected, why, and how it was budgeted."""
    retrieval_results = state.get("retrieval_results", {}) or {}
    trace = state.get("retrieval_trace", []) or []
    inventory = state.get("inventory", {}) or {}
    consumption = state.get("consumption", {}) or {}
    recommendation = recommendation or state.get("recommendation", {}) or {}

    sources = [
        {
            "name": "Inventory position",
            "kind": "live_state",
            "records": 1 if inventory else 0,
            "why": "Grounds the decision in current on-hand, reorder point, and safety stock.",
        },
        {
            "name": "Active order coverage",
            "kind": "live_state",
            "records": 1,
            "why": "Prevents duplicate purchasing by subtracting active orders from the gap.",
        },
        {
            "name": "Approved suppliers",
            "kind": "retrieved_context",
            "records": _valid_count(state.get("suppliers", [])),
            "why": "Constrains the model to known suppliers, pricing, MOQ, lead time, and reliability.",
        },
        {
            "name": "Consumption trend",
            "kind": "retrieved_context",
            "records": 1 if consumption else 0,
            "why": "Determines whether demand is stable, rising, or falling before sizing an order.",
        },
        {
            "name": "Atlas Search supplier capability",
            "kind": "retrieved_context",
            "records": _valid_count(state.get("supplier_search_results", [])),
            "why": "Confirms capability fit such as cold chain, emergency fulfillment, or compliance.",
        },
        {
            "name": "Vector-similar past orders",
            "kind": "retrieved_context",
            "records": _valid_count(state.get("similar_orders", [])),
            "why": "Adds precedent from semantically similar historical ordering situations.",
        },
        {
            "name": "Short-term memory",
            "kind": "memory",
            "records": _valid_count(state.get("short_term_memories", [])),
            "why": "Carries recent human approvals/rejections and in-flight decision signals.",
        },
        {
            "name": "Long-term semantic memory",
            "kind": "memory",
            "records": _valid_count(state.get("long_term_memories", [])),
            "why": "Carries learned supplier and quantity patterns from prior decisions.",
        },
        {
            "name": "Procedural rules",
            "kind": "policy_rules",
            "records": _valid_count(retrieval_results.get("get_applicable_procedures", [])),
            "why": "Adds human-confirmed supplier preferences when applicable.",
        },
        {
            "name": "Episode history",
            "kind": "memory",
            "records": _valid_count(retrieval_results.get("get_episode_history", [])),
            "why": "Shows full alert-to-recovery outcomes, not only individual order documents.",
        },
    ]

    validation_rules = [
        "Recommendation JSON schema: required supplier, quantity, rationale, confidence.",
        "Supplier ID must match an approved supplier returned for this SKU.",
        "Pharmaceutical and laboratory SKUs require FDA-registered suppliers.",
        "Coverage-gap rule: quantity must be zero when active orders already close the gap.",
        "Auto-approval requires high confidence and total cost below $5,000.",
    ]

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sku": state.get("alert", {}).get("sku"),
        "location": state.get("alert", {}).get("location"),
        "coverage_gap": state.get("coverage_gap"),
        "expedite": state.get("expedite", False),
        "context_sources": sources,
        "retrieval_trace": trace,
        "tool_sequence": _tool_names(trace),
        "token_budget": state.get("context_budget_report", {}) or {},
        "validation_rules": validation_rules,
        "recommendation_snapshot": {
            "supplier_id": recommendation.get("supplier_id"),
            "supplier_name": recommendation.get("supplier_name"),
            "quantity": recommendation.get("quantity"),
            "confidence": recommendation.get("confidence"),
        },
    }
