"""Pure order-decision policy used by the LangGraph save_order node."""

from __future__ import annotations

AUTO_APPROVE_MAX_USD = 5_000.00


def build_zero_gap_recommendation(state: dict) -> dict:
    """Return a zero-quantity recommendation when active coverage closes the gap."""
    alert = state["alert"]
    existing = state.get("existing_order_qty", 0)
    reorder = state.get("inventory", {}).get("reorder_point", alert.get("reorder_point", 0))
    on_hand = state.get("inventory", {}).get("on_hand", alert.get("on_hand", 0))
    return {
        "supplier_id": None,
        "supplier_name": "N/A",
        "quantity": 0,
        "rationale": (
            f"Existing active orders ({existing} units) plus on-hand stock "
            f"({on_hand} units) already meet or exceed the reorder point "
            f"({reorder} units). No additional stock is required."
        ),
        "confidence": "high",
    }


def compute_order_decision(state: dict, recommendation: dict | None = None) -> dict:
    """Compute pricing, approval route, and review reason for a recommendation."""
    coverage_gap = state.get("coverage_gap")
    rec = recommendation or state.get("recommendation") or {}
    if not rec or coverage_gap == 0:
        rec = build_zero_gap_recommendation(state)

    chosen = next(
        (s for s in state.get("suppliers", []) if s.get("supplier_id") == rec.get("supplier_id")),
        {},
    )
    quantity = rec.get("quantity", 0)
    unit_price = chosen.get("unit_price", 0.0)
    lead_time = chosen.get("lead_time_days", 0)
    confidence = rec.get("confidence", "medium")

    if coverage_gap is not None and coverage_gap == 0:
        quantity = 0

    total_cost = round(quantity * unit_price, 2)
    zero_order = quantity == 0 and total_cost == 0.0
    over_budget = total_cost >= AUTO_APPROVE_MAX_USD
    auto_approved = zero_order or (confidence == "high" and not over_budget)

    if zero_order:
        review_reason = None
    elif over_budget:
        review_reason = f"budget_threshold (${total_cost:,.0f} ≥ ${AUTO_APPROVE_MAX_USD:,.0f} limit)"
    elif confidence != "high":
        review_reason = f"confidence={confidence}"
    else:
        review_reason = None

    return {
        "recommendation": rec,
        "chosen_supplier": chosen,
        "quantity": quantity,
        "unit_price": unit_price,
        "lead_time": lead_time,
        "confidence": confidence,
        "total_cost": total_cost,
        "zero_order": zero_order,
        "over_budget": over_budget,
        "auto_approved": auto_approved,
        "review_reason": review_reason,
    }
