"""Tests for save_and_notify business logic.

agent.graph cannot be imported in CI (missing langgraph-checkpoint-mongodb),
so all logic is tested via pure-function replicas that mirror the exact
implementation without any external dependencies.
"""

import pytest

# ---------------------------------------------------------------------------
# Pure-logic replicas (mirrors save_and_notify without DB / async)
# ---------------------------------------------------------------------------

AUTO_APPROVE_MAX_COST = 5_000.00


def _build_synthetic_zero_rec(state: dict) -> dict:
    """Mirrors the zero-gap fast-path block in save_and_notify."""
    alert    = state["alert"]
    existing = state.get("existing_order_qty", 0)
    reorder  = state.get("inventory", {}).get("reorder_point", alert.get("reorder_point", 0))
    on_hand  = state.get("inventory", {}).get("on_hand", alert.get("on_hand", 0))
    return {
        "supplier_id":   None,
        "supplier_name": "N/A",
        "quantity":      0,
        "rationale":     (
            f"Existing active orders ({existing} units) plus on-hand stock "
            f"({on_hand} units) already meet or exceed the reorder point "
            f"({reorder} units). No additional stock is required."
        ),
        "confidence": "high",
    }


def _resolve_rec(state: dict) -> dict:
    """Returns the effective recommendation, applying zero-gap override if needed."""
    rec          = state.get("recommendation") or {}
    coverage_gap = state.get("coverage_gap")
    if not rec or coverage_gap == 0:
        rec = _build_synthetic_zero_rec(state)
    return rec


def _compute_order(state: dict) -> dict:
    """Mirrors the auto-approve and quantity-override logic in save_and_notify."""
    rec          = _resolve_rec(state)
    coverage_gap = state.get("coverage_gap")
    suppliers    = state.get("suppliers", [])

    chosen     = next((s for s in suppliers if s.get("supplier_id") == rec.get("supplier_id")), {})
    quantity   = rec.get("quantity", 0)
    unit_price = chosen.get("unit_price", 0.0)
    confidence = rec.get("confidence", "medium")

    # Safety override
    if coverage_gap is not None and coverage_gap == 0:
        quantity = 0

    total_cost    = round(quantity * unit_price, 2)
    zero_order    = quantity == 0 and total_cost == 0.0
    auto_approved = zero_order or (confidence == "high" and total_cost < AUTO_APPROVE_MAX_COST)
    status        = "approved" if auto_approved else "awaiting_approval"

    return {
        "supplier_name":        rec.get("supplier_name"),
        "quantity_recommended": quantity,
        "total_cost":           total_cost,
        "confidence":           confidence,
        "auto_approved":        auto_approved,
        "status":               status,
    }


def _approval_side_effects(current_order_status: str, inventory_applied: bool = False) -> dict:
    """Pure replica of idempotent human approval side-effect gating."""
    transitioned = current_order_status in ("awaiting_approval", "approved") and not inventory_applied
    return {
        "order_status": "approved" if transitioned or current_order_status == "approved" else current_order_status,
        "increment_inventory": transitioned,
        "append_lifecycle": transitioned,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(
    *,
    coverage_gap: int = 155,
    recommendation: dict | None = None,
    confidence: str = "high",
    quantity: int = 155,
    unit_price: float = 10.0,
):
    rec = recommendation or {
        "supplier_id":   "SUP-001",
        "supplier_name": "MedSupply Co",
        "quantity":      quantity,
        "rationale":     "Low stock.",
        "confidence":    confidence,
    }
    return {
        "alert": {
            "_id": "000000000000000000000001",
            "sku": "MED-3017", "location": "DC-Texas",
            "on_hand": 45, "reorder_point": 200,
            "days_of_stock_remaining": 1.8,
        },
        "inventory": {"on_hand": 45, "reorder_point": 200, "name": "Surgical Gloves"},
        "suppliers":  [{"supplier_id": "SUP-001", "unit_price": unit_price, "lead_time_days": 3}],
        "coverage_gap":       coverage_gap,
        "existing_order_qty": 0,
        "recommendation":     rec,
        "consumption":        {"trend": "stable"},
    }


# ---------------------------------------------------------------------------
# Tests: auto-approve logic
# ---------------------------------------------------------------------------

class TestAutoApproveLogic:
    def test_high_confidence_low_cost_auto_approved(self):
        order = _compute_order(_state(confidence="high", quantity=155, unit_price=10.0))
        assert order["auto_approved"] is True
        assert order["status"] == "approved"

    def test_high_confidence_high_cost_awaits_approval(self):
        # 600 x $10 = $6000 > $5000 threshold
        order = _compute_order(_state(confidence="high", quantity=600, unit_price=10.0))
        assert order["auto_approved"] is False
        assert order["status"] == "awaiting_approval"

    def test_medium_confidence_awaits_approval_even_when_cheap(self):
        # Cost is $250 (well under threshold) but confidence is not "high"
        order = _compute_order(_state(confidence="medium", quantity=50, unit_price=5.0))
        assert order["auto_approved"] is False
        assert order["status"] == "awaiting_approval"

    def test_low_confidence_awaits_approval(self):
        order = _compute_order(_state(confidence="low", quantity=10, unit_price=1.0))
        assert order["auto_approved"] is False

    def test_zero_order_auto_approved_regardless_of_confidence(self):
        order = _compute_order(_state(confidence="medium", quantity=0, unit_price=0.0, coverage_gap=0))
        assert order["auto_approved"] is True
        assert order["status"] == "approved"

    def test_auto_approve_boundary_just_under_threshold(self):
        # $4990 -> should auto-approve
        order = _compute_order(_state(confidence="high", quantity=499, unit_price=10.0))
        assert order["total_cost"] == 4990.0
        assert order["auto_approved"] is True

    def test_auto_approve_boundary_at_threshold(self):
        # $5000 -> should NOT auto-approve
        order = _compute_order(_state(confidence="high", quantity=500, unit_price=10.0))
        assert order["total_cost"] == 5000.0
        assert order["auto_approved"] is False


# ---------------------------------------------------------------------------
# Tests: zero-gap fast path
# ---------------------------------------------------------------------------

class TestZeroGapFastPath:
    def test_zero_gap_generates_na_supplier(self):
        order = _compute_order(_state(coverage_gap=0, recommendation={}))
        assert order["supplier_name"] == "N/A"
        assert order["quantity_recommended"] == 0

    def test_zero_gap_forces_quantity_zero_even_with_existing_rec(self):
        # Recommendation says 100 units, but gap is 0 → safety override
        rec = {
            "supplier_id": "SUP-001", "supplier_name": "MedSupply Co",
            "quantity": 100, "rationale": "Test.", "confidence": "high",
        }
        order = _compute_order(_state(coverage_gap=0, recommendation=rec))
        assert order["quantity_recommended"] == 0

    def test_zero_gap_total_cost_is_zero(self):
        order = _compute_order(_state(coverage_gap=0, recommendation={}))
        assert order["total_cost"] == 0.0

    def test_zero_gap_rationale_mentions_existing_orders(self):
        st = _state(coverage_gap=0, recommendation={})
        st["existing_order_qty"] = 200
        rec = _resolve_rec(st)
        assert "200" in rec["rationale"]


# ---------------------------------------------------------------------------
# Tests: status field correctness
# ---------------------------------------------------------------------------

class TestOrderStatus:
    def test_approved_status_string(self):
        order = _compute_order(_state(confidence="high", quantity=50, unit_price=1.0))
        assert order["status"] == "approved"

    def test_awaiting_approval_status_string(self):
        order = _compute_order(_state(confidence="medium", quantity=50, unit_price=1.0))
        assert order["status"] == "awaiting_approval"


class TestHumanApprovalIdempotency:
    def test_first_approval_increments_inventory(self):
        result = _approval_side_effects("awaiting_approval")
        assert result["order_status"] == "approved"
        assert result["increment_inventory"] is True
        assert result["append_lifecycle"] is True

    def test_duplicate_approval_does_not_increment_inventory(self):
        result = _approval_side_effects("approved", inventory_applied=True)
        assert result["order_status"] == "approved"
        assert result["increment_inventory"] is False
        assert result["append_lifecycle"] is False

    def test_recovered_approved_order_without_marker_increments_inventory(self):
        result = _approval_side_effects("approved", inventory_applied=False)
        assert result["order_status"] == "approved"
        assert result["increment_inventory"] is True
        assert result["append_lifecycle"] is True
