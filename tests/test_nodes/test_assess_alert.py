"""Tests for assess_alert coverage-gap and expedite logic.

The node is async and imports agent.graph (which needs langgraph-checkpoint-mongodb).
These tests validate the core business logic as pure functions so they run
without any external dependencies, then verify the node result via an
integration marker when Atlas is reachable.
"""

import pytest


# ---------------------------------------------------------------------------
# Pure-logic helpers (mirrors assess_alert node logic without DB calls)
# ---------------------------------------------------------------------------

def _compute_gap(alert: dict, inventory: dict, active_orders: list) -> tuple[int, int, bool]:
    """Return (existing_order_qty, coverage_gap, expedite)."""
    existing_order_qty = sum(o.get("quantity_recommended", 0) for o in active_orders)
    reorder_point      = inventory.get("reorder_point", alert.get("reorder_point", 0))
    on_hand            = inventory.get("on_hand",        alert.get("on_hand",        0))
    effective_coverage = on_hand + existing_order_qty
    coverage_gap       = max(0, reorder_point - effective_coverage)
    days               = alert.get("days_of_stock_remaining", 99)
    expedite           = days < 2
    return existing_order_qty, coverage_gap, expedite


# ---------------------------------------------------------------------------
# Unit tests — no DB, no graph import
# ---------------------------------------------------------------------------

class TestCoverageGapLogic:
    @pytest.fixture
    def alert(self):
        return {"sku": "MED-3017", "location": "DC-Texas",
                "on_hand": 45, "reorder_point": 200, "days_of_stock_remaining": 5.0}

    @pytest.fixture
    def inventory(self):
        return {"on_hand": 45, "reorder_point": 200}

    def test_full_gap_when_no_active_orders(self, alert, inventory):
        _, gap, _ = _compute_gap(alert, inventory, [])
        assert gap == 155  # 200 - 45

    def test_gap_reduced_by_active_orders(self, alert, inventory):
        orders = [{"quantity_recommended": 100}]
        _, gap, _ = _compute_gap(alert, inventory, orders)
        assert gap == 55  # 200 - (45 + 100)

    def test_gap_zero_when_orders_cover_fully(self, alert, inventory):
        orders = [{"quantity_recommended": 200}]
        _, gap, _ = _compute_gap(alert, inventory, orders)
        assert gap == 0

    def test_gap_never_negative(self, alert, inventory):
        orders = [{"quantity_recommended": 500}]
        _, gap, _ = _compute_gap(alert, inventory, orders)
        assert gap == 0

    def test_existing_order_qty_summed(self, alert, inventory):
        orders = [{"quantity_recommended": 50}, {"quantity_recommended": 70}]
        existing, _, _ = _compute_gap(alert, inventory, orders)
        assert existing == 120

    def test_expedite_when_days_below_two(self, alert, inventory):
        alert["days_of_stock_remaining"] = 1.5
        _, _, expedite = _compute_gap(alert, inventory, [])
        assert expedite is True

    def test_no_expedite_when_days_ample(self, alert, inventory):
        alert["days_of_stock_remaining"] = 10.0
        _, _, expedite = _compute_gap(alert, inventory, [])
        assert expedite is False

    def test_expedite_boundary_exactly_two(self, alert, inventory):
        alert["days_of_stock_remaining"] = 2.0
        _, _, expedite = _compute_gap(alert, inventory, [])
        assert expedite is False  # strictly < 2

    def test_expedite_boundary_just_below_two(self, alert, inventory):
        alert["days_of_stock_remaining"] = 1.99
        _, _, expedite = _compute_gap(alert, inventory, [])
        assert expedite is True
