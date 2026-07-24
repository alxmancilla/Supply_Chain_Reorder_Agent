"""Offline graph topology tests with mocked nodes and no Atlas dependency."""

import asyncio
import importlib


def _import_graph(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("GROVE_API_KEY", "test-key")
    monkeypatch.setenv("GROVE_API_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage")
    return importlib.import_module("agent.graph")


def test_graph_compiles_without_mongodb_checkpointer(monkeypatch):
    graph_module = _import_graph(monkeypatch)

    compiled = graph_module.build_graph(checkpointer_required=False)

    assert compiled is not None


def test_zero_gap_graph_path_skips_recommend(monkeypatch):
    graph_module = _import_graph(monkeypatch)
    events = []

    async def fake_assess(state):
        events.append("assess_alert")
        return {
            "inventory": {"on_hand": 200, "reorder_point": 100},
            "existing_order_qty": 0,
            "coverage_gap": 0,
            "expedite": False,
        }

    async def fake_recommend(state):
        raise AssertionError("recommend should be skipped for zero-gap alerts")

    async def fake_save_order(state):
        events.append("save_order")
        return {
            "order_id": "offline-order",
            "human_approved": True,
            "final_recommendation": {"quantity": 0, "confidence": "high"},
            "decision_source": "agent",
            "human_decision": None,
            "requeue_alert": False,
        }

    async def fake_write_memories(state):
        events.append("write_memories")
        return {}

    monkeypatch.setattr(graph_module, "assess_alert", fake_assess)
    monkeypatch.setattr(graph_module, "recommend", fake_recommend)
    monkeypatch.setattr(graph_module, "save_order", fake_save_order)
    monkeypatch.setattr(graph_module, "write_memories", fake_write_memories)

    compiled = graph_module.build_graph(checkpointer_required=False)
    alert = {
        "_id": "000000000000000000000001",
        "sku": "MED-3017",
        "location": "DC-Texas",
        "on_hand": 200,
        "on_order": 0,
        "reorder_point": 100,
        "days_of_stock_remaining": 10.0,
    }

    asyncio.run(compiled.ainvoke({"alert": alert}))

    assert events == ["assess_alert", "save_order", "write_memories"]
