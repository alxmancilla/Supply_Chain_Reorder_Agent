"""
End-to-end integration test: insert alert → run graph → assert proposed_order saved.

Uses the test_db fixture (isolated MongoDB database) and mocks all LLM and
Voyage AI calls so the test never hits external services.

Requires langgraph-checkpoint-mongodb in the environment; tests are skipped
when the package is absent (e.g., local dev without the full Docker deps).
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId
from tests.conftest import requires_db

pytestmark = requires_db

# ---------------------------------------------------------------------------
# Guard: skip entire module if the graph can't be imported
# ---------------------------------------------------------------------------

try:
    from agent.graph import graph as _graph, _serialize_doc as _sd
    _GRAPH_AVAILABLE = True
except ImportError:
    _GRAPH_AVAILABLE = False

requires_graph = pytest.mark.skipif(
    not _GRAPH_AVAILABLE,
    reason="langgraph-checkpoint-mongodb not installed — run inside Docker to exercise these tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_llm_response(content: str):
    m = MagicMock()
    m.content = content
    return m


ANALYSIS_JSON = json.dumps({
    "best_supplier_id":   "SUP-001",
    "best_supplier_name": "MedSupply Co",
    "confidence":         "high",
    "risk_flags":         [],
    "reasoning_trace":    "Strong fill rate, short lead time.",
})

RECOMMENDATION_JSON = json.dumps({
    "supplier_id":   "SUP-001",
    "supplier_name": "MedSupply Co",
    "quantity":      155,
    "rationale":     "Stock is critically low; ordering 155 units to restore coverage above reorder point.",
    "confidence":    "high",
})


def _pick_response(messages):
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        if "ANALYSIS" in str(content).upper():
            return _fake_llm_response(ANALYSIS_JSON)
    return _fake_llm_response(RECOMMENDATION_JSON)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_graph
class TestFullPipeline:
    @pytest.fixture(autouse=True)
    def setup_db(self, test_db, sample_alert, sample_inventory, sample_suppliers):
        test_db.inventory.insert_one({**sample_inventory})
        test_db.suppliers.insert_many(sample_suppliers)
        self.alert_doc = {
            **sample_alert,
            "_id": ObjectId(),
        }
        test_db.reorder_alerts.insert_one(self.alert_doc)
        self.db = test_db

    def test_proposed_order_created(self):
        """Running the graph for a valid alert should create a proposed_order."""
        alert_serialized = _sd(self.alert_doc)

        async def _run():
            fake_llm = AsyncMock()
            fake_llm.ainvoke.side_effect = _pick_response

            react_result = {
                "messages": []  # empty — forces retrieval fallback
            }
            zero_embed = [0.0] * 1024

            with patch("agent.graph._llm_primary", fake_llm), \
                 patch("agent.graph._llm_fallback", fake_llm), \
                 patch("agent.graph.llm", fake_llm), \
                 patch("agent.graph._retrieval_react_agent") as mock_react, \
                 patch("agent.graph._db_sync", self.db), \
                 patch("agent.graph._db_async", self.db), \
                 patch("agent.tools._db", self.db), \
                 patch("agent.tools._voyage") as mock_voyage, \
                 patch("agent.tools._get_embedding", return_value=zero_embed):

                mock_react.ainvoke = AsyncMock(return_value=react_result)
                mock_voyage.embed.return_value = MagicMock(embeddings=[zero_embed])

                config = {"configurable": {"thread_id": str(self.alert_doc["_id"])}}
                await _graph.ainvoke({"alert": alert_serialized}, config=config)

        asyncio.get_event_loop().run_until_complete(_run())

        order = self.db.proposed_orders.find_one({
            "sku":      self.alert_doc["sku"],
            "location": self.alert_doc["location"],
        })
        assert order is not None, "Expected proposed_order to be created"
        assert order["supplier_name"] in ("MedSupply Co", "N/A")
        assert order["status"] in ("approved", "awaiting_approval")

    def test_zero_gap_skips_llm(self):
        """When coverage_gap is 0, the graph should skip LLM nodes entirely."""
        # Add an active order that covers the gap
        self.db.proposed_orders.insert_one({
            "sku":                  self.alert_doc["sku"],
            "location":             self.alert_doc["location"],
            "quantity_recommended": 300,
            "status":               "awaiting_approval",
        })
        alert_serialized = _sd(self.alert_doc)
        llm_call_count   = 0

        async def _run():
            fake_llm = AsyncMock()
            async def counting_invoke(messages, **kw):
                nonlocal llm_call_count
                llm_call_count += 1
                return _pick_response(messages)
            fake_llm.ainvoke.side_effect = counting_invoke

            with patch("agent.graph._llm_primary", fake_llm), \
                 patch("agent.graph._llm_fallback", fake_llm), \
                 patch("agent.graph.llm", fake_llm), \
                 patch("agent.graph._retrieval_react_agent") as mock_react, \
                 patch("agent.graph._db_sync", self.db), \
                 patch("agent.graph._db_async", self.db), \
                 patch("agent.tools._db", self.db), \
                 patch("agent.tools._get_embedding", return_value=[0.0] * 1024):

                mock_react.ainvoke = AsyncMock(return_value={"messages": []})
                config = {"configurable": {"thread_id": f"zero-gap-{self.alert_doc['_id']}"}}
                await _graph.ainvoke({"alert": alert_serialized}, config=config)

        asyncio.get_event_loop().run_until_complete(_run())

        assert llm_call_count == 0, "LLM should not be invoked when coverage_gap is 0"

        order = self.db.proposed_orders.find_one({
            "sku":                  self.alert_doc["sku"],
            "quantity_recommended": 0,
        })
        assert order is not None
        assert order["quantity_recommended"] == 0
