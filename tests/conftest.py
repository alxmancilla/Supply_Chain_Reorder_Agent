"""
Shared pytest fixtures for the Supply Chain Reorder Agent test suite.

Fixture summary
───────────────
test_db         — isolated MongoDB database on the same cluster; dropped at end of session
mock_llm        — deterministic LLM returning canned JSON keyed by phase/content
mock_embeddings — zero-vector embeddings (agent.embeddings.Embeddings interface);
                  tests structure not semantics
sample_alert    — minimal valid reorder_alert document
sample_state    — minimal AgentState with alert, inventory, suppliers, etc.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# ---------------------------------------------------------------------------
# Test database
# ---------------------------------------------------------------------------

def _mongo_available() -> bool:
    """Return True only when MongoDB Atlas is reachable from this machine."""
    try:
        client = MongoClient(
            os.environ["MONGODB_URI"],
            serverSelectionTimeoutMS=3_000,
            connectTimeoutMS=3_000,
        )
        client.admin.command("ping")
        client.close()
        return True
    except Exception:
        return False


# Evaluate once per session so we don't add 3 s to every test collection.
_MONGO_AVAILABLE = _mongo_available()

requires_db = pytest.mark.skipif(
    not _MONGO_AVAILABLE,
    reason="MongoDB Atlas not reachable from this environment",
)


@pytest.fixture(scope="session")
def mongo_client():
    if not _MONGO_AVAILABLE:
        pytest.skip("MongoDB Atlas not reachable")
    client = MongoClient(
        os.environ["MONGODB_URI"],
        serverSelectionTimeoutMS=5_000,
        connectTimeoutMS=10_000,
        socketTimeoutMS=30_000,
    )
    yield client
    client.close()


@pytest.fixture(scope="function")
def test_db(mongo_client):
    """A fresh isolated test database, cleaned after each test function.

    Drops collections individually rather than the whole database — Atlas M0
    free-tier users have readWrite but not dbAdmin, so drop_database() raises
    OperationFailure(13 Unauthorized).
    """
    db_name = f"supply_chain_test_{uuid.uuid4().hex[:8]}"
    db = mongo_client[db_name]
    yield db
    for coll_name in db.list_collection_names():
        db[coll_name].drop()


# ---------------------------------------------------------------------------
# Mock LLM — returns deterministic JSON based on the system prompt content
# ---------------------------------------------------------------------------

_LLM_RESPONSES = {
    "analysis": json.dumps({
        "best_supplier_id":   "SUP-001",
        "best_supplier_name": "MedSupply Co",
        "confidence":         "high",
        "risk_flags":         [],
        "reasoning_trace":    "Supplier has 98% fill rate and 5-day lead time.",
    }),
    "recommendation": json.dumps({
        "supplier_id":   "SUP-001",
        "supplier_name": "MedSupply Co",
        "quantity":      500,
        "rationale":     "Stock is critically low; ordering 500 units from MedSupply Co to restore coverage above reorder point.",
        "confidence":    "high",
    }),
    "audit": json.dumps({"valid": True, "errors": []}),
}


class _FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class _MockLLM:
    """Synchronous + async LLM stub returning canned JSON based on system prompt."""

    def _pick_response(self, messages: list) -> str:
        system_content = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                system_content = m.get("content", "")
                break
            if hasattr(m, "content") and hasattr(m, "type"):
                system_content = str(m.content)
                break
        if "ANALYSIS" in system_content.upper():
            return _LLM_RESPONSES["analysis"]
        if "RECOMMENDATION" in system_content.upper():
            return _LLM_RESPONSES["recommendation"]
        return _LLM_RESPONSES["analysis"]

    async def ainvoke(self, messages, **kwargs):
        return _FakeLLMResponse(self._pick_response(messages))

    def invoke(self, messages, **kwargs):
        return _FakeLLMResponse(self._pick_response(messages))


@pytest.fixture
def mock_llm():
    return _MockLLM()


# ---------------------------------------------------------------------------
# Mock embedding provider — returns zero-vector embeddings (1024 dims),
# matching the agent.embeddings.Embeddings interface (embed_query/embed_documents)
# so it can be dropped in for agent.embeddings.embeddings via patch().
# ---------------------------------------------------------------------------

_ZERO_EMBEDDING = [0.0] * 1024


class _MockEmbeddings:
    def embed_query(self, text: str) -> list:
        return _ZERO_EMBEDDING

    def embed_documents(self, texts: list) -> list:
        return [_ZERO_EMBEDDING for _ in texts]


@pytest.fixture
def mock_embeddings():
    return _MockEmbeddings()


# ---------------------------------------------------------------------------
# Mock reranking provider — respects top_k but does no scoring.
# Sized to match agent.rerank.Reranker protocol.
# ---------------------------------------------------------------------------

class _MockReranker:
    def rerank(self, query: str, documents: list, text_key: str, top_k: int = 3) -> list:
        return documents[:top_k]


@pytest.fixture
def mock_reranker():
    return _MockReranker()


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_alert():
    return {
        "_id":                    str(ObjectId()),
        "sku":                    "MED-3017",
        "location":               "DC-Texas",
        "on_hand":                45,
        "on_order":               0,
        "reorder_point":          200,
        "avg_daily_consumption":  25.0,
        "days_of_stock_remaining": 1.8,
        "status":                 "pending",
        "created_at":             datetime.now(timezone.utc),
    }


@pytest.fixture
def sample_inventory():
    return {
        "sku":          "MED-3017",
        "name":         "Surgical Gloves (L)",
        "location":     "DC-Texas",
        "category":     "surgical",
        "on_hand":      45,
        "on_order":     0,
        "reorder_point": 200,
        "safety_stock": 50,
        "unit_cost":    1.25,
    }


@pytest.fixture
def sample_suppliers():
    return [
        {
            "supplier_id":          "SUP-001",
            "supplier_name":        "MedSupply Co",
            "sku":                  "MED-3017",
            "lead_time_days":       5,
            "fill_rate_pct":        98,
            "unit_price":           1.30,
            "on_time_delivery_pct": 96,
            "moq":                  100,
        },
        {
            "supplier_id":          "SUP-002",
            "supplier_name":        "QuickMed LLC",
            "sku":                  "MED-3017",
            "lead_time_days":       3,
            "fill_rate_pct":        88,
            "unit_price":           1.55,
            "on_time_delivery_pct": 91,
            "moq":                  50,
        },
    ]


@pytest.fixture
def sample_state(sample_alert, sample_inventory, sample_suppliers):
    return {
        "alert":                   sample_alert,
        "inventory":               sample_inventory,
        "existing_order_qty":      0,
        "coverage_gap":            155,
        "expedite":                True,
        "suppliers":               sample_suppliers,
        "consumption":             {
            "avg_daily": 25.0, "trend": "increasing",
            "total_consumed": 350, "peak_day": "2025-04-10", "days_sampled": 14,
        },
        "supplier_search_results": [],
        "similar_orders":          [],
        "short_term_memories":     [],
        "long_term_memories":      [],
        "retrieval_results":       {},
        "retrieval_trace":         [],
        "analysis":                {
            "best_supplier_id":   "SUP-001",
            "best_supplier_name": "MedSupply Co",
            "confidence":         "high",
            "risk_flags":         [],
            "reasoning_trace":    "High fill rate, acceptable lead time.",
        },
        "recommendation":          {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      500,
            "rationale":     "Stock is critically low; ordering 500 units to restore coverage.",
            "confidence":    "high",
        },
        "retry_count":             0,
        "prior_errors":            [],
        "escalate_flag":           False,
        "expedite":                True,
    }
