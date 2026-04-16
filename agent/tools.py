"""
LangChain tools — the agent's read-only window into MongoDB.

Tools:
  get_inventory_position       → basic find by SKU + location
  get_supplier_options         → basic find by SKU, sorted by fill rate
  get_consumption_trend        → time series aggregation pipeline
  search_suppliers_by_capability → Atlas Search: full-text over supplier notes
  find_similar_past_orders     → Atlas Vector Search: semantic similarity on order_history
"""

import os
import time
from datetime import datetime, timedelta, timezone

import voyageai
from dotenv import load_dotenv
from langchain_core.tools import tool
from pymongo import MongoClient

load_dotenv()

_client = MongoClient(
    os.environ["MONGODB_URI"],
    serverSelectionTimeoutMS=5_000,
    connectTimeoutMS=10_000,
    socketTimeoutMS=30_000,
)
_db = _client["supply_chain_demo"]

# Grove API gateway (Azure-style api-key auth)
_GROVE_API_KEY  = os.environ["GROVE_API_KEY"]
_GROVE_BASE_URL = os.environ["GROVE_API_BASE_URL"]

# Voyage AI embedding client
_voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
_EMBEDDING_MODEL = "voyage-4-large"


def _get_embedding(text: str, max_retries: int = 3) -> list:
    """Embed a text string with Voyage AI voyage-4-large.

    Retries up to max_retries times with exponential back-off (1 s → 2 s → 4 s)
    so transient Voyage AI errors don't cause vector search tools to fail silently.
    """
    for attempt in range(max_retries):
        try:
            result = _voyage.embed([text], model=_EMBEDDING_MODEL, input_type="query")
            return result.embeddings[0]
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [WARN] Embedding attempt {attempt + 1} failed ({exc}); retrying in {wait} s …")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Tool 1: inventory position (basic find)
# ---------------------------------------------------------------------------

@tool
def get_inventory_position(sku: str, location: str) -> dict:
    """Get current inventory position for a SKU at a location including
    on-hand, on-order, reorder point, safety stock, and unit cost."""
    doc = _db.inventory.find_one({"sku": sku, "location": location}, {"_id": 0})
    if not doc:
        return {"error": f"No inventory record found for {sku} @ {location}"}
    return doc


# ---------------------------------------------------------------------------
# Tool 2: supplier options (basic find, sorted by fill rate)
# ---------------------------------------------------------------------------

@tool
def get_supplier_options(sku: str) -> list:
    """Get all approved suppliers for a SKU, sorted by fill rate (best first).
    Returns lead times, pricing, and reliability metrics."""
    docs = list(
        _db.suppliers.find({"sku": sku}, {"_id": 0, "notes": 0}).sort("fill_rate_pct", -1)
    )
    return docs if docs else [{"error": f"No suppliers found for {sku}"}]


# ---------------------------------------------------------------------------
# Tool 3: consumption trend (time series aggregation)
# ---------------------------------------------------------------------------

@tool
def get_consumption_trend(sku: str, location: str, days: int = 14) -> dict:
    """Get consumption trend for a SKU over the past N days.
    Returns avg_daily, trend (increasing/stable/decreasing), total consumed, peak_day."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    pipeline = [
        {"$match": {"sku": sku, "location": location, "timestamp": {"$gte": cutoff}}},
        {
            "$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "daily_total": {"$sum": "$quantity"},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    rows = list(_db.consumption_history.aggregate(pipeline))
    if not rows:
        return {"avg_daily": 0, "trend": "unknown", "total_consumed": 0, "peak_day": None, "days_sampled": 0}

    totals = [r["daily_total"] for r in rows]
    avg_daily = round(sum(totals) / len(totals), 1)

    mid = len(totals) // 2
    if mid > 0:
        first_half  = sum(totals[:mid]) / mid
        second_half = sum(totals[mid:]) / len(totals[mid:])
        trend = "increasing" if second_half > first_half * 1.10 else \
                "decreasing" if second_half < first_half * 0.90 else "stable"
    else:
        trend = "stable"

    return {
        "avg_daily":      avg_daily,
        "trend":          trend,
        "total_consumed": sum(totals),
        "peak_day":       max(rows, key=lambda r: r["daily_total"])["_id"],
        "days_sampled":   len(rows),
    }


# ---------------------------------------------------------------------------
# Tool 4: Atlas Search — full-text supplier search
# ---------------------------------------------------------------------------

@tool
def search_suppliers_by_capability(query: str) -> list:
    """
    Atlas Search: full-text search over supplier names and capability notes across
    the entire supplier catalog. Returns suppliers ranked by relevance score.
    Searching the whole catalog means the hit count varies (0–5) based on how many
    suppliers have relevant capabilities for the current situation.
    Use this to find suppliers with specific capabilities (e.g. 'expedited cold-chain',
    'emergency pharmaceutical', 'fast PPE delivery').
    """
    pipeline = [
        {
            "$search": {
                "index": "suppliers_text_search",
                # compound operator: supplier_name matches score 3× higher than
                # notes matches so exact name hits outrank fuzzy capability hits.
                # sku is intentionally excluded — searching "cold-chain" should
                # not accidentally match SKU strings.
                "compound": {
                    "should": [
                        {
                            "text": {
                                "query": query,
                                "path":  "supplier_name",
                                "score": {"boost": {"value": 3}},
                            }
                        },
                        {
                            "text": {
                                "query": query,
                                "path":  "notes",
                                "fuzzy": {"maxEdits": 1},
                            }
                        },
                    ]
                },
            }
        },
        {"$addFields": {"search_score": {"$meta": "searchScore"}}},
        {
            "$project": {
                "_id": 0,
                "supplier_id":           1,
                "supplier_name":         1,
                "sku":                   1,
                "lead_time_days":        1,
                "moq":                   1,
                "unit_price":            1,
                "fill_rate_pct":         1,
                "on_time_delivery_pct":  1,
                "notes":                 1,
                "search_score":          1,
            }
        },
        {"$limit": 5},
    ]

    try:
        results = list(_db.suppliers.aggregate(pipeline))
    except Exception as exc:
        return [{"error": f"Atlas Search unavailable: {exc}"}]

    return results if results else [{"info": f"No capability match for query '{query}'"}]


# ---------------------------------------------------------------------------
# Tool 5: Atlas Vector Search — semantic similarity on historical orders
# ---------------------------------------------------------------------------

@tool
def find_similar_past_orders(
    situation_text: str, limit: int = 3, min_score: float = 0.78,
    category: str | None = None,
) -> list:
    """
    Atlas Vector Search: semantic similarity search over historical purchase orders.
    Given a plain-English description of the current inventory situation, returns the
    most similar past orders above the similarity threshold — including their outcomes
    and rationales — to inform the current recommendation.
    Only orders with similarity_score >= min_score are returned, so the result count
    varies (0 to limit) based on how many genuine precedents exist.
    Pass category (e.g. 'medical', 'ppe') to pre-filter to same-category orders,
    improving relevance precision without shrinking the candidate pool unnecessarily.
    """
    try:
        query_embedding = _get_embedding(situation_text)
    except Exception as exc:
        return [{"error": f"Embedding generation failed: {exc}"}]

    # Over-fetch so the score filter doesn't exhaust the candidate pool.
    fetch_limit = limit + 10

    vector_search_stage: dict = {
        "index":         "order_history_vector_index",
        "path":          "embedding",
        "queryVector":   query_embedding,
        "numCandidates": max(50, fetch_limit * 4),
        "limit":         fetch_limit,
    }
    if category:
        # Atlas pre-filter: narrows ANN search to same-category orders before
        # scoring, reducing noise from unrelated product categories.
        vector_search_stage["filter"] = {"category": {"$eq": category}}

    pipeline = [
        {"$vectorSearch": vector_search_stage},
        # Materialise the score so we can filter on it
        {"$addFields": {"similarity_score": {"$meta": "vectorSearchScore"}}},
        # Drop orders below the similarity threshold
        {"$match": {"similarity_score": {"$gte": min_score}}},
        {"$limit": limit},
        {
            "$project": {
                "_id":                    0,
                "sku":                    1,
                "location":               1,
                "supplier_name":          1,
                "quantity":               1,
                "days_of_stock_at_order": 1,
                "trend_at_order":         1,
                "outcome":                1,
                "rationale":              1,
                "similarity_score":       1,
            }
        },
    ]

    try:
        results = list(_db.order_history.aggregate(pipeline))
    except Exception as exc:
        return [{"error": f"Vector Search unavailable: {exc}"}]

    return results if results else [{"info": "No similar past orders found above similarity threshold"}]


# ---------------------------------------------------------------------------
# Short-term memory — per-SKU rolling 24 h window (plain MongoDB reads/writes)
# These are regular functions, not @tool, called directly from graph.py nodes.
# ---------------------------------------------------------------------------

def get_short_term_memories(sku: str, location: str, hours: int = 24) -> list:
    """Return recent agent decisions for this SKU+location from the last N hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    try:
        return list(_db.short_term_memory.find(
            {"sku": sku, "location": location, "decided_at": {"$gte": since}},
            {"_id": 0},
            sort=[("decided_at", -1)],
            limit=5,
        ))
    except Exception as exc:
        return [{"error": f"Short-term memory unavailable: {exc}"}]


def write_short_term_memory_sync(
    sku: str, location: str, recommendation: dict,
    days_remaining: float, auto_approved: bool,
) -> None:
    """Persist a decision summary to short_term_memory (TTL=24 h)."""
    try:
        _db.short_term_memory.insert_one({
            "sku":                      sku,
            "location":                 location,
            "decided_at":               datetime.now(timezone.utc),
            "supplier_name":            recommendation.get("supplier_name"),
            "supplier_id":              recommendation.get("supplier_id"),
            "quantity":                 recommendation.get("quantity"),
            "confidence":               recommendation.get("confidence"),
            "days_of_stock_at_decision": days_remaining,
            "auto_approved":            auto_approved,
            "rationale_summary":        recommendation.get("rationale", "")[:300],
        })
    except Exception as exc:
        print(f"  [WARN] Short-term memory write failed: {exc}")


# ---------------------------------------------------------------------------
# Long-term memory — semantic summaries + Atlas Vector Search on agent_memory
# ---------------------------------------------------------------------------

def get_long_term_memories(
    situation_text: str, limit: int = 3, min_score: float = 0.72
) -> list:
    """
    Vector search on agent_memory: retrieve the most relevant past learnings
    for the current situation using Voyage AI embeddings.
    The collection grows as the agent makes decisions — empty on first run.
    """
    try:
        query_embedding = _get_embedding(situation_text)
    except Exception as exc:
        return [{"error": f"Embedding generation failed: {exc}"}]

    fetch_limit = limit + 5
    pipeline = [
        {
            "$vectorSearch": {
                "index":         "agent_memory_vector_index",
                "path":          "embedding",
                "queryVector":   query_embedding,
                "numCandidates": max(50, fetch_limit * 4),
                "limit":         fetch_limit,
            }
        },
        {"$addFields": {"memory_score": {"$meta": "vectorSearchScore"}}},
        {"$match": {"memory_score": {"$gte": min_score}}},
        {"$limit": limit},
        {
            "$project": {
                "_id":          0,
                "sku":          1,
                "location":     1,
                "content":      1,
                "confidence":   1,
                "created_at":   1,
                "memory_score": 1,
            }
        },
    ]
    try:
        return list(_db.agent_memory.aggregate(pipeline))
    except Exception as exc:
        return [{"error": f"Long-term memory retrieval failed: {exc}"}]


def write_long_term_memory_sync(
    sku: str, item_name: str, location: str, recommendation: dict,
    days_remaining: float, trend: str, auto_approved: bool,
) -> None:
    """
    Generate a natural-language summary of this decision, embed it with
    Voyage AI, and store it in agent_memory for future vector retrieval.
    """
    confidence = recommendation.get("confidence", "medium")
    quantity   = recommendation.get("quantity", 0)
    supplier   = recommendation.get("supplier_name", "")
    rationale  = recommendation.get("rationale", "")[:400]

    content = (
        f"{sku} ({item_name}) at {location}: {days_remaining} days of stock remaining, "
        f"{trend} consumption trend. Ordered {quantity} units from {supplier} "
        f"(confidence: {confidence}). {rationale} "
        f"{'Auto-approved.' if auto_approved else 'Sent for manual approval.'}"
    )
    try:
        embedding = _get_embedding(content)
        _db.agent_memory.insert_one({
            "sku":          sku,
            "location":     location,
            "content":      content,
            "embedding":    embedding,
            "confidence":   confidence,
            "auto_approved": auto_approved,
            "created_at":   datetime.now(timezone.utc),
        })
    except Exception as exc:
        print(f"  [WARN] Long-term memory write failed: {exc}")


# ---------------------------------------------------------------------------
# Human-in-the-loop memory — called from app.py on Approve / Reject
# ---------------------------------------------------------------------------

def write_human_decision_memory(order_doc: dict, decision: str) -> None:
    """Persist a human approve/reject decision to both memory layers.

    Called from app.py when a human clicks ✅ Approve or ❌ Reject.
    Runs in a background thread so the UI is not blocked by the Voyage AI
    embedding call required for the long-term memory entry.

    Short-term memory  — plain insert; picked up by get_short_term_memories
                         on the agent's very next run for this SKU+location.
                         Includes decided_by='human' and human_decision so the
                         LLM prompt can surface it explicitly.

    Long-term memory   — embedded natural-language summary inserted into
                         agent_memory; retrieved via Vector Search across all
                         future alerts so the agent learns approval/rejection
                         patterns over time.

    Args:
        order_doc: the proposed_orders document (as a dict with _id as ObjectId).
        decision:  'approved' or 'rejected'.
    """
    sku        = order_doc.get("sku", "")
    location   = order_doc.get("location", "")
    supplier   = order_doc.get("supplier_name", "unknown supplier")
    supplier_id = order_doc.get("supplier_id")
    quantity   = order_doc.get("quantity_recommended", 0)
    confidence = order_doc.get("confidence", "medium")
    total_cost = order_doc.get("total_cost", 0.0)
    rationale  = order_doc.get("rationale", "")[:300]
    decided_at = datetime.now(timezone.utc)

    # ── Short-term memory (no embedding needed) ───────────────────────────────
    try:
        _db.short_term_memory.insert_one({
            "sku":                       sku,
            "location":                  location,
            "decided_at":                decided_at,
            "supplier_name":             supplier,
            "supplier_id":               supplier_id,
            "quantity":                  quantity,
            "confidence":                confidence,
            "days_of_stock_at_decision": None,   # not available from order doc alone
            "auto_approved":             False,
            "decided_by":                "human",
            "human_decision":            decision,
            "rationale_summary":         rationale,
        })
    except Exception as exc:
        print(f"  [WARN] Human short-term memory write failed: {exc}")

    # ── Long-term memory (embedded) ───────────────────────────────────────────
    action_phrase = (
        f"Human APPROVED order" if decision == "approved"
        else f"Human REJECTED order"
    )
    content = (
        f"{action_phrase} for {sku} at {location}: {quantity} units from {supplier} "
        f"(agent confidence: {confidence}, total cost: ${total_cost:,.0f}). "
        f"{'Rationale: ' + rationale if rationale else 'No rationale recorded.'}"
    )
    try:
        embedding = _get_embedding(content)
        _db.agent_memory.insert_one({
            "sku":             sku,
            "location":        location,
            "content":         content,
            "embedding":       embedding,
            "confidence":      confidence,
            "auto_approved":   False,
            "decided_by":      "human",
            "human_decision":  decision,
            "created_at":      decided_at,
        })
    except Exception as exc:
        print(f"  [WARN] Human long-term memory write failed: {exc}")
