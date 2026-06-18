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

from bson import ObjectId
import voyageai
from dotenv import load_dotenv
from langchain_core.tools import tool

from agent.db import db_sync as _db
from agent.logger import get_logger

load_dotenv()

log = get_logger(__name__)

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
                log.warning("embedding attempt failed, retrying", extra={
                    "attempt": attempt + 1, "wait_s": wait, "error": str(exc),
                })
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# JSON-safe serialiser — LangChain converts tool return values to strings via
# str(), which produces Python repr for datetime/ObjectId objects.  Neither
# json.loads nor ast.literal_eval can parse those repr strings, causing
# downstream nodes to receive a raw string instead of a list/dict.
# _json_safe() recursively converts non-JSON-serialisable types before the
# value is returned, so str(result) is always valid Python-literal JSON.
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively convert datetime → ISO string, ObjectId → str."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


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
    Atlas Hybrid Search: combines a supplier-name pipeline and a fuzzy capability-notes
    pipeline via $rankFusion (Reciprocal Rank Fusion). RRF merges independent rank lists
    rather than blending raw scores, so a supplier that ranks highly in either path rises
    to the top without being drowned out by the other path's score distribution.
    Use this to find suppliers with specific capabilities (e.g. 'expedited cold-chain',
    'emergency pharmaceutical', 'fast PPE delivery').
    """
    _proj = {
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

    # Primary: $rankFusion — two independent $search pipelines merged via RRF.
    # capability_match weighted higher (0.6) because the tool is primarily used
    # for capability queries, not exact supplier-name lookups.
    pipeline = [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "name_match": [
                            {
                                "$search": {
                                    "index": "suppliers_text_search",
                                    "text": {
                                        "query": query,
                                        "path":  "supplier_name",
                                    },
                                }
                            }
                        ],
                        "capability_match": [
                            {
                                "$search": {
                                    "index": "suppliers_text_search",
                                    "text": {
                                        "query": query,
                                        "path":  "notes",
                                        "fuzzy": {"maxEdits": 1},
                                    },
                                }
                            }
                        ],
                    }
                },
                "combination": {
                    "weights": {
                        "name_match":       0.4,
                        "capability_match": 0.6,
                    }
                },
            }
        },
        {"$addFields": {"search_score": {"$meta": "searchScore"}}},
        {"$project": _proj},
        {"$limit": 5},
    ]

    try:
        results = list(_db.suppliers.aggregate(pipeline))
        return results if results else [{"info": f"No capability match for query '{query}'"}]
    except Exception as exc:
        # $rankFusion requires Atlas M10+; fall back to compound $search on
        # older tiers or non-Atlas deployments.
        log.warning(
            "rankFusion unavailable, falling back to compound $search",
            extra={"error": str(exc)},
        )

    fallback_pipeline = [
        {
            "$search": {
                "index": "suppliers_text_search",
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
        {"$project": _proj},
        {"$limit": 5},
    ]

    try:
        results = list(_db.suppliers.aggregate(fallback_pipeline))
    except Exception as exc2:
        return [{"error": f"Atlas Search unavailable: {exc2}"}]

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
        return _json_safe(list(_db.short_term_memory.find(
            {"sku": sku, "location": location, "decided_at": {"$gte": since}},
            {"_id": 0},
            sort=[("decided_at", -1)],
            limit=5,
        )))
    except Exception as exc:
        return [{"error": f"Short-term memory unavailable: {exc}"}]


def _record_failed_memory_write(
    sku: str,
    location: str,
    memory_type: str,
    error: Exception,
    payload: dict,
) -> None:
    """Insert a structured failure record into failed_memory_writes.

    Called whenever a memory write (short-term or long-term) raises an exception
    so the failure can be retried by memory_retry_worker.py instead of being
    silently dropped.
    """
    now = datetime.now(timezone.utc)
    try:
        _db.failed_memory_writes.insert_one({
            "sku":          sku,
            "location":     location,
            "memory_type":  memory_type,
            "error":        str(error),
            "payload":      payload,
            "retry_count":  0,
            "created_at":   now,
            "last_attempt": now,
            "resolved":     False,
        })
    except Exception as inner_exc:
        log.error("could not record failed memory write to DB", extra={"error": str(inner_exc)})


def write_short_term_memory_sync(
    sku: str, location: str, recommendation: dict,
    days_remaining: float, auto_approved: bool,
) -> None:
    """Persist a decision summary to short_term_memory (TTL=24 h)."""
    payload = {
        "sku":                      sku,
        "location":                 location,
        "decided_at":               datetime.now(timezone.utc).isoformat(),
        "supplier_name":            recommendation.get("supplier_name"),
        "supplier_id":              recommendation.get("supplier_id"),
        "quantity":                 recommendation.get("quantity"),
        "confidence":               recommendation.get("confidence"),
        "days_of_stock_at_decision": days_remaining,
        "auto_approved":            auto_approved,
        "rationale_summary":        recommendation.get("rationale", "")[:300],
    }
    try:
        _db.short_term_memory.insert_one({
            **payload,
            "decided_at": datetime.now(timezone.utc),
        })
    except Exception as exc:
        _record_failed_memory_write(sku, location, "short_term", exc, payload)


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
        return _json_safe(list(_db.agent_memory.aggregate(pipeline)))
    except Exception as exc:
        return [{"error": f"Long-term memory retrieval failed: {exc}"}]


def write_order_history_sync(
    sku: str, location: str, category: str,
    supplier_id: str | None, supplier_name: str,
    quantity: int, unit_price: float, total_cost: float,
    days_of_stock: float, trend: str,
    rationale: str, proposed_order_id,
) -> None:
    """Embed the rationale and insert a live order into order_history with outcome='pending'.

    The outcome is updated later: 'human_approved', 'rejected', or 'delivered_on_time'
    by app.py (approve/reject handlers) and the simulator (stock recovery detection).
    proposed_order_id links back to proposed_orders for targeted outcome updates.
    """
    try:
        embedding = _get_embedding(rationale)
        _db.order_history.insert_one({
            "sku":                  sku,
            "location":             location,
            "category":             category,
            "supplier_id":          supplier_id,
            "supplier_name":        supplier_name,
            "quantity":             quantity,
            "unit_price":           unit_price,
            "total_cost":           total_cost,
            "days_of_stock_at_order": days_of_stock,
            "trend_at_order":       trend,
            "outcome":              "pending",
            "ordered_at":           datetime.now(timezone.utc),
            "rationale":            rationale,
            "embedding":            embedding,
            "proposed_order_id":    proposed_order_id,
        })
    except Exception as exc:
        log.warning("could not write to order_history", extra={"sku": sku, "error": str(exc)})


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
    payload = {
        "sku":           sku,
        "location":      location,
        "content":       content,
        "confidence":    confidence,
        "auto_approved": auto_approved,
        "created_at":    datetime.now(timezone.utc).isoformat(),
    }
    try:
        embedding = _get_embedding(content)
        _db.agent_memory.insert_one({
            **payload,
            "embedding":  embedding,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as exc:
        _record_failed_memory_write(sku, location, "long_term", exc, payload)


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
    st_payload = {
        "sku":                       sku,
        "location":                  location,
        "decided_at":                decided_at.isoformat(),
        "supplier_name":             supplier,
        "supplier_id":               supplier_id,
        "quantity":                  quantity,
        "confidence":                confidence,
        "days_of_stock_at_decision": None,
        "auto_approved":             False,
        "decided_by":                "human",
        "human_decision":            decision,
        "rationale_summary":         rationale,
    }
    try:
        _db.short_term_memory.insert_one({**st_payload, "decided_at": decided_at})
    except Exception as exc:
        _record_failed_memory_write(sku, location, "short_term", exc, st_payload)

    # ── Long-term memory (embedded) ───────────────────────────────────────────
    action_phrase = "Human APPROVED order" if decision == "approved" else "Human REJECTED order"
    content = (
        f"{action_phrase} for {sku} at {location}: {quantity} units from {supplier} "
        f"(agent confidence: {confidence}, total cost: ${total_cost:,.0f}). "
        f"{'Rationale: ' + rationale if rationale else 'No rationale recorded.'}"
    )
    lt_payload = {
        "sku":            sku,
        "location":       location,
        "content":        content,
        "confidence":     confidence,
        "auto_approved":  False,
        "decided_by":     "human",
        "human_decision": decision,
        "created_at":     decided_at.isoformat(),
    }
    try:
        embedding = _get_embedding(content)
        _db.agent_memory.insert_one({
            **lt_payload,
            "embedding":  embedding,
            "created_at": decided_at,
        })
    except Exception as exc:
        _record_failed_memory_write(sku, location, "long_term", exc, lt_payload)


# ---------------------------------------------------------------------------
# @tool wrappers for memory reads — used by the ReAct retrieval agent so the
# LLM can call these as tools during its tool-calling loop.
# The underlying functions remain callable directly for backward compatibility.
# ---------------------------------------------------------------------------

@tool
def get_recent_decisions(sku: str, location: str) -> list:
    """Get recent agent and human decisions for this SKU+location from the last 24 hours.
    Returns up to 5 entries including supplier choice, quantity, confidence level, and
    whether a human approved or rejected the order. Check this before recommending to
    avoid duplicating an in-flight order or repeating a rejected supplier."""
    return get_short_term_memories(sku, location)


@tool
def get_learned_patterns(situation_description: str) -> list:
    """Retrieve semantically similar past procurement patterns from long-term memory.
    Provide a plain-English description of the current situation (SKU, location,
    urgency level, stock level, and category) to find relevant historical decisions
    and their outcomes. Returns entries scored by semantic similarity."""
    return get_long_term_memories(situation_description)


# ---------------------------------------------------------------------------
# Recommendation validator — called by the audit_agent node (not an @tool)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Episodic memory — alert_lifecycle collection
# Tracks the full event timeline for every alert (create → decision → outcome)
# ---------------------------------------------------------------------------

def append_lifecycle_event(
    alert_id: str,
    event_type: str,
    payload: dict | None = None,
    sku: str = "",
    location: str = "",
) -> None:
    """Append a timestamped event to the alert's lifecycle document.

    Creates the document on first call for an alert_id (upsert).
    event_type values: alert_created, agent_decision, human_approved,
                       human_rejected, order_placed, stock_recovered
    """
    try:
        oid = ObjectId(alert_id)
    except Exception:
        oid = alert_id  # type: ignore[assignment]

    event = {"type": event_type, "at": datetime.now(timezone.utc), **(payload or {})}
    try:
        _db.alert_lifecycle.update_one(
            {"alert_id": oid},
            {
                "$push": {"events": event},
                "$setOnInsert": {
                    "alert_id": oid,
                    "sku":      sku,
                    "location": location,
                },
            },
            upsert=True,
        )
    except Exception as exc:
        log.warning("append_lifecycle_event failed", extra={
            "event_type": event_type, "error": str(exc),
        })


@tool
def get_applicable_procedures(sku_category: str, location: str) -> list:
    """Fetch human-confirmed procedural preference rules for a given category and location.

    These rules encode learned preferences derived from repeated approval patterns
    (e.g., 'always prefer Supplier X for surgical supplies at Location A').
    Only human-confirmed rules are returned — candidate rules pending review are excluded.
    Use these as strong prior preferences when all else is equal.
    """
    try:
        rules = _json_safe(list(_db.procedures.find(
            {
                "sku_category": sku_category,
                "location":     location,
                "human_confirmed": True,
            },
            {"_id": 0, "preferred_supplier_id": 1, "preferred_supplier_name": 1,
             "condition": 1, "evidence_count": 1, "last_updated": 1},
            sort=[("evidence_count", -1)],
            limit=5,
        )))
        return rules if rules else [
            {"info": f"No confirmed procedural rules for {sku_category} @ {location}"}
        ]
    except Exception as exc:
        return [{"error": f"Procedures unavailable: {exc}"}]


@tool
def get_episode_history(sku: str, location: str, limit: int = 3) -> list:
    """Fetch recent full alert lifecycle episodes for this SKU+location.

    Each episode shows the complete sequence of events from alert creation
    through human decision and stock recovery, so you can see what actually
    happened in past procurement cycles and whether orders resolved the gap.
    """
    try:
        episodes = _json_safe(list(_db.alert_lifecycle.find(
            {"sku": sku, "location": location},
            {"_id": 0},
            sort=[("events.0.at", -1)],
            limit=limit,
        )))
        return episodes if episodes else [
            {"info": f"No lifecycle episodes found for {sku} @ {location}"}
        ]
    except Exception as exc:
        return [{"error": f"Episode history unavailable: {exc}"}]


# ---------------------------------------------------------------------------
# Recommendation validator (called by audit_agent node, not an @tool)
# ---------------------------------------------------------------------------

def validate_recommendation(rec: dict, state: dict) -> list:
    """Validate a procurement recommendation against schema and business rules.

    Returns a list of error strings. An empty list means the recommendation is valid.
    """
    errors: list = []

    # ── Schema: required fields must be present and non-None ─────────────
    # supplier_id/supplier_name may be None on the zero-gap fast path (quantity=0).
    zero_gap = state.get("coverage_gap") == 0
    required = ["supplier_id", "supplier_name", "quantity", "rationale", "confidence"]
    for field in required:
        allow_none = zero_gap and field in ("supplier_id", "supplier_name")
        if field not in rec or (rec[field] is None and not allow_none):
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return errors  # type/business checks are meaningless if schema is broken

    # ── Types ──────────────────────────────────────────────────────────────
    try:
        qty = int(rec["quantity"])
        if qty < 0:
            errors.append(f"'quantity' must be >= 0, got {qty}")
    except (ValueError, TypeError):
        errors.append(f"'quantity' must be an integer, got {rec['quantity']!r}")
        qty = -1

    if str(rec.get("confidence", "")).lower() not in {"high", "medium", "low"}:
        errors.append(
            f"'confidence' must be 'high', 'medium', or 'low'; got {rec.get('confidence')!r}"
        )

    if len(str(rec.get("rationale", "")).strip()) < 10:
        errors.append("'rationale' is too short (minimum 10 characters required)")

    # ── Business rules ─────────────────────────────────────────────────────
    coverage_gap = state.get("coverage_gap")
    if coverage_gap == 0 and qty != 0:
        errors.append(
            f"coverage_gap is 0 (existing orders are sufficient) but quantity={qty} — "
            "must return quantity = 0 when the gap is already closed"
        )

    # Regulatory: pharmaceutical and laboratory SKUs must use an FDA-registered supplier.
    _FDA_REQUIRED = {"pharmaceutical", "laboratory"}
    category = state.get("inventory", {}).get("category", "")
    if category in _FDA_REQUIRED and not zero_gap:
        chosen_id = rec.get("supplier_id")
        suppliers  = state.get("suppliers", [])
        chosen_sup = next((s for s in suppliers if s.get("supplier_id") == chosen_id), None)
        if chosen_sup is not None and not chosen_sup.get("fda_registered", True):
            errors.append(
                f"Regulatory violation: '{chosen_sup.get('supplier_name', chosen_id)}' is not "
                f"FDA-registered. {category.capitalize()} SKUs require an FDA-licensed wholesale "
                "distributor (21 CFR Part 205). Choose a different supplier."
            )

    return errors
