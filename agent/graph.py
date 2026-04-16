"""
LangGraph agent — the core of the demo.

State machine:
    START → assess_alert → gather_context → reason_and_draft → save_and_notify → END

gather_context runs four MongoDB queries concurrently:
  1. get_supplier_options         (basic find, fill-rate sorted)
  2. get_consumption_trend        (time series aggregation)
  3. search_suppliers_by_capability  ← Atlas Search
  4. find_similar_past_orders        ← Atlas Vector Search

Run as a standalone process — watches reorder_alerts via Change Stream:
    python agent/graph.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import TypedDict

# Ensure the project root is on sys.path so `agent.*` imports work when the
# script is run directly as `python agent/graph.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bson import ObjectId
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver

from agent.prompts import SYSTEM_PROMPT, build_recommendation_prompt
from agent.tools import (
    find_similar_past_orders,
    get_consumption_trend,
    get_inventory_position,
    get_long_term_memories,
    get_short_term_memories,
    get_supplier_options,
    search_suppliers_by_capability,
    write_long_term_memory_sync,
    write_short_term_memory_sync,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
_MONGO_KWARGS = dict(
    serverSelectionTimeoutMS=5_000,
    connectTimeoutMS=10_000,
    socketTimeoutMS=30_000,
)
_sync_client  = MongoClient(os.environ["MONGODB_URI"], **_MONGO_KWARGS)
_async_client = AsyncIOMotorClient(os.environ["MONGODB_URI"], **_MONGO_KWARGS)
_db_sync      = _sync_client["supply_chain_demo"]
_db_async     = _async_client["supply_chain_demo"]

# ---------------------------------------------------------------------------
# LLM — Grove API gateway (Azure-style api-key auth)
# ---------------------------------------------------------------------------
_GROVE_API_KEY  = os.environ["GROVE_API_KEY"]
_GROVE_BASE_URL = os.environ["GROVE_API_BASE_URL"]

llm = ChatOpenAI(
    model="gpt-5.4",
    openai_api_key=_GROVE_API_KEY,
    openai_api_base=_GROVE_BASE_URL,
    default_headers={"api-key": _GROVE_API_KEY},
    temperature=0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_doc(doc: dict) -> dict:
    """
    Convert top-level BSON ObjectId values to strings so the LangGraph
    MongoDBSaver checkpointer (which uses msgpack) can serialize the state.
    Call this on any MongoDB document before placing it in AgentState.
    """
    return {k: str(v) if isinstance(v, ObjectId) else v for k, v in doc.items()}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    alert:                   dict   # Incoming reorder_alert document
    inventory:               dict   # get_inventory_position result
    suppliers:               list   # get_supplier_options result
    consumption:             dict   # get_consumption_trend result
    supplier_search_results: list   # Atlas Search result
    similar_orders:          list   # Atlas Vector Search result
    short_term_memories:     list   # Per-SKU decisions in last 24 h
    long_term_memories:      list   # Semantic memories from agent_memory
    existing_order_qty:      int    # Units already ordered (awaiting or approved) for this SKU+location
    coverage_gap:            int    # Units still needed to reach the reorder point
    recommendation:          dict   # LLM output
    order_id:                str    # _id of saved proposed_order


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def assess_alert(state: AgentState) -> AgentState:
    """Node 1 — fetch live inventory position and compute the coverage gap.

    The coverage gap is how many additional units are still needed to reach
    the reorder point after accounting for ALL active proposed orders
    (awaiting_approval OR approved but not yet delivered).  This prevents the
    agent from recommending a full reorder when a partial order already exists.
    """
    alert = state["alert"]
    inv   = get_inventory_position.invoke({"sku": alert["sku"], "location": alert["location"]})

    # Sum quantity_recommended across all active orders for this SKU + location.
    # We deliberately include 'awaiting_approval' orders because their quantity
    # is NOT yet reflected in inventory.on_order (that only updates on approval).
    active_orders = list(_db_sync.proposed_orders.find(
        {
            "sku":      alert["sku"],
            "location": alert["location"],
            "status":   {"$in": ["awaiting_approval", "approved"]},
        },
        {"quantity_recommended": 1, "_id": 0},
    ))
    existing_order_qty = sum(o.get("quantity_recommended", 0) for o in active_orders)

    reorder_point      = inv.get("reorder_point", alert.get("reorder_point", 0))
    on_hand            = inv.get("on_hand",        alert.get("on_hand",        0))
    effective_coverage = on_hand + existing_order_qty
    coverage_gap       = max(0, reorder_point - effective_coverage)

    print(
        f"  [assess_alert]  {alert['sku']} — "
        f"on_hand={on_hand}, existing_orders={existing_order_qty}, "
        f"reorder_point={reorder_point}, coverage_gap={coverage_gap}"
    )
    return {
        **state,
        "inventory":         inv,
        "existing_order_qty": existing_order_qty,
        "coverage_gap":       coverage_gap,
    }


async def gather_context(state: AgentState) -> AgentState:
    """
    Node 2 — run six MongoDB queries concurrently:
      • get_supplier_options              (basic find)
      • get_consumption_trend             (time series aggregate)
      • search_suppliers_by_capability    ← Atlas Search
      • find_similar_past_orders          ← Atlas Vector Search on order_history
      • get_short_term_memories           ← per-SKU rolling 24 h window
      • get_long_term_memories            ← Atlas Vector Search on agent_memory
    """
    alert     = state["alert"]
    inventory = state["inventory"]
    category  = inventory.get("category", "medical")
    item_name = inventory.get("name", "")
    days      = alert.get("days_of_stock_remaining", 0)

    if days < 2:
        urgency_phrase = "urgent emergency expedited critical shortage"
    elif days < 5:
        urgency_phrase = "expedited fast priority reliable"
    else:
        urgency_phrase = "standard routine planned scheduled"

    capability_query = f"{item_name} {urgency_phrase}"
    situation_text = (
        f"{alert['sku']} {item_name} inventory alert at {alert['location']}. "
        f"On hand: {alert['on_hand']} units, {days} days of stock remaining. "
        f"Category: {category}. "
        f"Average daily consumption: {alert.get('avg_daily_consumption', 'unknown')} units/day."
    )
    vector_limit = 5 if days < 2 else 3

    loop = asyncio.get_event_loop()

    suppliers_fut    = loop.run_in_executor(
        None, lambda: get_supplier_options.invoke({"sku": alert["sku"]})
    )
    consumption_fut  = loop.run_in_executor(
        None, lambda: get_consumption_trend.invoke(
            {"sku": alert["sku"], "location": alert["location"], "days": 14}
        )
    )
    search_fut       = loop.run_in_executor(
        None, lambda: search_suppliers_by_capability.invoke({"query": capability_query})
    )
    vector_fut       = loop.run_in_executor(
        None, lambda lim=vector_limit: find_similar_past_orders.invoke(
            {"situation_text": situation_text, "limit": lim, "category": category}
        )
    )
    short_term_fut   = loop.run_in_executor(
        None, lambda: get_short_term_memories(alert["sku"], alert["location"])
    )
    long_term_fut    = loop.run_in_executor(
        None, lambda st=situation_text: get_long_term_memories(st)
    )

    (suppliers, consumption, supplier_search,
     similar_orders, short_term_mems, long_term_mems) = await asyncio.gather(
        suppliers_fut, consumption_fut, search_fut,
        vector_fut, short_term_fut, long_term_fut,
    )

    def _count(results: list) -> int:
        return len([r for r in results if "error" not in r and "info" not in r])

    print(
        f"  [gather_context] {alert['sku']} — "
        f"{len(suppliers)} supplier(s), avg_daily={consumption.get('avg_daily')} u/d, "
        f"atlas_search={_count(supplier_search)} hits, "
        f"vector_search={_count(similar_orders)} similar orders, "
        f"short_term={_count(short_term_mems)} recent, "
        f"long_term={_count(long_term_mems)} memories"
    )

    return {
        **state,
        "suppliers":               suppliers,
        "consumption":             consumption,
        "supplier_search_results": supplier_search,
        "similar_orders":          similar_orders,
        "short_term_memories":     short_term_mems,
        "long_term_memories":      long_term_mems,
    }


_JSON_STRIP_FIX = (
    "Your previous response was not valid JSON. "
    "Return ONLY the raw JSON object — no markdown fences, no explanation, no trailing text."
)
_MAX_JSON_RETRIES = 3


def _parse_llm_json(raw: str) -> dict:
    """Strip markdown fences then parse JSON. Raises json.JSONDecodeError on failure."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


async def reason_and_draft(state: AgentState) -> AgentState:
    """Node 3 — single LLM call: produce a purchase order recommendation.

    Retries up to _MAX_JSON_RETRIES times if the model returns malformed JSON,
    feeding the parse error back so the model can self-correct.
    """
    prompt = build_recommendation_prompt(state)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    recommendation = None
    for attempt in range(_MAX_JSON_RETRIES):
        response = await llm.ainvoke(messages)
        raw      = response.content
        try:
            recommendation = _parse_llm_json(raw)
            break
        except json.JSONDecodeError as exc:
            if attempt < _MAX_JSON_RETRIES - 1:
                print(
                    f"  [reason_and_draft] JSON parse error (attempt {attempt + 1}/"
                    f"{_MAX_JSON_RETRIES}): {exc} — asking model to self-correct …"
                )
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user",      "content": f"{_JSON_STRIP_FIX}\nError: {exc}"},
                ]
            else:
                raise ValueError(
                    f"LLM returned invalid JSON after {_MAX_JSON_RETRIES} attempts: {exc}"
                ) from exc

    print(
        f"  [reason_and_draft] {state['alert']['sku']} — "
        f"recommend {recommendation.get('quantity')} units "
        f"from {recommendation.get('supplier_name')} "
        f"(confidence: {recommendation.get('confidence')})"
    )
    return {**state, "recommendation": recommendation}


async def save_and_notify(state: AgentState) -> AgentState:
    """Node 4 — write proposed_order, auto-approve if high confidence, write memories."""
    alert = state["alert"]
    rec   = state["recommendation"]

    chosen = next(
        (s for s in state["suppliers"] if s.get("supplier_id") == rec.get("supplier_id")),
        {},
    )
    quantity   = rec.get("quantity", 0)
    unit_price = chosen.get("unit_price", 0.0)

    # Safety override: if existing active orders already cover the reorder point
    # (coverage_gap == 0), no additional stock is needed regardless of what the
    # LLM suggested.  Setting quantity to 0 triggers the zero-order auto-approve
    # path below and closes the alert cleanly without a duplicate order.
    coverage_gap = state.get("coverage_gap")
    if coverage_gap is not None and coverage_gap == 0:
        quantity = 0
    lead_time  = chosen.get("lead_time_days", 0)
    confidence = rec.get("confidence", "medium")

    # Auto-approve conditions (either is sufficient):
    #   1. High confidence AND total cost is below the approval threshold.
    #   2. Zero-quantity / zero-cost order (agent decided no stock action is
    #      needed — nothing to spend, so no human review required).
    # Orders at or above the cost threshold always require human review
    # regardless of confidence, giving finance visibility over high-value spend.
    AUTO_APPROVE_MAX_COST = 2_500.00
    total_cost    = round(quantity * unit_price, 2)
    zero_order    = quantity == 0 and total_cost == 0.0
    auto_approved = zero_order or (confidence == "high" and total_cost < AUTO_APPROVE_MAX_COST)
    status        = "approved" if auto_approved else "awaiting_approval"

    clean_similar = [
        {k: v for k, v in o.items() if k != "embedding"}
        for o in state.get("similar_orders", [])
        if "error" not in o and "info" not in o
    ]

    order = {
        "sku":                    alert["sku"],
        "location":               alert["location"],
        "supplier_id":            rec.get("supplier_id"),
        "supplier_name":          rec.get("supplier_name"),
        "quantity_recommended":   quantity,
        "unit_price":             unit_price,
        "total_cost":             total_cost,
        "expected_delivery_days": lead_time,
        "rationale":              rec.get("rationale", ""),
        "confidence":             confidence,
        "similar_orders":         clean_similar,
        "atlas_search_used":      len([
            r for r in state.get("supplier_search_results", []) if "error" not in r
        ]) > 0,
        "status":                 status,
        "auto_approved":          auto_approved,
        "created_at":             datetime.now(timezone.utc),
        # alert["_id"] was serialized to str by _serialize_doc; restore ObjectId
        # so the reference stored in proposed_orders links correctly in MongoDB.
        "alert_id":               ObjectId(alert["_id"]),
    }

    result   = await _db_async.proposed_orders.insert_one(order)
    order_id = str(result.inserted_id)

    await _db_async.reorder_alerts.update_one(
        {"_id": ObjectId(alert["_id"])},
        {"$set": {"status": "processed", "order_id": result.inserted_id}},
    )

    # Reflect the approved quantity in inventory.on_order so the dashboard
    # shows accurate "On Order" numbers and the simulator's effective_stock
    # check (on_hand + on_order) prevents redundant alerts for this SKU.
    if auto_approved:
        await _db_async.inventory.update_one(
            {"sku": alert["sku"], "location": alert["location"]},
            {"$inc": {"on_order": quantity}},
        )

    # Write short-term and long-term memories concurrently
    days_remaining = alert.get("days_of_stock_remaining", 0)
    trend          = state.get("consumption", {}).get("trend", "stable")
    item_name      = state.get("inventory", {}).get("name", "")
    loop = asyncio.get_event_loop()
    await asyncio.gather(
        loop.run_in_executor(
            None, write_short_term_memory_sync,
            alert["sku"], alert["location"], rec, days_remaining, auto_approved,
        ),
        loop.run_in_executor(
            None, write_long_term_memory_sync,
            alert["sku"], item_name, alert["location"],
            rec, days_remaining, trend, auto_approved,
        ),
    )

    if auto_approved and zero_order:
        approval_tag = "AUTO-APPROVED (zero-quantity / zero-cost order)"
    elif auto_approved:
        approval_tag = "AUTO-APPROVED (high confidence + cost below threshold)"
    elif confidence != "high":
        approval_tag = "awaiting approval (confidence not high)"
    else:
        approval_tag = f"awaiting approval (cost ${total_cost:,.2f} ≥ $10,000 threshold)"
    print(
        f"  [save_and_notify] Order saved (id={order_id}), "
        f"total=${order['total_cost']:.2f}, "
        f"status={approval_tag}, "
        f"similar_orders_stored={len(clean_similar)}"
    )
    return {**state, "order_id": order_id}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("assess_alert",     assess_alert)
    builder.add_node("gather_context",   gather_context)
    builder.add_node("reason_and_draft", reason_and_draft)
    builder.add_node("save_and_notify",  save_and_notify)

    builder.add_edge(START,              "assess_alert")
    builder.add_edge("assess_alert",     "gather_context")
    builder.add_edge("gather_context",   "reason_and_draft")
    builder.add_edge("reason_and_draft", "save_and_notify")
    builder.add_edge("save_and_notify",  END)

    # MongoDBSaver persists the full graph state for every alert invocation.
    # Checkpoints land in the `checkpoints` collection of supply_chain_demo.
    # Each alert gets its own thread_id (= alert _id) so runs are isolated.
    checkpointer = MongoDBSaver(_sync_client, db_name="supply_chain_demo")
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


# ---------------------------------------------------------------------------
# Change Stream listener — with resume token persistence and crash recovery
# ---------------------------------------------------------------------------

_AGENT_STATE_COLLECTION = "agent_state"
_RESUME_TOKEN_DOC_ID    = "change_stream_resume_token"


def _load_resume_token() -> dict | None:
    """Load the last persisted Change Stream resume token from MongoDB.

    Returns None if no token has been saved yet (first run) so the stream
    starts from the current position.
    """
    doc = _db_sync[_AGENT_STATE_COLLECTION].find_one({"_id": _RESUME_TOKEN_DOC_ID})
    if doc and "token" in doc:
        print(f"[AGENT] Resuming Change Stream from saved token …")
        return doc["token"]
    return None


def _save_resume_token(token: dict) -> None:
    """Persist the Change Stream resume token so restarts don't miss events."""
    _db_sync[_AGENT_STATE_COLLECTION].update_one(
        {"_id": _RESUME_TOKEN_DOC_ID},
        {"$set": {"token": token, "saved_at": datetime.now(timezone.utc)}},
        upsert=True,
    )


async def _process_one_alert(alert_doc: dict) -> None:
    """Invoke the agent graph for a single alert document."""
    sku        = alert_doc.get("sku", "?")
    thread_id  = str(alert_doc["_id"])
    alert_doc  = _serialize_doc(alert_doc)
    config     = {"configurable": {"thread_id": thread_id}}
    await graph.ainvoke({"alert": alert_doc}, config=config)
    print(f"[AGENT] Done — order drafted for {sku}")


async def watch_alerts() -> None:
    """Watch reorder_alerts for new inserts and invoke the agent graph.

    Resilience features:
    • Resume token  — persisted after every processed event so a restart
                      picks up exactly where it left off (no missed alerts).
    • Crash recovery — wraps the stream in a retry loop that handles
                       ChangeStreamFatalError and InvalidateError (raised when
                       the collection is dropped, e.g. during a demo reset).
                       After an invalidation the loop waits briefly then
                       re-opens a fresh stream from the current position.
    """
    # Watch both inserts (new alerts from the simulator) AND updates where
    # status is reset to "pending" (alert requeued after a human rejection).
    # The $or pipeline is evaluated server-side so only matching events are
    # delivered to the agent — no spurious wakeups for unrelated updates.
    pipeline = [{"$match": {"$or": [
        {"operationType": "insert"},
        {
            "operationType": "update",
            "updateDescription.updatedFields.status": "pending",
        },
    ]}}]
    retry_delay = 2  # seconds; doubles on each consecutive failure up to 30 s

    while True:
        resume_token = _load_resume_token()
        # updateLookup performs a fresh read from the collection after each event
        # so the full document is available for both insert and update events.
        # "required" only guarantees the full document for inserts.
        watch_kwargs = dict(full_document="updateLookup")
        if resume_token:
            watch_kwargs["resume_after"] = resume_token

        print("Agent listening for reorder alerts via Change Stream …\n")
        try:
            async with _db_async.reorder_alerts.watch(pipeline, **watch_kwargs) as stream:
                retry_delay = 2  # reset back-off on successful connection
                async for change in stream:
                    # Persist the token immediately — even if processing fails
                    # we won't reprocess the same event on the next restart.
                    _save_resume_token(change["_id"])

                    alert_doc = change.get("fullDocument")
                    if not alert_doc or alert_doc.get("status") != "pending":
                        continue

                    sku = alert_doc.get("sku", "?")
                    print(f"\n[AGENT] Processing alert for {sku} …")
                    try:
                        await _process_one_alert(alert_doc)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[AGENT] Error processing {sku}: {exc}")

        except Exception as exc:  # noqa: BLE001
            # Covers ChangeStreamFatalError, InvalidateError (collection drop),
            # and transient network errors.
            err_type = type(exc).__name__
            print(f"[AGENT] Change Stream interrupted ({err_type}: {exc})")
            print(f"[AGENT] Reconnecting in {retry_delay} s …")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)
            # After a collection invalidation, drop the stale token so the
            # new stream starts from the current head rather than a position
            # that no longer exists.
            if "InvalidateError" in err_type or "CursorNotFound" in err_type:
                _db_sync[_AGENT_STATE_COLLECTION].delete_one({"_id": _RESUME_TOKEN_DOC_ID})
                print("[AGENT] Cleared stale resume token — stream will start fresh.")


if __name__ == "__main__":
    asyncio.run(watch_alerts())
