"""
LangGraph agent — multi-agent reorder pipeline.

Graph topology:
    START
      → assess_alert          (DB-only: inventory position + coverage gap)
      → route_by_urgency      (conditional: zero-gap fast path vs retrieval)
          ↘ save_and_notify   (zero-gap fast path — no LLM needed)
          ↘ retrieval_agent   (ReAct tool-calling loop: gathers context)
              → analysis_agent        (evaluates suppliers, assigns confidence)
                  → recommendation_agent  (calculates quantity, writes rationale)
                      → audit_agent       (validates schema + business rules)
                          ↘ recommendation_agent  (retry, max 2 times)
                          ↘ save_and_notify        (valid recommendation)
      → END

ReAct retrieval tools:
  1. get_inventory_position        (basic find)
  2. get_supplier_options          (basic find, fill-rate sorted)
  3. get_consumption_trend         (time series aggregation)
  4. search_suppliers_by_capability   ← Atlas Search
  5. find_similar_past_orders         ← Atlas Vector Search on order_history
  6. get_recent_decisions             ← per-SKU rolling 24 h window
  7. get_learned_patterns             ← Atlas Vector Search on agent_memory

Run as a standalone process — watches reorder_alerts via Change Stream:
    python agent/graph.py
"""

import ast
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bson import ObjectId
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver

from agent import sku_lock
from agent.logger import get_logger
from agent.schemas import validate_alert, write_dead_letter
from agent.prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    RECOMMENDATION_SYSTEM_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
    build_analysis_prompt,
    build_recommendation_prompt,
)
from agent.tools import (
    append_lifecycle_event,
    find_similar_past_orders,
    get_applicable_procedures,
    get_consumption_trend,
    get_episode_history,
    get_inventory_position,
    get_learned_patterns,
    get_long_term_memories,
    get_recent_decisions,
    get_short_term_memories,
    get_supplier_options,
    search_suppliers_by_capability,
    validate_recommendation,
    write_long_term_memory_sync,
    write_order_history_sync,
    write_short_term_memory_sync,
)

load_dotenv()

log = get_logger(__name__)

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
# LLM — Grove API gateway with fallback + circuit breaker
# ---------------------------------------------------------------------------
_GROVE_API_KEY  = os.environ["GROVE_API_KEY"]
_GROVE_BASE_URL = os.environ["GROVE_API_BASE_URL"]

# Optional fallback: set GROVE_FALLBACK_API_KEY and GROVE_FALLBACK_BASE_URL in .env
# to enable a secondary LLM endpoint when the primary fails.
_FALLBACK_API_KEY  = os.getenv("GROVE_FALLBACK_API_KEY", _GROVE_API_KEY)
_FALLBACK_BASE_URL = os.getenv("GROVE_FALLBACK_BASE_URL", _GROVE_BASE_URL)
_FALLBACK_MODEL    = os.getenv("GROVE_FALLBACK_MODEL", "gpt-5.4")

_llm_primary = ChatOpenAI(
    model="gpt-5.4",
    openai_api_key=_GROVE_API_KEY,
    openai_api_base=_GROVE_BASE_URL,
    default_headers={"api-key": _GROVE_API_KEY},
    temperature=0,
)
_llm_fallback = ChatOpenAI(
    model=_FALLBACK_MODEL,
    openai_api_key=_FALLBACK_API_KEY,
    openai_api_base=_FALLBACK_BASE_URL,
    default_headers={"api-key": _FALLBACK_API_KEY},
    temperature=0,
)

# Circuit breaker state — module-level so it survives across graph invocations.
_circuit_failures: int = 0
_CIRCUIT_THRESHOLD: int = 3   # consecutive failures before opening the circuit


class CircuitOpenError(RuntimeError):
    """Raised when the LLM circuit breaker is open."""


def _get_llm() -> ChatOpenAI:
    """Return the active LLM, raising CircuitOpenError if the circuit is open."""
    if _circuit_failures >= _CIRCUIT_THRESHOLD:
        raise CircuitOpenError(
            f"LLM circuit breaker open after {_circuit_failures} consecutive failures — "
            "routing alert to human review queue"
        )
    return _llm_primary


async def _llm_invoke(messages: list) -> object:
    """Invoke the LLM with automatic fallback to the secondary endpoint.

    Resets the circuit breaker on success. Increments the failure counter and
    tries the fallback on primary failure. If both fail, opens the circuit.
    """
    global _circuit_failures

    llm_to_use = _get_llm()
    try:
        response = await llm_to_use.ainvoke(messages)
        _circuit_failures = 0   # reset on success
        return response
    except CircuitOpenError:
        raise
    except Exception as primary_exc:
        log.warning("primary LLM failed, trying fallback", extra={"error": str(primary_exc)})
        try:
            response = await _llm_fallback.ainvoke(messages)
            _circuit_failures = 0
            return response
        except Exception as fallback_exc:
            _circuit_failures += 1
            log.error(
                "fallback LLM also failed",
                extra={
                    "error":               str(fallback_exc),
                    "consecutive_failures": _circuit_failures,
                    "threshold":           _CIRCUIT_THRESHOLD,
                },
            )
            raise fallback_exc


# Keep the bare `llm` name for the ReAct agent (create_react_agent requires it at init time)
llm = _llm_primary

# ---------------------------------------------------------------------------
# ReAct retrieval agent — LLM-driven tool selection
# ---------------------------------------------------------------------------
_RETRIEVAL_TOOLS = [
    get_inventory_position,
    get_supplier_options,
    get_consumption_trend,
    search_suppliers_by_capability,
    find_similar_past_orders,
    get_recent_decisions,
    get_learned_patterns,
    get_episode_history,
    get_applicable_procedures,
]

_retrieval_react_agent = create_react_agent(llm, _RETRIEVAL_TOOLS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_doc(doc: dict) -> dict:
    """Convert top-level BSON ObjectId values to strings for MongoDBSaver."""
    return {k: str(v) if isinstance(v, ObjectId) else v for k, v in doc.items()}


def _extract_retrieval_results(messages: list) -> tuple[dict, list]:
    """Extract tool call results and a call trace from ReAct agent messages.

    Returns (results_by_tool_name, trace_list). If a tool is called multiple
    times, the last result wins.
    """
    results: dict = {}
    trace: list   = []

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                trace.append({
                    "tool": tc["name"],
                    "args": {k: str(v)[:150] for k, v in tc.get("args", {}).items()},
                })
        elif isinstance(msg, ToolMessage):
            raw = msg.content
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    # LangChain serialises Python lists/dicts as repr strings
                    # (e.g. "[{'key': 'val'}]") when the tool returns non-JSON.
                    # ast.literal_eval handles that; fall back to raw string only
                    # if neither parser works.
                    try:
                        parsed = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        parsed = raw
            else:
                parsed = raw
            results[msg.name] = parsed

    return results, trace


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


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # ── Existing fields (populated by assess_alert) ───────────────────────
    alert:                   dict   # Incoming reorder_alert document
    inventory:               dict   # get_inventory_position result
    existing_order_qty:      int    # Units already on active orders for this SKU+location
    coverage_gap:            int    # Units still needed to reach the reorder point
    expedite:                bool   # True when days_of_stock_remaining < 2

    # ── Retrieval fields (populated by retrieval_agent_node) ─────────────
    suppliers:               list   # get_supplier_options result
    consumption:             dict   # get_consumption_trend result
    supplier_search_results: list   # Atlas Search result
    similar_orders:          list   # Atlas Vector Search on order_history
    short_term_memories:     list   # Per-SKU decisions in last 24 h
    long_term_memories:      list   # Semantic memories from agent_memory
    retrieval_results:       dict   # Raw tool results keyed by tool name
    retrieval_trace:         list   # Ordered list of {tool, args} dicts

    # ── Analysis fields (populated by analysis_agent_node) ────────────────
    analysis:                dict   # {best_supplier_id, confidence, risk_flags, reasoning_trace}

    # ── Recommendation fields (populated by recommendation_agent_node) ────
    recommendation:          dict   # {supplier_id, supplier_name, quantity, rationale, confidence}

    # ── Audit fields (populated by audit_agent_node) ──────────────────────
    audit_result:            dict   # {valid: bool, errors: list[str]}
    audit_retries:           int    # Number of failed audit attempts so far

    # ── Persistence fields (populated by save_and_notify) ─────────────────
    order_id:                str    # _id of the saved proposed_order


# ---------------------------------------------------------------------------
# Node 1: assess_alert — DB-only, computes coverage gap
# ---------------------------------------------------------------------------

async def assess_alert(state: AgentState) -> AgentState:
    """Fetch live inventory position and compute the coverage gap.

    The coverage gap is how many additional units are still needed to reach
    the reorder point after accounting for ALL active proposed orders
    (awaiting_approval OR approved but not yet delivered). This prevents the
    agent from recommending a full reorder when a partial order already exists.
    """
    alert = state["alert"]
    inv   = get_inventory_position.invoke({"sku": alert["sku"], "location": alert["location"]})

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
    days               = alert.get("days_of_stock_remaining", 99)
    expedite           = days < 2

    log.info("coverage gap computed", extra={
        "phase":              "assess_alert",
        "sku":                alert["sku"],
        "location":           alert.get("location"),
        "on_hand":            on_hand,
        "existing_orders":    existing_order_qty,
        "reorder_point":      reorder_point,
        "coverage_gap":       coverage_gap,
        "expedite":           expedite,
    })
    return {
        **state,
        "inventory":          inv,
        "existing_order_qty": existing_order_qty,
        "coverage_gap":       coverage_gap,
        "expedite":           expedite,
    }


# ---------------------------------------------------------------------------
# Node 2: retrieval_agent — ReAct tool-calling loop
# ---------------------------------------------------------------------------

async def retrieval_agent_node(state: AgentState) -> AgentState:
    """Run the ReAct retrieval agent to gather procurement context via tool calls.

    The LLM decides which tools to call and in what order based on the alert
    urgency and what data is needed. Results are extracted from the message
    history and mapped back to the standard AgentState fields so downstream
    nodes (analysis, recommendation, save) can read them as before.
    """
    alert     = state["alert"]
    inventory = state.get("inventory", {})
    days      = alert.get("days_of_stock_remaining", 99)
    gap       = state.get("coverage_gap", 0)
    expedite  = state.get("expedite", False)

    urgency_label = "CRITICAL" if days < 2 else "ELEVATED" if days < 5 else "STANDARD"

    context = (
        f"Gather context for a procurement decision.\n\n"
        f"SKU:              {alert['sku']}\n"
        f"Item name:        {inventory.get('name', 'unknown')}\n"
        f"Location:         {alert['location']}\n"
        f"Category:         {inventory.get('category', 'medical')}\n"
        f"On hand:          {alert.get('on_hand', 0)} units\n"
        f"Days of stock:    {days}\n"
        f"Urgency:          {urgency_label}\n"
        f"Coverage gap:     {gap} units needed\n"
    )
    if expedite:
        context += "⚡ EXPEDITE MODE — prioritise speed and availability over cost.\n"

    try:
        result = await _retrieval_react_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=RETRIEVAL_SYSTEM_PROMPT + "\n\n" + context),
                ]
            },
            config={"recursion_limit": 20},
        )
        retrieval_results, retrieval_trace = _extract_retrieval_results(result["messages"])
    except Exception as exc:
        log.warning("ReAct agent failed, falling back to direct calls", extra={
            "phase": "retrieval_agent", "sku": alert.get("sku"), "error": str(exc),
        })
        retrieval_results, retrieval_trace = {}, []

    # Map tool results to the existing state fields for downstream compatibility.
    # Fallback to direct calls if the agent missed a required tool.
    suppliers = retrieval_results.get("get_supplier_options", [])
    if not suppliers or (len(suppliers) == 1 and "error" in suppliers[0]):
        suppliers = get_supplier_options.invoke({"sku": alert["sku"]})

    consumption = retrieval_results.get("get_consumption_trend", {})
    if not consumption or consumption.get("avg_daily") is None:
        consumption = get_consumption_trend.invoke(
            {"sku": alert["sku"], "location": alert["location"], "days": 14}
        )

    supplier_search = retrieval_results.get("search_suppliers_by_capability", [])
    similar_orders  = retrieval_results.get("find_similar_past_orders", [])
    short_term_mems = retrieval_results.get("get_recent_decisions", [])
    long_term_mems  = retrieval_results.get("get_learned_patterns", [])

    def _count(results: list) -> int:
        return len([r for r in results if "error" not in r and "info" not in r])

    log.info("retrieval complete", extra={
        "phase":         "retrieval_agent",
        "sku":           alert["sku"],
        "location":      alert.get("location"),
        "tool_calls":    len(retrieval_trace),
        "suppliers":     len(suppliers),
        "avg_daily":     consumption.get("avg_daily"),
        "trend":         consumption.get("trend", "?"),
        "atlas_hits":    _count(supplier_search),
        "vector_hits":   _count(similar_orders),
        "short_term":    _count(short_term_mems),
        "long_term":     _count(long_term_mems),
    })

    return {
        **state,
        "suppliers":               suppliers,
        "consumption":             consumption,
        "supplier_search_results": supplier_search,
        "similar_orders":          similar_orders,
        "short_term_memories":     short_term_mems,
        "long_term_memories":      long_term_mems,
        "retrieval_results":       retrieval_results,
        "retrieval_trace":         retrieval_trace,
    }


# ---------------------------------------------------------------------------
# Node 3: analysis_agent — supplier evaluation and confidence scoring
# ---------------------------------------------------------------------------

async def analysis_agent_node(state: AgentState) -> AgentState:
    """Evaluate retrieved context and produce a structured risk assessment.

    Outputs: best_supplier_id, confidence (high/medium/low), risk_flags, reasoning_trace.
    Does NOT calculate quantities — that is the recommendation agent's responsibility.
    """
    prompt   = build_analysis_prompt(state)
    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    analysis = None
    for attempt in range(_MAX_JSON_RETRIES):
        response = await _llm_invoke(messages)
        try:
            analysis = _parse_llm_json(response.content)
            break
        except json.JSONDecodeError as exc:
            if attempt < _MAX_JSON_RETRIES - 1:
                print(
                    f"  [analysis_agent] JSON parse error (attempt {attempt + 1}/"
                    f"{_MAX_JSON_RETRIES}): {exc} — asking model to self-correct …"
                )
                messages = messages + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user",      "content": f"{_JSON_STRIP_FIX}\nError: {exc}"},
                ]
            else:
                raise ValueError(
                    f"Analysis agent returned invalid JSON after {_MAX_JSON_RETRIES} attempts: {exc}"
                ) from exc

    log.info("analysis complete", extra={
        "phase":      "analysis_agent",
        "sku":        state["alert"]["sku"],
        "location":   state["alert"].get("location"),
        "supplier":   analysis.get("best_supplier_name"),
        "confidence": analysis.get("confidence"),
        "risk_flags": analysis.get("risk_flags", []),
    })
    return {**state, "analysis": analysis}


# ---------------------------------------------------------------------------
# Node 4: recommendation_agent — quantity calculation and rationale
# ---------------------------------------------------------------------------

async def recommendation_agent_node(state: AgentState) -> AgentState:
    """Draft the final purchase order recommendation.

    Receives the analyst's assessment (confidence, best supplier, risk flags)
    and produces a quantity + plain-English rationale for the human approver.
    Appends any previous audit validation errors to the prompt on retry.
    """
    prompt = build_recommendation_prompt(state)

    # On retry: prepend previous validation errors so the agent fixes them.
    prior_errors = state.get("audit_result", {}).get("errors", [])
    if prior_errors:
        error_block = (
            "\n\nPREVIOUS VALIDATION ERRORS — fix all of these before resubmitting:\n"
            + "\n".join(f"  - {e}" for e in prior_errors)
        )
        prompt = prompt + error_block

    messages = [
        {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    recommendation = None
    for attempt in range(_MAX_JSON_RETRIES):
        response = await _llm_invoke(messages)
        try:
            recommendation = _parse_llm_json(response.content)
            break
        except json.JSONDecodeError as exc:
            if attempt < _MAX_JSON_RETRIES - 1:
                print(
                    f"  [recommendation_agent] JSON parse error (attempt {attempt + 1}/"
                    f"{_MAX_JSON_RETRIES}): {exc} — asking model to self-correct …"
                )
                messages = messages + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user",      "content": f"{_JSON_STRIP_FIX}\nError: {exc}"},
                ]
            else:
                raise ValueError(
                    f"Recommendation agent returned invalid JSON after {_MAX_JSON_RETRIES} attempts: {exc}"
                ) from exc

    log.info("recommendation drafted", extra={
        "phase":      "recommendation_agent",
        "sku":        state["alert"]["sku"],
        "location":   state["alert"].get("location"),
        "quantity":   recommendation.get("quantity"),
        "supplier":   recommendation.get("supplier_name"),
        "confidence": recommendation.get("confidence"),
    })
    return {**state, "recommendation": recommendation}


# ---------------------------------------------------------------------------
# Node 5: audit_agent — schema and business-rule validation
# ---------------------------------------------------------------------------

async def audit_agent_node(state: AgentState) -> AgentState:
    """Validate the recommendation against schema and business rules.

    If invalid, the recommendation_agent node will be retried (up to 2 times)
    with the error list injected into its prompt.
    """
    rec    = state.get("recommendation", {})
    errors = validate_recommendation(rec, state)
    valid  = len(errors) == 0
    retries = state.get("audit_retries", 0)

    if valid:
        log.info("recommendation valid", extra={
            "phase":      "audit_agent",
            "sku":        state["alert"]["sku"],
            "location":   state["alert"].get("location"),
            "quantity":   rec.get("quantity"),
            "confidence": rec.get("confidence"),
        })
    else:
        log.warning("recommendation invalid", extra={
            "phase":    "audit_agent",
            "sku":      state["alert"]["sku"],
            "location": state["alert"].get("location"),
            "retry":    retries + 1,
            "errors":   errors,
        })

    return {
        **state,
        "audit_result":  {"valid": valid, "errors": errors},
        "audit_retries": retries + (0 if valid else 1),
    }


# ---------------------------------------------------------------------------
# Node 6: save_and_notify — write proposed_order, memories, update inventory
# ---------------------------------------------------------------------------

async def save_and_notify(state: AgentState) -> AgentState:
    """Write proposed_order, auto-approve if high confidence, write memories."""
    alert        = state["alert"]
    coverage_gap = state.get("coverage_gap")

    # Zero-gap fast path: route_by_urgency jumped here without running any LLM.
    # Build a synthetic zero-order recommendation so the rest of the function
    # works uniformly.
    rec = state.get("recommendation") or {}
    if not rec or coverage_gap == 0:
        existing = state.get("existing_order_qty", 0)
        reorder  = state.get("inventory", {}).get(
            "reorder_point", alert.get("reorder_point", 0)
        )
        on_hand  = state.get("inventory", {}).get(
            "on_hand", alert.get("on_hand", 0)
        )
        rec = {
            "supplier_id":   None,
            "supplier_name": "N/A",
            "quantity":      0,
            "rationale":     (
                f"Existing active orders ({existing} units) plus on-hand stock "
                f"({on_hand} units) already meet or exceed the reorder point "
                f"({reorder} units). No additional stock is required."
            ),
            "confidence":    "high",
        }

    chosen     = next(
        (s for s in state.get("suppliers", []) if s.get("supplier_id") == rec.get("supplier_id")),
        {},
    )
    quantity   = rec.get("quantity", 0)
    unit_price = chosen.get("unit_price", 0.0)
    lead_time  = chosen.get("lead_time_days", 0)
    confidence = rec.get("confidence", "medium")

    # Safety override: if existing orders already close the gap, force zero.
    if coverage_gap is not None and coverage_gap == 0:
        quantity = 0

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
        "retrieval_trace":        state.get("retrieval_trace", []),
        "status":                 status,
        "auto_approved":          auto_approved,
        "created_at":             datetime.now(timezone.utc),
        "alert_id":               ObjectId(alert["_id"]),
    }

    result   = await _db_async.proposed_orders.insert_one(order)
    order_id = str(result.inserted_id)

    # Record confidence outcome for calibration tracking
    await _db_async.confidence_outcomes.insert_one({
        "alert_id":              ObjectId(alert["_id"]),
        "order_id":              result.inserted_id,
        "predicted_confidence":  confidence,
        "auto_approved":         auto_approved,
        "human_decision":        None,
        "days_to_stock_recovery": None,
        "outcome":               "pending",
        "recorded_at":           datetime.now(timezone.utc),
    })

    # Record agent decision in lifecycle log
    loop_lc = asyncio.get_event_loop()
    loop_lc.run_in_executor(
        None, append_lifecycle_event,
        alert["_id"], "agent_decision",
        {
            "supplier":   rec.get("supplier_name"),
            "quantity":   quantity,
            "confidence": confidence,
            "auto_approved": auto_approved,
            "order_id":   order_id,
        },
        alert["sku"], alert["location"],
    )
    loop_lc.run_in_executor(
        None, append_lifecycle_event,
        alert["_id"], "order_placed",
        {"order_id": order_id, "status": status},
        alert["sku"], alert["location"],
    )

    await _db_async.reorder_alerts.update_one(
        {"_id": ObjectId(alert["_id"])},
        {"$set": {"status": "processed", "order_id": result.inserted_id}},
    )

    if auto_approved:
        await _db_async.inventory.update_one(
            {"sku": alert["sku"], "location": alert["location"]},
            {"$inc": {"on_order": quantity}},
        )

    days_remaining = alert.get("days_of_stock_remaining", 0)
    trend          = state.get("consumption", {}).get("trend", "stable")
    item_name      = state.get("inventory", {}).get("name", "")
    category       = state.get("inventory", {}).get("category", "unknown")
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
        loop.run_in_executor(
            None, write_order_history_sync,
            alert["sku"], alert["location"], category,
            rec.get("supplier_id"), rec.get("supplier_name", "N/A"),
            quantity, unit_price, total_cost,
            days_remaining, trend,
            rec.get("rationale", ""), result.inserted_id,
        ),
    )

    log.info("order saved", extra={
        "phase":                 "save_and_notify",
        "sku":                   alert["sku"],
        "location":              alert.get("location"),
        "order_id":              order_id,
        "quantity":              quantity,
        "total_cost":            order["total_cost"],
        "status":                status,
        "auto_approved":         auto_approved,
        "zero_order":            zero_order,
        "similar_orders_stored": len(clean_similar),
    })
    return {**state, "order_id": order_id}


# ---------------------------------------------------------------------------
# Node: escalate_alert — persist escalation record and notify webhook
# ---------------------------------------------------------------------------

async def _notify_escalation_webhook(url: str, sku: str, alert: dict, reason: str) -> None:
    """POST a JSON escalation notification to the configured webhook URL."""
    payload = json.dumps({
        "sku":          sku,
        "location":     alert.get("location", ""),
        "reason":       reason,
        "escalated_at": datetime.now(timezone.utc).isoformat(),
    }).encode()

    def _post() -> None:
        import urllib.request
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _post)
        print(f"  [escalate] Webhook notified for {sku}")
    except Exception as exc:
        print(f"  [escalate] Webhook notification failed: {exc}")


async def escalate_alert(state: AgentState) -> AgentState:
    """Write an escalation record after max audit retries are exhausted.

    Triggered by route_after_audit when audit_retries >= 2 and the
    recommendation is still invalid.  Writes to escalation_queue, marks the
    alert as 'escalated', and optionally POSTs a webhook notification.
    """
    alert  = state["alert"]
    sku    = alert.get("sku", "?")
    reason = "audit_max_retries"

    alert_oid = ObjectId(alert["_id"]) if isinstance(alert["_id"], str) else alert["_id"]

    lifecycle_doc = _db_sync.alert_lifecycle.find_one({"alert_id": str(alert.get("_id", ""))})
    rejection_history = []
    if lifecycle_doc:
        rejection_history = [
            e for e in lifecycle_doc.get("events", [])
            if e.get("type") == "human_rejected"
        ]

    await _db_async.escalation_queue.update_one(
        {"alert_id": alert_oid},
        {"$set": {
            "alert_id":            alert_oid,
            "sku":                 sku,
            "location":            alert.get("location", ""),
            "rejection_count":     alert.get("rejection_count", 0),
            "escalated_at":        datetime.now(timezone.utc),
            "rejection_history":   rejection_history,
            "last_recommendation": state.get("recommendation", {}),
            "last_audit_errors":   state.get("audit_result", {}).get("errors", []),
            "escalation_reason":   reason,
        }},
        upsert=True,
    )
    await _db_async.reorder_alerts.update_one(
        {"_id": alert_oid},
        {"$set": {"status": "escalated"}},
    )

    webhook_url = os.getenv("ESCALATION_WEBHOOK_URL")
    if webhook_url:
        await _notify_escalation_webhook(webhook_url, sku, alert, reason)

    log.warning("alert escalated", extra={
        "phase":    "escalate",
        "sku":      sku,
        "location": alert.get("location"),
        "reason":   reason,
    })
    return {**state}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_by_urgency(state: AgentState) -> str:
    """Short-circuit to save_and_notify when coverage_gap is already zero."""
    if state.get("coverage_gap") == 0:
        return "save_and_notify"
    return "retrieval_agent"


def route_after_audit(state: AgentState) -> str:
    """Route after audit: valid → save, max retries → escalate, else → retry."""
    if state.get("audit_result", {}).get("valid"):
        return "save_and_notify"
    if state.get("audit_retries", 0) >= 2:
        log.warning("audit max retries reached, escalating", extra={
            "phase": "audit_agent",
            "sku":   state["alert"]["sku"],
        })
        return "escalate"
    return "recommendation_agent"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("assess_alert",         assess_alert)
    builder.add_node("retrieval_agent",      retrieval_agent_node)
    builder.add_node("analysis_agent",       analysis_agent_node)
    builder.add_node("recommendation_agent", recommendation_agent_node)
    builder.add_node("audit_agent",          audit_agent_node)
    builder.add_node("escalate",             escalate_alert)
    builder.add_node("save_and_notify",      save_and_notify)

    builder.add_edge(START, "assess_alert")
    builder.add_conditional_edges(
        "assess_alert",
        route_by_urgency,
        {"save_and_notify": "save_and_notify", "retrieval_agent": "retrieval_agent"},
    )
    builder.add_edge("retrieval_agent",      "analysis_agent")
    builder.add_edge("analysis_agent",       "recommendation_agent")
    builder.add_edge("recommendation_agent", "audit_agent")
    builder.add_conditional_edges(
        "audit_agent",
        route_after_audit,
        {
            "save_and_notify":      "save_and_notify",
            "recommendation_agent": "recommendation_agent",
            "escalate":             "escalate",
        },
    )
    builder.add_edge("escalate",      END)
    builder.add_edge("save_and_notify", END)

    checkpointer = MongoDBSaver(_sync_client, db_name="supply_chain_demo")
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


# ---------------------------------------------------------------------------
# Change Stream listener — resume token persistence + crash recovery
# ---------------------------------------------------------------------------

_AGENT_STATE_COLLECTION = "agent_state"
_RESUME_TOKEN_DOC_ID    = "change_stream_resume_token"


def _load_resume_token() -> dict | None:
    doc = _db_sync[_AGENT_STATE_COLLECTION].find_one({"_id": _RESUME_TOKEN_DOC_ID})
    if doc and "token" in doc:
        log.info("resuming Change Stream from saved token")
        return doc["token"]
    return None


def _save_resume_token(token: dict) -> None:
    _db_sync[_AGENT_STATE_COLLECTION].update_one(
        {"_id": _RESUME_TOKEN_DOC_ID},
        {"$set": {"token": token, "saved_at": datetime.now(timezone.utc)}},
        upsert=True,
    )


async def _process_one_alert(alert_doc: dict) -> None:
    """Invoke the agent graph for a single alert document.

    Pre-graph checks (in order):
    1. LLM circuit breaker open → route to human_review_queue.
    2. rejection_count >= 3 → escalate directly without re-running the LLM.
    3. Atomic alert claim (pending → processing) — prevents two workers from
       racing to handle the same alert document.
    4. SKU-level distributed lock — prevents concurrent pipelines for the same
       (sku, location), closing the TOCTOU gap between assess_alert reading the
       coverage gap and save_and_notify writing the order.
    """
    sku       = alert_doc.get("sku", "?")
    location  = alert_doc.get("location", "")
    thread_id = str(alert_doc["_id"])
    alert_id  = thread_id

    # ── 1. Circuit breaker ────────────────────────────────────────────────────
    if _circuit_failures >= _CIRCUIT_THRESHOLD:
        log.error("circuit breaker open, routing to human review queue", extra={
            "phase": "process_alert", "sku": sku,
        })
        _db_sync.human_review_queue.update_one(
            {"alert_id": ObjectId(alert_doc["_id"])},
            {"$set": {
                "alert_id":       ObjectId(alert_doc["_id"]),
                "sku":            sku,
                "location":       location,
                "reason":         "llm_circuit_open",
                "queued_at":      datetime.now(timezone.utc),
                "alert_snapshot": alert_doc,
            }},
            upsert=True,
        )
        return

    # ── 2. Rejection escalation ───────────────────────────────────────────────
    rejection_count = alert_doc.get("rejection_count", 0)
    if rejection_count >= 3:
        log.warning("too many human rejections, escalating directly", extra={
            "phase": "process_alert", "sku": sku, "rejection_count": rejection_count,
        })
        alert_oid = ObjectId(alert_doc["_id"])
        await _db_async.escalation_queue.update_one(
            {"alert_id": alert_oid},
            {"$set": {
                "alert_id":          alert_oid,
                "sku":               sku,
                "location":          location,
                "rejection_count":   rejection_count,
                "escalated_at":      datetime.now(timezone.utc),
                "escalation_reason": "human_rejections",
            }},
            upsert=True,
        )
        await _db_async.reorder_alerts.update_one(
            {"_id": alert_oid},
            {"$set": {"status": "escalated"}},
        )
        webhook_url = os.getenv("ESCALATION_WEBHOOK_URL")
        if webhook_url:
            await _notify_escalation_webhook(webhook_url, sku, alert_doc, "human_rejections")
        return

    # ── 3. Atomic alert claim (pending → processing) ──────────────────────────
    # Only one worker can win this update. If claimed is None, another worker
    # already grabbed this alert (or it was already processed).
    claimed = await _db_async.reorder_alerts.find_one_and_update(
        {"_id": ObjectId(alert_doc["_id"]), "status": "pending"},
        {"$set": {"status": "processing"}},
    )
    if claimed is None:
        log.info("alert already claimed by another worker, skipping", extra={
            "phase": "process_alert", "sku": sku, "alert_id": alert_id,
        })
        return

    # ── 4. SKU-level distributed lock ─────────────────────────────────────────
    # Retry with exponential backoff (1 s → 2 s → 4 s → 8 s → 16 s → 16 s)
    # up to ~47 s total. A concurrent pipeline for this SKU typically finishes
    # in < 30 s (one LLM round-trip), so this window is sufficient.
    lock_acquired = False
    for attempt in range(6):
        lock_acquired = await sku_lock.acquire(_db_async, sku, location, alert_id)
        if lock_acquired:
            break
        wait = min(2 ** attempt, 16)
        log.info("sku lock contention, retrying", extra={
            "phase": "process_alert", "sku": sku,
            "attempt": attempt + 1, "wait_s": wait,
        })
        await asyncio.sleep(wait)

    if not lock_acquired:
        log.warning("could not acquire sku lock after retries, deferring alert", extra={
            "phase": "process_alert", "sku": sku, "alert_id": alert_id,
        })
        # Reset to pending so the Change Stream re-fires this alert.
        await _db_async.reorder_alerts.update_one(
            {"_id": ObjectId(alert_doc["_id"])},
            {"$set": {"status": "pending"}},
        )
        return

    # ── 5. Invoke the agent graph ─────────────────────────────────────────────
    try:
        alert_doc = _serialize_doc(alert_doc)
        config    = {"configurable": {"thread_id": thread_id}}
        await graph.ainvoke({"alert": alert_doc}, config=config)
        log.info("alert processed", extra={"phase": "process_alert", "sku": sku})
    finally:
        await sku_lock.release(_db_async, sku, location, alert_id)


async def _drain_existing_pending_alerts() -> None:
    """Process any pending alerts that existed before the Change Stream opened.

    Change Streams only deliver events that occur after the cursor is opened, so
    alerts inserted by seed.py (or a prior agent crash) are invisible to the stream.
    This scan runs once on startup to close that gap.  The atomic pending→processing
    claim inside _process_one_alert prevents any alert from being double-processed.
    """
    cursor = _db_async.reorder_alerts.find({"status": "pending"})
    count = 0
    async for alert_doc in cursor:
        sku = alert_doc.get("sku", "?")
        log.info("startup: processing pre-existing pending alert", extra={"sku": sku})
        try:
            await _process_one_alert(alert_doc)
            count += 1
        except Exception as exc:  # noqa: BLE001
            log.error("startup: error processing pre-existing alert",
                       extra={"sku": sku, "error": str(exc)})
    if count:
        log.info("startup: drained pre-existing pending alerts", extra={"count": count})


async def watch_alerts() -> None:
    """Watch reorder_alerts via Change Stream and invoke the agent graph.

    Resilience: resume token persistence + exponential-backoff reconnection
    + InvalidateError handling for demo resets (collection drop).
    """
    pipeline = [{"$match": {"$or": [
        {"operationType": "insert"},
        {
            "operationType": "update",
            "updateDescription.updatedFields.status": "pending",
        },
    ]}}]
    retry_delay = 2

    await sku_lock.ensure_indexes(_db_async)
    log.info("sku_processing_locks indexes ensured")

    await _drain_existing_pending_alerts()

    while True:
        resume_token = _load_resume_token()
        watch_kwargs = dict(full_document="updateLookup")
        if resume_token:
            watch_kwargs["resume_after"] = resume_token

        log.info("watching for reorder alerts via Change Stream")
        try:
            async with _db_async.reorder_alerts.watch(pipeline, **watch_kwargs) as stream:
                retry_delay = 2
                async for change in stream:
                    _save_resume_token(change["_id"])

                    alert_doc = change.get("fullDocument")
                    if not alert_doc or alert_doc.get("status") != "pending":
                        continue

                    sku = alert_doc.get("sku", "?")
                    log.info("processing alert", extra={"sku": sku})

                    _, errors = validate_alert(alert_doc)
                    if errors:
                        log.error("alert validation failed", extra={"sku": sku, "errors": errors})
                        write_dead_letter(
                            _db_sync, "change_stream", "reorder_alert",
                            alert_doc, errors,
                        )
                        continue

                    try:
                        await _process_one_alert(alert_doc)
                    except Exception as exc:  # noqa: BLE001
                        log.error("error processing alert", extra={"sku": sku, "error": str(exc)})

        except Exception as exc:  # noqa: BLE001
            err_type = type(exc).__name__
            log.error("Change Stream interrupted, reconnecting", extra={
                "error_type":  err_type,
                "error":       str(exc),
                "retry_delay": retry_delay,
            })
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)
            if "InvalidateError" in err_type or "CursorNotFound" in err_type:
                _db_sync[_AGENT_STATE_COLLECTION].delete_one({"_id": _RESUME_TOKEN_DOC_ID})
                log.info("cleared stale resume token, stream will start fresh")


if __name__ == "__main__":
    asyncio.run(watch_alerts())
