"""
LangGraph agent — supply chain reorder pipeline.

Graph topology (5 nodes):
    START
      → assess_alert     (DB-only: inventory position + coverage gap)
      → route_by_urgency (conditional: zero-gap fast path vs recommend)
          ↘ save_order   (zero-gap fast path — no LLM needed)
          ↘ recommend    (retrieval + analysis + recommendation + validation)
              → save_order  (write order; interrupt() for human review)
                  → write_memories  (short-term, long-term, history)
      → escalate         (validation failed — writes escalation record)
      → END

Node responsibilities:
  assess_alert   — read MongoDB, compute coverage gap
  recommend      — ReAct tool loop → LLM analysis → LLM recommendation → validate
  save_order     — write proposed_order; interrupt() for human review
  write_memories — persist decision to all three memory layers
  escalate       — write escalation record if recommend cannot produce a valid order

Explain mode (EXPLAIN_MODE=1):
  Each node prints a plain-English description of what it is doing.
  Useful for learning and live demos without reading JSON logs.

ReAct retrieval tools (called inside recommend):
  1. get_inventory_position           (basic find)
  2. get_supplier_options             (basic find, fill-rate sorted)
  3. get_consumption_trend            (time series aggregation)
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
from datetime import datetime, timedelta, timezone
from typing import TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bson import ObjectId
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent  # noqa: F401 — still works; langchain.agents.create_react_agent has a different signature
from langgraph.types import interrupt, Command  # noqa: F401 — Command re-exported for app.py

try:
    from langgraph.checkpoint.mongodb import MongoDBSaver
except ImportError:  # optional in offline unit-test environments
    MongoDBSaver = None  # type: ignore[assignment]

from agent import sku_lock
from agent.db import sync_client as _sync_client, db_sync as _db_sync, db_async as _db_async
from agent.context_manifest import build_context_manifest
from agent.logger import get_logger
from agent.order_policy import compute_order_decision
from agent.alerts import validate_alert_document
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
# Business-rule thresholds
# ---------------------------------------------------------------------------
# Orders at or above this cost always route to human review regardless of
# confidence level.  A procurement specialist, not the agent, approves high-
# value orders.  Adjust agent/order_policy.py to change this policy.

# Alerts claimed as processing are recovered on startup if the worker dies before
# completing the graph. This is intentionally longer than a typical LLM run.
_STALE_PROCESSING_AFTER_SECONDS = 15 * 60

# Categories that require an FDA-registered wholesale distributor.
# Selecting a non-FDA supplier for these categories is a hard compliance
# violation caught by inline recommendation validation.
_FDA_REQUIRED_CATEGORIES = frozenset({"pharmaceutical", "laboratory"})

# ---------------------------------------------------------------------------
# Explain mode — set EXPLAIN_MODE=1 to see plain-English node descriptions
# ---------------------------------------------------------------------------
# Each agent node calls _explain() at the start and (optionally) end of its
# work.  This is a teaching aid: beginners can run with EXPLAIN_MODE=1 and
# see exactly what the graph is doing without reading the JSON structured logs.
#
# Usage:
#   EXPLAIN_MODE=1 python agent/graph.py
# ---------------------------------------------------------------------------
_EXPLAIN = os.getenv("EXPLAIN_MODE", "").lower() in {"1", "true", "yes"}


def _explain(msg: str) -> None:
    """Print a plain-English description of what the agent is doing right now.

    Enabled when EXPLAIN_MODE=1 is set in the environment.  Messages are
    flushed immediately so they appear in real time alongside the JSON logs.
    """
    if _EXPLAIN:
        print(f"\n  💡 [EXPLAIN] {msg}", flush=True)

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
_CIRCUIT_DOC_ID: str = "llm_circuit_breaker"


class CircuitOpenError(RuntimeError):
    """Raised when the LLM circuit breaker is open."""


def _get_circuit_failures() -> int:
    """Read circuit-breaker failures from MongoDB, falling back to local memory."""
    global _circuit_failures
    try:
        doc = _db_sync[_AGENT_STATE_COLLECTION].find_one({"_id": _CIRCUIT_DOC_ID})
        if doc and "failures" in doc:
            _circuit_failures = int(doc.get("failures", 0))
    except Exception as exc:
        log.warning("could not read circuit breaker state", extra={"error": str(exc)})
    return _circuit_failures


def _set_circuit_failures(count: int) -> None:
    """Persist circuit-breaker failures so dashboard and agent processes agree."""
    global _circuit_failures
    _circuit_failures = max(0, int(count))
    try:
        _db_sync[_AGENT_STATE_COLLECTION].update_one(
            {"_id": _CIRCUIT_DOC_ID},
            {"$set": {"failures": _circuit_failures, "updated_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
    except Exception as exc:
        log.warning("could not persist circuit breaker state", extra={"error": str(exc)})


def reset_circuit_breaker() -> None:
    """Reset the LLM circuit breaker across processes."""
    _set_circuit_failures(0)


def _get_llm() -> ChatOpenAI:
    """Return the active LLM, raising CircuitOpenError if the circuit is open."""
    failures = _get_circuit_failures()
    if failures >= _CIRCUIT_THRESHOLD:
        raise CircuitOpenError(
            f"LLM circuit breaker open after {failures} consecutive failures — "
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
        _set_circuit_failures(0)   # reset on success
        return response
    except CircuitOpenError:
        raise
    except Exception as primary_exc:
        log.warning("primary LLM failed, trying fallback", extra={"error": str(primary_exc)})
        try:
            response = await _llm_fallback.ainvoke(messages)
            _set_circuit_failures(0)
            return response
        except Exception as fallback_exc:
            _set_circuit_failures(_get_circuit_failures() + 1)
            log.error(
                "fallback LLM also failed",
                extra={
                    "error":               str(fallback_exc),
                    "consecutive_failures": _get_circuit_failures(),
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

    # ── Retrieval fields (populated by recommend node) ───────────────────
    suppliers:               list   # get_supplier_options result
    consumption:             dict   # get_consumption_trend result
    supplier_search_results: list   # Atlas Search result
    similar_orders:          list   # Atlas Vector Search on order_history
    short_term_memories:     list   # Per-SKU decisions in last 24 h
    long_term_memories:      list   # Semantic memories from agent_memory
    retrieval_results:       dict   # Raw tool results keyed by tool name
    retrieval_trace:         list   # Ordered list of {tool, args} dicts
    context_budget_report:   dict   # Token budget report for recommendation context

    # ── Recommend fields (populated by recommend node) ────────────────────
    analysis:                dict   # {best_supplier_id, confidence, risk_flags, reasoning_trace}
    recommendation:          dict   # {supplier_id, supplier_name, quantity, rationale, confidence}
    escalate_flag:           bool   # True when recommend exhausts retries — routes to escalate

    # ── Persistence fields (populated by save_order / write_memories) ────────
    order_id:             str    # _id of the saved proposed_order
    human_approved:       bool   # True when the order was ultimately approved
    final_recommendation: dict   # The rec dict actually used (LLM-generated or synthetic zero-order)
    decision_source:      str    # "agent" for auto-approval, "human" for HITL decisions
    human_decision:       str    # "approved" / "rejected" for HITL decisions
    requeue_alert:        bool   # True when a rejected alert should be retried after memory writes


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

    _explain(
        f"[assess_alert] Checking inventory for {alert['sku']} @ {alert['location']}. "
        "I'll look at on-hand stock and any existing orders to compute how many units "
        "we still need to reach the reorder point (the 'coverage gap')."
    )

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

    _explain(
        f"[assess_alert] Result: on-hand={on_hand}, on-order={existing_order_qty}, "
        f"reorder point={reorder_point} → coverage gap={coverage_gap} units. "
        + ("⚡ EXPEDITE — less than 2 days of stock remaining!" if expedite else "Stock is not critical.")
        + (" Zero gap: skipping LLM, jumping straight to save_order." if coverage_gap == 0 else "")
    )

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
    # Return only the keys this node computed — LangGraph merges partial
    # dicts with the existing state via per-channel reducers.
    return {
        "inventory":          inv,
        "existing_order_qty": existing_order_qty,
        "coverage_gap":       coverage_gap,
        "expedite":           expedite,
    }


# ---------------------------------------------------------------------------
# Node 2: recommend — retrieval → analysis → recommendation → validation
# ---------------------------------------------------------------------------
#
# Retrieval, analysis, recommendation, and validation are merged here so beginners
# see a single, readable step:
#   Step 1 — ReAct tool-calling loop (gather suppliers, trends, memory)
#   Step 2 — LLM analysis            (rank suppliers, assign confidence)
#   Step 3 — LLM recommendation      (calculate quantity, write rationale)
#   Step 4 — Inline validation        (Pydantic rules + FDA compliance; retries internally)
# ---------------------------------------------------------------------------

async def recommend(state: AgentState) -> dict:
    """Retrieval → Analysis → Recommendation → Validation in a single node.

    Step 1 — ReAct tool-calling loop
        The LLM decides which tools to call (Atlas Search, Vector Search,
        time series, memory) until it has enough context.
    Step 2 — LLM analysis
        Rank suppliers by fill rate, lead time, FDA compliance, and cost.
        Assign a confidence score: high / medium / low.
    Step 3 — LLM recommendation
        Calculate order quantity, select the best supplier, write rationale.
    Step 4 — Inline validation
        Pydantic schema check + FDA / budget rules.  Retries internally
        (up to _MAX_JSON_RETRIES times) before escalating.
    """
    alert     = state["alert"]
    inventory = state.get("inventory", {})
    days      = alert.get("days_of_stock_remaining", 99)
    gap       = state.get("coverage_gap", 0)
    expedite  = state.get("expedite", False)

    urgency_label = "CRITICAL" if days < 2 else "ELEVATED" if days < 5 else "STANDARD"

    # ── Step 1: Retrieval (ReAct tool-calling loop) ──────────────────────────
    _explain(
        f"[recommend] Step 1/3 — ReAct retrieval for {alert['sku']} (urgency={urgency_label}). "
        "The LLM picks tools: supplier search (Atlas Search), similar orders (Vector Search), "
        "consumption trend, short-term and long-term memory."
    )

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
            {"messages": [HumanMessage(content=RETRIEVAL_SYSTEM_PROMPT + "\n\n" + context)]},
            config={"recursion_limit": 20},
        )
        retrieval_results, retrieval_trace = _extract_retrieval_results(result["messages"])
    except Exception as exc:
        log.warning("ReAct agent failed, falling back to direct calls", extra={
            "phase": "recommend", "sku": alert.get("sku"), "error": str(exc),
        })
        retrieval_results, retrieval_trace = {}, []

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

    if long_term_mems:
        long_term_mems = [
            m for m in long_term_mems
            if not isinstance(m, dict)
            or "error" in m
            or "info" in m
            or m.get("location") in (None, alert["location"])
        ]
    else:
        long_term_mems = get_long_term_memories(
            context,
            location=alert["location"],
        )

    def _count(results: list) -> int:
        return len([r for r in results if "error" not in r and "info" not in r])

    _explain(
        f"[recommend] Retrieval done — {len(retrieval_trace)} tool call(s). "
        f"{len(suppliers)} suppliers · {_count(similar_orders)} similar past orders "
        f"· trend={consumption.get('trend', '?')}."
    )
    log.info("retrieval complete", extra={
        "phase": "recommend", "sku": alert["sku"], "location": alert.get("location"),
        "tool_calls": len(retrieval_trace), "suppliers": len(suppliers),
        "avg_daily": consumption.get("avg_daily"), "trend": consumption.get("trend", "?"),
        "atlas_hits": _count(supplier_search), "vector_hits": _count(similar_orders),
    })

    # Build a local context dict for prompt builders (not written to state yet)
    ctx = {
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

    # ── Step 2: Analysis ─────────────────────────────────────────────────────
    _explain(
        f"[recommend] Step 2/3 — LLM evaluating {len(suppliers)} supplier(s): "
        "fill rate · lead time · FDA compliance · cost → confidence score."
    )

    analysis_messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user",   "content": build_analysis_prompt(ctx)},
    ]
    analysis = None
    for attempt in range(_MAX_JSON_RETRIES):
        response = await _llm_invoke(analysis_messages)
        try:
            analysis = _parse_llm_json(response.content)
            break
        except json.JSONDecodeError as exc:
            if attempt < _MAX_JSON_RETRIES - 1:
                analysis_messages += [
                    {"role": "assistant", "content": response.content},
                    {"role": "user",      "content": f"{_JSON_STRIP_FIX}\nError: {exc}"},
                ]
            else:
                raise ValueError(f"Analysis returned invalid JSON after {_MAX_JSON_RETRIES} attempts: {exc}") from exc

    _explain(
        f"[recommend] Best supplier: {analysis.get('best_supplier_name', '?')}, "
        f"confidence={analysis.get('confidence', '?')}. "
        f"Risk flags: {analysis.get('risk_flags', []) or 'none'}."
    )
    ctx = {**ctx, "analysis": analysis}

    # ── Step 3: Recommendation + inline validation ───────────────────────────
    _explain("[recommend] Step 3/3 — LLM calculating order quantity and writing rationale.")

    recommendation: dict = {}
    prior_errors: list[str] = []
    escalate_flag = False
    context_budget_report: dict = {}

    for attempt in range(_MAX_JSON_RETRIES):
        prompt_state = {**ctx, "audit_result": {"errors": prior_errors}}
        rec_prompt = build_recommendation_prompt(prompt_state)
        context_budget_report = prompt_state.get("_context_budget_report", context_budget_report)
        if prior_errors:
            rec_prompt += (
                "\n\nPREVIOUS VALIDATION ERRORS — fix all of these:\n"
                + "\n".join(f"  - {e}" for e in prior_errors)
            )
        rec_messages = [
            {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
            {"role": "user",   "content": rec_prompt},
        ]
        for json_attempt in range(_MAX_JSON_RETRIES):
            response = await _llm_invoke(rec_messages)
            try:
                recommendation = _parse_llm_json(response.content)
                break
            except json.JSONDecodeError as exc:
                if json_attempt < _MAX_JSON_RETRIES - 1:
                    rec_messages += [
                        {"role": "assistant", "content": response.content},
                        {"role": "user",      "content": f"{_JSON_STRIP_FIX}\nError: {exc}"},
                    ]
                else:
                    raise ValueError(f"Recommendation returned invalid JSON: {exc}") from exc

        errors = validate_recommendation(recommendation, ctx)
        if not errors:
            _explain(f"[recommend] ✅ Validation passed — handing off to save_order.")
            break
        prior_errors = errors
        _explain(f"[recommend] ❌ Validation failed ({len(errors)} error(s)): {errors}. "
                 + ("Escalating." if attempt + 1 >= _MAX_JSON_RETRIES else "Retrying …"))
        log.warning("validation failed", extra={
            "phase": "recommend", "sku": alert["sku"], "attempt": attempt + 1, "errors": errors,
        })
        if attempt + 1 >= _MAX_JSON_RETRIES:
            escalate_flag = True

    log.info("recommend complete", extra={
        "phase": "recommend", "sku": alert["sku"], "location": alert.get("location"),
        "supplier": recommendation.get("supplier_name"),
        "quantity": recommendation.get("quantity"),
        "confidence": recommendation.get("confidence"),
        "escalate": escalate_flag,
    })

    return {
        "suppliers":               suppliers,
        "consumption":             consumption,
        "supplier_search_results": supplier_search,
        "similar_orders":          similar_orders,
        "short_term_memories":     short_term_mems,
        "long_term_memories":      long_term_mems,
        "retrieval_results":       retrieval_results,
        "retrieval_trace":         retrieval_trace,
        "context_budget_report":   context_budget_report,
        "analysis":                analysis,
        "recommendation":          recommendation,
        "escalate_flag":           escalate_flag,
    }


# ---------------------------------------------------------------------------
# Node 3: save_order — write proposed_order, handle auto-approve / interrupt
# Node 4: write_memories — persist decision to all three memory layers
# ---------------------------------------------------------------------------

async def save_order(state: AgentState) -> dict:
    """Node 6a — Write the proposed order, then auto-approve or pause for human review.

    Responsibilities (single):
      • Insert the proposed_order document into MongoDB.
      • If confidence is high and cost < ceiling → auto-approve immediately.
      • Otherwise → call interrupt() to pause the graph and wait for a human.

    # ── How interrupt() works ────────────────────────────────────────────────
    # interrupt() is a LangGraph primitive.  When called:
    #   1. LangGraph serialises the full graph state and writes it to the
    #      `checkpoints` collection via MongoDBSaver — the graph is now FROZEN.
    #   2. graph.ainvoke() returns to the caller (the Change Stream handler).
    #   3. The Streamlit dashboard detects `status: awaiting_approval` in MongoDB
    #      and shows the review card to the procurement specialist.
    #   4. On button click, the dashboard calls:
    #        graph.invoke(Command(resume={"approved": True, "approver": "jsmith"}),
    #                     config={"configurable": {"thread_id": alert_id}})
    #   5. LangGraph reloads the checkpoint from MongoDB and re-enters THIS node
    #      right after the interrupt() call — execution continues with the human
    #      decision available as a plain Python dict.
    #
    # Open MongoDB Atlas → `checkpoints` collection while the graph is paused.
    # You will see the full agent state frozen in a document — that is the
    # entire LangGraph + MongoDBSaver story visible in one Atlas query.
    # ────────────────────────────────────────────────────────────────────────

    Returns (partial dict — LangGraph merges into state via reducers):
        order_id             — MongoDB _id of the inserted proposed_order
        human_approved       — True if auto-approved or human said yes
        final_recommendation — the rec dict used (LLM-generated or synthetic)
    """
    alert        = state["alert"]
    coverage_gap = state.get("coverage_gap")

    # ── Build recommendation and approval policy ─────────────────────────────
    # Zero-gap fast path: route_by_urgency jumped here without running any LLM.
    # compute_order_decision() synthesises a zero-quantity rec and applies the
    # same approval policy used by tests and docs.
    decision = compute_order_decision(state)
    rec           = decision["recommendation"]
    quantity      = decision["quantity"]
    unit_price    = decision["unit_price"]
    lead_time     = decision["lead_time"]
    confidence    = decision["confidence"]
    total_cost    = decision["total_cost"]
    zero_order    = decision["zero_order"]
    auto_approved = decision["auto_approved"]
    review_reason = decision["review_reason"]

    clean_similar = [
        {k: v for k, v in o.items() if k != "embedding"}
        for o in state.get("similar_orders", [])
        if "error" not in o and "info" not in o
    ]

    _explain(
        f"Writing proposed order for {alert['sku']} @ {alert['location']} — "
        f"{quantity:,} units from {rec.get('supplier_name', 'N/A')}, "
        f"${total_cost:,.2f}, confidence={confidence}. "
        + ("Auto-approved ✅" if auto_approved else f"Needs human review ⏸  ({review_reason})")
    )

    # ── Insert proposed_order ────────────────────────────────────────────────
    alert_oid = ObjectId(alert["_id"]) if isinstance(alert["_id"], str) else alert["_id"]
    context_manifest = build_context_manifest(state, rec)
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
        "atlas_search_used":      len([r for r in state.get("supplier_search_results", []) if "error" not in r]) > 0,
        "retrieval_trace":        state.get("retrieval_trace", []),
        "context_manifest":      context_manifest,
        "status":                 "approved" if auto_approved else "awaiting_approval",
        "auto_approved":          auto_approved,
        "review_reason":          review_reason,
        "created_at":             datetime.now(timezone.utc),
        "alert_id":               alert_oid,
    }

    loop = asyncio.get_running_loop()

    existing_order = await _db_async.proposed_orders.find_one(
        {
            "alert_id": alert_oid,
            "status": {"$in": ["awaiting_approval", "approved"]},
        },
        sort=[("created_at", -1)],
    )

    existing_human_approved = False

    if existing_order:
        # LangGraph re-runs the node body when resuming after interrupt(). Any
        # writes before interrupt() must therefore be idempotent. Reuse the
        # already-created review order instead of inserting duplicates.
        order = existing_order
        order_id = existing_order["_id"]
        rec = {
            "supplier_id":   existing_order.get("supplier_id"),
            "supplier_name": existing_order.get("supplier_name"),
            "quantity":      existing_order.get("quantity_recommended", 0),
            "rationale":     existing_order.get("rationale", ""),
            "confidence":    existing_order.get("confidence", "medium"),
        }
        quantity      = existing_order.get("quantity_recommended", quantity)
        unit_price    = existing_order.get("unit_price", unit_price)
        total_cost    = existing_order.get("total_cost", total_cost)
        confidence    = existing_order.get("confidence", confidence)
        auto_approved = existing_order.get("auto_approved", auto_approved)
        review_reason = existing_order.get("review_reason", review_reason)

        existing_human_approved = existing_order.get("status") == "approved" and not auto_approved
    else:
        result   = await _db_async.proposed_orders.insert_one(order)
        order_id = result.inserted_id

        await _db_async.confidence_outcomes.update_one(
            {"alert_id": alert_oid, "order_id": order_id},
            {"$setOnInsert": {
                "alert_id":               alert_oid,
                "order_id":               order_id,
                "predicted_confidence":   confidence,
                "auto_approved":          auto_approved,
                "human_decision":         None,
                "days_to_stock_recovery": None,
                "outcome":                "pending",
                "recorded_at":            datetime.now(timezone.utc),
            }},
            upsert=True,
        )

        loop.run_in_executor(
            None, append_lifecycle_event,
            alert["_id"], "agent_decision",
            {"supplier": rec.get("supplier_name"), "quantity": quantity,
             "confidence": confidence, "auto_approved": auto_approved, "order_id": str(order_id)},
            alert["sku"], alert["location"],
        )

    async def _apply_approved_order_side_effects(approved_by: str | None = None) -> bool:
        """Apply approval side effects exactly once for this order.

        The order carries `inventory_applied_at` as the idempotency marker. The
        marker, order status, inventory increment, confidence outcome, and alert
        completion commit in one MongoDB transaction.
        """
        now = datetime.now(timezone.utc)
        set_fields = {"status": "approved", "inventory_applied_at": now}
        if approved_by:
            set_fields.update({"approved_by": approved_by, "approved_at": now})

        applied = False
        async with await _db_async.client.start_session() as session:
            async with session.start_transaction():
                apply_update = await _db_async.proposed_orders.update_one(
                    {
                        "_id": order_id,
                        "status": {"$in": ["awaiting_approval", "approved"]},
                        "inventory_applied_at": {"$exists": False},
                    },
                    {"$set": set_fields},
                    session=session,
                )
                applied = apply_update.modified_count == 1

                current_order = await _db_async.proposed_orders.find_one(
                    {"_id": order_id}, {"status": 1}, session=session,
                )
                accepted = applied or (current_order and current_order.get("status") == "approved")

                if applied:
                    await _db_async.inventory.update_one(
                        {"sku": alert["sku"], "location": alert["location"]},
                        {"$inc": {"on_order": quantity}},
                        session=session,
                    )
                    await _db_async.confidence_outcomes.update_one(
                        {"order_id": order_id},
                        {"$set": {
                            "human_decision": "approved" if approved_by else None,
                            "outcome": "resolved",
                        }},
                        session=session,
                    )

                if accepted:
                    await _db_async.reorder_alerts.update_one(
                        {"_id": alert_oid},
                        {"$set": {"status": "processed", "order_id": order_id}},
                        session=session,
                    )

        return applied

    if existing_human_approved:
        approved_by = existing_order.get("approved_by", "human") if existing_order else "human"
        applied = await _apply_approved_order_side_effects(approved_by)
        if applied:
            loop.run_in_executor(
                None, append_lifecycle_event,
                alert["_id"], "human_approved",
                {"order_id": str(order_id), "approved_by": approved_by},
                alert["sku"], alert["location"],
            )
        log.info("human-approved order already processed, skipping duplicate resume", extra={
            "phase": "save_order", "sku": alert["sku"], "order_id": str(order_id),
        })
        return {
            "order_id":             str(order_id),
            "human_approved":       True,
            "final_recommendation": rec,
            "decision_source":      "human",
            "human_decision":       "approved",
            "requeue_alert":        False,
        }

    # ── Human-in-the-loop path ───────────────────────────────────────────────
    if not auto_approved:
        # Tag the alert so the Change Stream doesn't re-fire it while we wait.
        await _db_async.reorder_alerts.update_one(
            {"_id": alert_oid},
            {"$set": {"status": "awaiting_human_approval", "order_id": order_id}},
        )

        log.info("order awaiting human approval — pausing graph", extra={
            "phase": "save_order", "sku": alert["sku"],
            "order_id": str(order_id), "review_reason": review_reason,
        })

        # ── GRAPH PAUSES HERE ────────────────────────────────────────────────
        # LangGraph writes the full state to `checkpoints` (MongoDBSaver).
        # The Streamlit dashboard resumes with Command(resume={...}).
        # ─────────────────────────────────────────────────────────────────────
        human_decision: dict = interrupt({
            "order_id":       str(order_id),
            "sku":            alert["sku"],
            "location":       alert["location"],
            "supplier":       rec.get("supplier_name"),
            "quantity":       quantity,
            "total_cost_usd": total_cost,
            "review_reason":  review_reason,
        })
        # ── GRAPH RESUMES HERE (after dashboard calls Command(resume=...)) ───

        approved = human_decision.get("approved", False)
        approver = human_decision.get("approver", "human")

        if approved:
            applied = await _apply_approved_order_side_effects(approver)
            if applied:
                loop.run_in_executor(
                    None, append_lifecycle_event,
                    alert["_id"], "human_approved",
                    {"order_id": str(order_id), "approved_by": approver},
                    alert["sku"], alert["location"],
                )
            else:
                log.info("approval already applied, skipping inventory increment", extra={
                    "phase": "save_order", "sku": alert["sku"], "order_id": str(order_id),
                })
            log.info("human approved order", extra={
                "phase": "save_order", "sku": alert["sku"],
                "order_id": str(order_id), "approved_by": approver,
            })
        else:
            # Rejection is re-queued only after write_memories runs. This ensures
            # the next agent attempt sees the human rejection signal in memory.
            reason = human_decision.get("reason", "human_rejected")
            await _db_async.proposed_orders.update_one(
                {"_id": order_id}, {"$set": {"status": "rejected"}},
            )
            await _db_async.confidence_outcomes.update_one(
                {"order_id": order_id},
                {"$set": {"human_decision": "rejected", "outcome": "escalated"}},
            )
            loop.run_in_executor(
                None, append_lifecycle_event,
                alert["_id"], "human_rejected",
                {"order_id": str(order_id), "reason": reason},
                alert["sku"], alert["location"],
            )
            log.info("human rejected order, alert reset to pending", extra={
                "phase": "save_order", "sku": alert["sku"], "order_id": str(order_id),
            })

        _explain(
            f"Human decision received for {alert['sku']}: "
            + ("APPROVED ✅ — memories will now be written." if approved
               else "REJECTED ❌ — alert reset to pending for next agent run.")
        )

        return {
            "order_id":             str(order_id),
            "human_approved":       approved,
            "final_recommendation": rec,
            "decision_source":      "human",
            "human_decision":       "approved" if approved else "rejected",
            "requeue_alert":        not approved,
        }

    # ── Auto-approved path ───────────────────────────────────────────────────
    applied = await _apply_approved_order_side_effects()
    if applied:
        loop.run_in_executor(
            None, append_lifecycle_event,
            alert["_id"], "order_placed",
            {"order_id": str(order_id), "status": "approved"},
            alert["sku"], alert["location"],
        )
    else:
        log.info("approved order side effects already applied", extra={
            "phase": "save_order", "sku": alert["sku"], "order_id": str(order_id),
        })

    log.info("order auto-approved", extra={
        "phase":                 "save_order",
        "sku":                   alert["sku"],
        "location":              alert.get("location"),
        "order_id":              str(order_id),
        "quantity":              quantity,
        "total_cost":            total_cost,
        "zero_order":            zero_order,
        "similar_orders_stored": len(clean_similar),
    })
    return {
        "order_id":             str(order_id),
        "human_approved":       True,
        "final_recommendation": rec,
        "decision_source":      "agent",
        "human_decision":       None,
        "requeue_alert":        False,
    }


# ---------------------------------------------------------------------------
# Node 6b: write_memories — persist decision to all three memory layers
# ---------------------------------------------------------------------------

async def write_memories(state: AgentState) -> dict:
    """Node 6b — Write the agent's decision to short-term, long-term, and order-history memory.

    This node runs after save_order completes (whether auto-approved or
    human-reviewed).  Separating memory writes into their own node keeps each
    node focused on a single responsibility and makes the graph topology easier
    to read for beginners.

    Three memory layers written in parallel:
    ┌─────────────────────────┬─────────────────────────────────────────────┐
    │ Layer                   │ Purpose                                     │
    ├─────────────────────────┼─────────────────────────────────────────────┤
    │ short_term_memory       │ Exact decision for this SKU+location, 24 h  │
    │                         │ TTL.  Prevents the agent from re-ordering   │
    │                         │ the same item within the same day.          │
    ├─────────────────────────┼─────────────────────────────────────────────┤
    │ agent_memory            │ Semantic summary + Voyage AI embedding.     │
    │                         │ Retrieved later via Atlas Vector Search     │
    │                         │ (`get_learned_patterns` tool).              │
    ├─────────────────────────┼─────────────────────────────────────────────┤
    │ order_history           │ Full order record for future similarity     │
    │                         │ lookup (`find_similar_past_orders` tool).   │
    └─────────────────────────┴─────────────────────────────────────────────┘
    """
    alert    = state["alert"]
    rec      = state.get("final_recommendation") or {}
    approved = state.get("human_approved", True)
    decision_source = state.get("decision_source", "agent")
    human_decision = state.get("human_decision") if decision_source == "human" else None

    chosen     = next(
        (s for s in state.get("suppliers", []) if s.get("supplier_id") == rec.get("supplier_id")),
        {},
    )
    quantity   = rec.get("quantity", 0)
    unit_price = chosen.get("unit_price", 0.0)
    total_cost = round(quantity * unit_price, 2)

    days_remaining = alert.get("days_of_stock_remaining", 0)
    trend          = state.get("consumption", {}).get("trend", "stable")
    item_name      = state.get("inventory", {}).get("name", "")
    category       = state.get("inventory", {}).get("category", "unknown")

    # order_id was stored as a string by save_order; convert back to ObjectId
    # for order_history's proposed_order_id reference field.
    order_id_str = state.get("order_id", "")
    try:
        order_id_obj = ObjectId(order_id_str)
    except Exception:
        order_id_obj = None

    _explain(
        f"Writing decision to memory for {alert['sku']} — "
        f"short-term (24 h TTL), long-term semantic embedding, order history. "
        f"Outcome: {'approved ✅' if approved else 'rejected ❌'}"
    )

    loop = asyncio.get_running_loop()
    await asyncio.gather(
        # ── short_term_memory ──────────────────────────────────────────────
        # Simple find-or-upsert with a 24-hour TTL index.
        # Shows: basic CRUD + TTL indexes in MongoDB.
        loop.run_in_executor(
            None, write_short_term_memory_sync,
            alert["sku"], alert["location"], rec, days_remaining,
            approved and decision_source != "human",
            decision_source, human_decision,
        ),
        # ── agent_memory (long-term) ───────────────────────────────────────
        # Generates a Voyage AI embedding of the decision summary and stores
        # it in agent_memory.  Retrieved later with $vectorSearch.
        # Shows: vector embeddings + Atlas Vector Search round-trip.
        loop.run_in_executor(
            None, write_long_term_memory_sync,
            alert["sku"], item_name, alert["location"],
            rec, days_remaining, trend,
            approved and decision_source != "human",
            decision_source, human_decision,
        ),
        # ── order_history ──────────────────────────────────────────────────
        # Detailed record of every order placed, used by find_similar_past_orders
        # to seed the recommendation agent with historical context.
        # Shows: aggregation-friendly schema design.
        loop.run_in_executor(
            None, write_order_history_sync,
            alert["sku"], alert["location"], category,
            rec.get("supplier_id"), rec.get("supplier_name", "N/A"),
            quantity, unit_price, total_cost,
            days_remaining, trend, rec.get("rationale", ""), order_id_obj,
        ),
    )

    log.info("memories written", extra={
        "phase":    "write_memories",
        "sku":      alert["sku"],
        "location": alert.get("location"),
        "approved": approved,
        "order_id": order_id_str,
    })

    if state.get("requeue_alert"):
        alert_oid = ObjectId(alert["_id"]) if isinstance(alert["_id"], str) else alert["_id"]
        await _db_async.reorder_alerts.update_one(
            {"_id": alert_oid},
            {
                "$set":   {"status": "pending"},
                "$unset": {"order_id": ""},
                "$inc":   {"rejection_count": 1},
            },
        )
        log.info("rejected alert re-queued after memory write", extra={
            "phase": "write_memories", "sku": alert["sku"], "order_id": order_id_str,
        })
    return {}


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

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _post)
        print(f"  [escalate] Webhook notified for {sku}")
    except Exception as exc:
        print(f"  [escalate] Webhook notification failed: {exc}")


async def escalate_alert(state: AgentState) -> AgentState:
    """Write an escalation record when recommend exhausts all validation retries.

    Writes to escalation_queue, marks the alert as 'escalated', and
    optionally POSTs a webhook notification.
    """
    alert  = state["alert"]
    sku    = alert.get("sku", "?")
    reason = "validation_max_retries"

    _explain(
        f"[escalate_alert] ⚠️  Escalating {sku} — the recommend node failed validation "
        "after all retries without producing a valid order. "
        "Writing to escalation_queue so a human specialist can investigate."
    )

    alert_oid = ObjectId(alert["_id"]) if isinstance(alert["_id"], str) else alert["_id"]

    lifecycle_doc = _db_sync.alert_lifecycle.find_one({
        "alert_id": {"$in": [alert_oid, str(alert_oid)]},
    })
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
    return {}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_by_urgency(state: AgentState) -> str:
    """Zero-gap fast path: skip the LLM and go straight to save_order.

    If existing on-order stock already covers the reorder point there is
    nothing to order.  save_order will write a synthetic zero-quantity record.
    """
    if state.get("coverage_gap") == 0:
        return "save_order"
    return "recommend"


def route_after_recommend(state: AgentState) -> str:
    """Route after recommend: escalate if validation failed, otherwise save."""
    if state.get("escalate_flag"):
        log.warning("recommend exhausted retries, escalating", extra={
            "phase": "recommend", "sku": state["alert"]["sku"],
        })
        return "escalate"
    return "save_order"


# ---------------------------------------------------------------------------
# Graph construction — 5-node topology
# ---------------------------------------------------------------------------
#
#   START
#     → assess_alert
#         ↘ save_order  (zero-gap fast path — no LLM needed)
#         ↘ recommend   (retrieval + analysis + recommendation + validation)
#               ↘ escalate    (validation exhausted)
#               ↘ save_order  (valid recommendation)
#                   → write_memories
#     → END
#
# ---------------------------------------------------------------------------

def build_graph(checkpointer_required: bool = True):
    builder = StateGraph(AgentState)

    builder.add_node("assess_alert",   assess_alert)
    builder.add_node("recommend",      recommend)
    builder.add_node("escalate",       escalate_alert)
    builder.add_node("save_order",     save_order)
    builder.add_node("write_memories", write_memories)

    builder.add_edge(START, "assess_alert")
    builder.add_conditional_edges(
        "assess_alert",
        route_by_urgency,
        {"save_order": "save_order", "recommend": "recommend"},
    )
    builder.add_conditional_edges(
        "recommend",
        route_after_recommend,
        {"save_order": "save_order", "escalate": "escalate"},
    )
    builder.add_edge("escalate",       END)
    builder.add_edge("save_order",     "write_memories")
    builder.add_edge("write_memories", END)

    if MongoDBSaver is None:
        if checkpointer_required:
            raise RuntimeError(
                "langgraph-checkpoint-mongodb is required for checkpointed agent runs; "
                "install dependencies from requirements.txt"
            )
        return builder.compile()

    checkpointer = MongoDBSaver(_sync_client, db_name="supply_chain_demo")
    return builder.compile(checkpointer=checkpointer)


graph = build_graph(checkpointer_required=MongoDBSaver is not None)


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
       coverage gap and save_order writing the order.
    """
    sku       = alert_doc.get("sku", "?")
    location  = alert_doc.get("location", "")
    thread_id = str(alert_doc["_id"])
    alert_id  = thread_id

    # ── 1. Circuit breaker ────────────────────────────────────────────────────
    if _get_circuit_failures() >= _CIRCUIT_THRESHOLD:
        log.error("circuit breaker open, routing to human review queue", extra={
            "phase": "process_alert", "sku": sku,
        })
        alert_oid = ObjectId(alert_doc["_id"])
        _db_sync.human_review_queue.update_one(
            {"alert_id": alert_oid},
            {"$set": {
                "alert_id":       alert_oid,
                "sku":            sku,
                "location":       location,
                "reason":         "llm_circuit_open",
                "queued_at":      datetime.now(timezone.utc),
                "alert_snapshot": alert_doc,
            }},
            upsert=True,
        )
        _db_sync.reorder_alerts.update_one(
            {"_id": alert_oid},
            {"$set": {"status": "human_review", "last_error": "llm_circuit_open"}},
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
        {"$set": {"status": "processing", "processing_started_at": datetime.now(timezone.utc)}},
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
    except Exception as exc:  # noqa: BLE001
        alert_oid = ObjectId(alert_id)
        error_count = int(claimed.get("processing_error_count", 0)) + 1
        error_msg = str(exc)[:1000]

        if isinstance(exc, CircuitOpenError) or _get_circuit_failures() >= _CIRCUIT_THRESHOLD:
            await _db_async.human_review_queue.update_one(
                {"alert_id": alert_oid},
                {"$set": {
                    "alert_id":       alert_oid,
                    "sku":            sku,
                    "location":       location,
                    "reason":         "llm_circuit_open",
                    "queued_at":      datetime.now(timezone.utc),
                    "alert_snapshot": claimed,
                    "last_error":     error_msg,
                }},
                upsert=True,
            )
            await _db_async.reorder_alerts.update_one(
                {"_id": alert_oid},
                {"$set": {"status": "human_review", "last_error": error_msg}},
            )
            log.error("alert routed to human review after circuit failure", extra={
                "phase": "process_alert", "sku": sku, "error": error_msg,
            })
        elif error_count >= 3:
            await _db_async.escalation_queue.update_one(
                {"alert_id": alert_oid},
                {"$set": {
                    "alert_id":          alert_oid,
                    "sku":               sku,
                    "location":          location,
                    "processing_errors": error_count,
                    "escalated_at":      datetime.now(timezone.utc),
                    "escalation_reason": "processing_errors",
                    "last_error":        error_msg,
                }},
                upsert=True,
            )
            await _db_async.reorder_alerts.update_one(
                {"_id": alert_oid},
                {"$set": {
                    "status": "escalated",
                    "processing_error_count": error_count,
                    "last_error": error_msg,
                }},
            )
            webhook_url = os.getenv("ESCALATION_WEBHOOK_URL")
            if webhook_url:
                await _notify_escalation_webhook(webhook_url, sku, claimed, "processing_errors")
            log.error("alert escalated after repeated processing errors", extra={
                "phase": "process_alert", "sku": sku,
                "processing_errors": error_count, "error": error_msg,
            })
        else:
            await _db_async.reorder_alerts.update_one(
                {"_id": alert_oid},
                {"$set": {
                    "status": "pending",
                    "processing_error_count": error_count,
                    "last_error": error_msg,
                    "last_failed_at": datetime.now(timezone.utc),
                }},
            )
            log.error("alert processing failed, reset to pending", extra={
                "phase": "process_alert", "sku": sku,
                "processing_errors": error_count, "error": error_msg,
            })
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
        errors = validate_alert_document(_db_sync, alert_doc, "startup_drain")
        if errors:
            log.error("startup: pre-existing alert validation failed",
                      extra={"sku": sku, "errors": errors})
            continue
        try:
            await _process_one_alert(alert_doc)
            count += 1
        except Exception as exc:  # noqa: BLE001
            log.error("startup: error processing pre-existing alert",
                       extra={"sku": sku, "error": str(exc)})
    if count:
        log.info("startup: drained pre-existing pending alerts", extra={"count": count})


async def _recover_stale_processing_alerts() -> int:
    """Requeue alerts left in processing by a crashed worker.

    The alert claim path stamps processing_started_at. On startup, any processing
    alert older than the timeout is safe to retry because no in-process graph can
    survive an agent process restart. Documents without the timestamp are treated
    as legacy stale records from before this recovery field existed.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=_STALE_PROCESSING_AFTER_SECONDS)
    result = await _db_async.reorder_alerts.update_many(
        {
            "status": "processing",
            "$or": [
                {"processing_started_at": {"$lt": cutoff}},
                {"processing_started_at": {"$exists": False}},
            ],
        },
        {
            "$set": {
                "status": "pending",
                "recovered_from_stale_processing_at": datetime.now(timezone.utc),
            },
            "$unset": {"processing_started_at": ""},
        },
    )
    if result.modified_count:
        log.warning("startup: recovered stale processing alerts", extra={
            "count": result.modified_count,
            "stale_after_seconds": _STALE_PROCESSING_AFTER_SECONDS,
        })
    return result.modified_count


# ---------------------------------------------------------------------------
# Vector-dim preflight — catches index/model mismatches before they silently
# produce zero search hits.  A mismatch only happens when the embedding model
# is swapped without reseeding; this makes the failure loud and immediate.
# ---------------------------------------------------------------------------

#: Expected embedding dimensions for every vector index used by the agent.
#: Map collection -> index -> expected dimensions. Update this dict whenever
#: the embedding model or index definition changes.
_VECTOR_INDEX_DIMS: dict[str, dict[str, int]] = {
    "order_history": {
        "order_history_vector_index": 1024,   # voyage-4-large
    },
    "agent_memory": {
        "agent_memory_vector_index": 1024,    # voyage-4-large
    },
}


def _assert_vector_index_dims() -> None:
    """Verify that each Atlas Vector Search index has the expected numDimensions.

    Queries the $listSearchIndexes pipeline.  Logs a WARNING (not a crash) when
    an index is missing or has the wrong dimension — the agent can still run, but
    vector search will return zero results until the index is rebuilt.
    """
    for collection_name, indexes_by_name in _VECTOR_INDEX_DIMS.items():
        collection = _db_sync[collection_name]
        for index_name, expected_dims in indexes_by_name.items():
            try:
                indexes = list(collection.aggregate([
                    {"$listSearchIndexes": {"name": index_name}},
                ]))
            except Exception as exc:
                log.warning(
                    "vector-dim preflight: could not list indexes",
                    extra={"collection": collection_name, "index": index_name, "error": str(exc)},
                )
                continue

            if not indexes:
                log.warning(
                    "vector-dim preflight: index missing",
                    extra={
                        "collection": collection_name,
                        "index":      index_name,
                        "fix":        "re-run data/seed.py to rebuild the index",
                    },
                )
                continue

            vector_fields = [
                field
                for field in indexes[0].get("latestDefinition", {}).get("fields", [])
                if field.get("type") == "vector"
            ]
            if not vector_fields:
                log.warning(
                    "vector-dim preflight: index has no vector field",
                    extra={"collection": collection_name, "index": index_name},
                )
                continue

            actual = vector_fields[0].get("numDimensions")
            if actual != expected_dims:
                log.warning(
                    "vector index dimension mismatch — vector search will return zero hits",
                    extra={
                        "collection":    collection_name,
                        "index":         index_name,
                        "expected_dims": expected_dims,
                        "actual_dims":   actual,
                        "fix":           "re-run data/seed.py to rebuild the index",
                    },
                )
            else:
                log.info(
                    "vector index dims OK",
                    extra={"collection": collection_name, "index": index_name, "dims": actual},
                )


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

    if MongoDBSaver is None:
        raise RuntimeError(
            "langgraph-checkpoint-mongodb is required to run the agent watcher with HITL recovery"
        )

    await sku_lock.ensure_indexes(_db_async)
    log.info("sku_processing_locks indexes ensured")

    _assert_vector_index_dims()

    await _recover_stale_processing_alerts()
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

                    errors = validate_alert_document(_db_sync, alert_doc, "change_stream")
                    if errors:
                        log.error("alert validation failed", extra={"sku": sku, "errors": errors})
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
