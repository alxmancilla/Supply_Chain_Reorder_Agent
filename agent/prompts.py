"""
LLM prompt templates for the reorder agent.
"""

SYSTEM_PROMPT = """\
You are an autonomous supply chain reorder agent for a healthcare distribution network.
Your job is to analyse inventory alerts and draft purchase order recommendations that a
human approver will review before submission.

Guidelines:
- Prioritise patient safety: healthcare items should never run to zero stock.
- Cover 30 days of demand plus a safety buffer when calculating order quantity.
- Round order quantities UP to the nearest supplier MOQ (minimum order quantity).
- Use Atlas Search relevance scores to identify suppliers with the best capability match
  for the current situation (e.g. cold-chain, expedited delivery, emergency stock).
- Review similar past orders from Vector Search: favour suppliers and quantities that
  delivered good outcomes in analogous situations.
- Consult SHORT-TERM MEMORY to avoid conflicting with a very recent order for the same
  SKU — if an order was placed in the last few hours, factor that into your quantity.
  Pay close attention to entries where decided_by='human':
    • human_decision='rejected' → a human overrode your last recommendation; you MUST
      change your approach: try a different supplier, adjust the quantity, or lower
      confidence. Explain what you changed and why in the rationale.
    • human_decision='approved' → a human validated that supplier and quantity; strong
      signal to reuse the same supplier if conditions are still comparable.
- Consult LONG-TERM MEMORY for learned patterns: preferred suppliers, typical quantities,
  outcomes, and any past human approvals or rejections for this SKU. Weight these
  insights alongside the current context.
- Express the rationale in plain English for a non-technical approver.
- Confidence levels — assign based on ALL factors below:
    high   → ALL of: ≥5 days remaining, trend is stable or decreasing,
             preferred supplier fill rate ≥95% AND confirmed by Atlas Search,
             AND ≥14 days of consumption history sampled
    medium → does not qualify for high but no critical flags:
             2–4 days remaining, OR trend is increasing, OR two suppliers
             closely matched, OR only 7–13 days of history sampled
    low    → ANY of: <2 days remaining, OR <7 days of consumption history,
             OR preferred supplier fill rate <85%, OR trend is unknown,
             OR no supplier confirmed by Atlas Search
"""


def build_recommendation_prompt(state: dict) -> str:
    alert          = state["alert"]
    inventory      = state["inventory"]
    consumption    = state["consumption"]
    suppliers      = state["suppliers"]
    search_results = state.get("supplier_search_results", [])
    similar_orders = state.get("similar_orders", [])
    short_term     = state.get("short_term_memories", [])
    long_term      = state.get("long_term_memories", [])

    # ── Basic supplier list (fill-rate sorted) ────────────────────────────
    supplier_lines = "\n".join(
        f"  {i+1}. {s['supplier_name']} (ID: {s['supplier_id']}) — "
        f"lead {s['lead_time_days']}d, MOQ {s['moq']}, "
        f"${s['unit_price']:.2f}/unit, "
        f"fill {s['fill_rate_pct']}%, on-time {s['on_time_delivery_pct']}%"
        for i, s in enumerate(suppliers)
        if "error" not in s
    ) or "  No supplier data available."

    # ── Atlas Search results (capability-ranked) ──────────────────────────
    search_lines = "\n".join(
        f"  • {r['supplier_name']} (score {r.get('search_score', 0):.3f}) — {r.get('notes', '')[:120]}…"
        for r in search_results
        if "error" not in r and "info" not in r
    ) or "  Atlas Search index not yet available — using fill-rate ranking only."

    # ── Vector Search results (similar past orders) ───────────────────────
    similar_lines = "\n".join(
        f"  • {r['sku']} @ {r.get('location','?')}: "
        f"ordered {r.get('quantity','?')} units from {r.get('supplier_name','?')}, "
        f"{r.get('days_of_stock_at_order','?')}d remaining, trend={r.get('trend_at_order','?')}, "
        f"outcome={r.get('outcome','?')}, score={r.get('similarity_score', 0):.3f}\n"
        f"    Rationale: \"{r.get('rationale','')[:200]}\""
        for r in similar_orders
        if "error" not in r and "info" not in r
    ) or "  No similar historical orders found."

    # ── Short-term memory (per-SKU last 24 h) ────────────────────────────
    def _hours_ago(mem: dict) -> str:
        from datetime import datetime as dt, timezone as tz
        decided = mem.get("decided_at")
        if not decided:
            return "?"
        delta = dt.now(tz.utc) - decided.replace(tzinfo=tz.utc) if decided.tzinfo is None else dt.now(tz.utc) - decided
        return f"{int(delta.total_seconds() // 3600)}h {int((delta.total_seconds() % 3600) // 60)}m"

    def _fmt_short_term(m: dict) -> str:
        decided_by = m.get("decided_by", "agent")
        human_dec  = m.get("human_decision")
        # Build a decision-source tag so the LLM can clearly distinguish
        # agent auto-decisions from human overrides.
        if decided_by == "human" and human_dec == "rejected":
            decision_tag = " ⚠ HUMAN REJECTED"
        elif decided_by == "human" and human_dec == "approved":
            decision_tag = " ✓ HUMAN APPROVED"
        elif m.get("auto_approved"):
            decision_tag = " AUTO-APPROVED"
        else:
            decision_tag = " awaiting/agent"

        days = m.get("days_of_stock_at_decision")
        days_str = f"{days}d stock at decision" if days is not None else "stock level unknown"

        return (
            f"  • {_hours_ago(m)} ago: {m.get('quantity','?')} units from "
            f"{m.get('supplier_name','?')} "
            f"(confidence: {m.get('confidence','?')}, {days_str}{decision_tag})"
        )

    short_term_lines = "\n".join(
        _fmt_short_term(m) for m in short_term if "error" not in m
    ) or "  No recent orders for this SKU in the last 24 h."

    # ── Long-term memory (semantic summaries from agent_memory) ──────────
    long_term_lines = "\n".join(
        f"  • [relevance {m.get('memory_score', 0):.3f}] {m.get('content','')[:250]}"
        for m in long_term
        if "error" not in m
    ) or "  No long-term memories yet — collection grows as the agent makes decisions."

    # ── Existing active orders & coverage gap ────────────────────────────
    existing_order_qty = state.get("existing_order_qty", 0)
    coverage_gap       = state.get("coverage_gap")
    reorder_point      = inventory.get("reorder_point", alert.get("reorder_point", 0))
    on_hand            = inventory.get("on_hand",        alert.get("on_hand",        0))

    if coverage_gap is None:
        coverage_section = "  Coverage gap data not available — use standard 30-day formula."
    elif coverage_gap == 0:
        coverage_section = (
            f"  Existing active orders total {existing_order_qty} units.\n"
            f"  on_hand ({on_hand}) + existing orders ({existing_order_qty}) = "
            f"{on_hand + existing_order_qty} units, which already meets or exceeds "
            f"the reorder point ({reorder_point}).\n"
            f"  → No additional stock is required. You MUST return quantity = 0."
        )
    else:
        coverage_section = (
            f"  Reorder point:          {reorder_point} units\n"
            f"  On hand:                {on_hand} units\n"
            f"  Existing active orders: {existing_order_qty} units "
            f"(awaiting approval or approved, not yet delivered)\n"
            f"  Effective coverage:     {on_hand + existing_order_qty} units\n"
            f"  COVERAGE GAP:           {coverage_gap} units still needed\n"
            f"  → Order only enough to close this gap (rounded up to the nearest MOQ)."
        )

    return f"""\
INVENTORY ALERT
---------------
SKU:              {alert['sku']}
Name:             {inventory.get('name', 'N/A')}
Location:         {alert['location']}
On Hand:          {inventory.get('on_hand', alert['on_hand'])} units
On Order:         {inventory.get('on_order', alert['on_order'])} units
Reorder Point:    {inventory.get('reorder_point', alert['reorder_point'])} units
Safety Stock:     {inventory.get('safety_stock', 'N/A')} units
Days of Stock:    {alert['days_of_stock_remaining']}

CONSUMPTION TREND (last 14 days)
---------------------------------
Average Daily:    {consumption.get('avg_daily', 'N/A')} units/day
Trend:            {consumption.get('trend', 'unknown')}
Peak Day:         {consumption.get('peak_day', 'N/A')}
Days Sampled:     {consumption.get('days_sampled', 0)}

APPROVED SUPPLIERS (sorted by fill rate)
-----------------------------------------
{supplier_lines}

ATLAS SEARCH — SUPPLIER CAPABILITY MATCH
-----------------------------------------
(Ranked by text relevance to current situation: category + urgency)
{search_lines}

ATLAS VECTOR SEARCH — SIMILAR PAST ORDERS
------------------------------------------
(Semantically nearest historical orders — use outcomes to guide your decision)
{similar_lines}

SHORT-TERM MEMORY — RECENT DECISIONS FOR THIS SKU (last 24 h)
--------------------------------------------------------------
(Entries marked ⚠ HUMAN REJECTED mean a human overrode your last recommendation —
 you MUST change your approach. Entries marked ✓ HUMAN APPROVED are strong signals
 to reuse that supplier. AUTO-APPROVED entries are agent decisions already in flight.)
{short_term_lines}

LONG-TERM MEMORY — LEARNED PATTERNS (semantic, from agent_memory)
------------------------------------------------------------------
(Patterns the agent has learned across past runs — weight alongside current context)
{long_term_lines}

EXISTING ACTIVE ORDERS & COVERAGE GAP
--------------------------------------
{coverage_section}

TASK
----
Return a JSON object with exactly these fields:
{{
  "supplier_id":   "<chosen supplier ID>",
  "supplier_name": "<chosen supplier name>",
  "quantity":      <integer — order quantity>,
  "rationale":     "<2-3 sentence plain English explanation for the approver, referencing past outcomes if relevant>",
  "confidence":    "<high | medium | low>"
}}

Calculation hint:
  - If coverage gap = 0: return quantity = 0 (existing orders are sufficient; no new stock needed).
  - If coverage gap > 0: quantity = ceil(coverage_gap / MOQ) * MOQ
    (order only what is needed to bridge the gap to the reorder point, rounded up to MOQ).
  - If coverage gap data is unavailable: quantity = ceil((avg_daily * 30 + safety_stock) / MOQ) * MOQ

Return ONLY valid JSON. No markdown fences, no extra keys, no text outside the JSON.
"""
