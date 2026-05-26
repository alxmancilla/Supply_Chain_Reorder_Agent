"""
LLM prompt templates for the multi-agent reorder pipeline.

Agents and their prompts:
  retrieval_agent      → RETRIEVAL_SYSTEM_PROMPT   (ReAct tool-calling loop)
  analysis_agent       → ANALYSIS_SYSTEM_PROMPT    (supplier eval + confidence)
  recommendation_agent → RECOMMENDATION_SYSTEM_PROMPT (quantity + rationale)

Prompt builders:
  build_analysis_prompt(state)        → user prompt for analysis_agent
  build_recommendation_prompt(state)  → user prompt for recommendation_agent
"""

# ---------------------------------------------------------------------------
# Retrieval agent — ReAct tool-calling loop
# ---------------------------------------------------------------------------

RETRIEVAL_SYSTEM_PROMPT = """\
You are a data-retrieval specialist for a healthcare supply chain reorder system.

Your ONLY task is to call the appropriate tools to gather context for a procurement decision.
Do NOT make procurement recommendations. Do NOT analyse supplier performance.
Just gather the data the downstream analysis agent will need.

REQUIRED tool calls for every alert (always call these four):
  1. get_supplier_options(sku)              — approved suppliers and pricing
  2. get_consumption_trend(sku, location)   — 14-day demand trend
  3. get_recent_decisions(sku, location)    — orders placed in the last 24 h
  4. get_learned_patterns(situation)        — relevant past procurement patterns
     (situation = plain-English description: SKU, location, urgency, stock level, category)

CONDITIONAL tool calls based on urgency:
  • search_suppliers_by_capability(query)       — always helpful; use urgency-appropriate query
    (e.g. "urgent emergency expedited <item>" for critical, "standard routine <item>" for stable)
  • find_similar_past_orders(situation, category) — call when ≥ 5 days of stock remain
    (less useful for critical alerts where speed matters more than precedent)

After calling all needed tools, write a brief plain-text summary (3–5 sentences):
  • Number of suppliers found and the top supplier by fill rate
  • Average daily consumption and trend direction
  • Any recent decisions or human overrides for this SKU
  • Relevant long-term memory patterns found (if any)
"""

# ---------------------------------------------------------------------------
# Analysis agent — supplier evaluation and risk assessment
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """\
You are a supply chain risk analyst for a healthcare distribution network.

You receive retrieved inventory data, supplier options, consumption trends, past orders,
and memory entries. Evaluate this data and produce a structured risk and supplier assessment.

Your output MUST be a JSON object with EXACTLY these fields:
{
  "best_supplier_id":   "<supplier_id of your recommended supplier>",
  "best_supplier_name": "<supplier name>",
  "confidence":         "<high | medium | low>",
  "risk_flags":         ["<concise risk string>", "…"],
  "reasoning_trace":    "<2-3 sentences explaining your supplier choice and confidence level>"
}

Confidence criteria — assign based on ALL factors:
  high   → ALL of: ≥5 days remaining, trend is stable or decreasing,
           preferred supplier fill rate ≥95% confirmed by Atlas Search,
           AND ≥14 days of consumption history sampled
  medium → does not qualify for high but no critical flags:
           2–4 days remaining, OR trend increasing, OR two suppliers closely matched,
           OR only 7–13 days of history sampled
  low    → ANY of: <2 days remaining, fill rate <85%, trend unknown,
           <7 days of history, no supplier confirmed by Atlas Search

Supplier selection rules:
  - Prioritise fill rate and lead time for critical situations (< 2 days stock).
  - Factor in Atlas Search capability scores where available.
  - Respect human decisions from memory:
      ⚠ HUMAN REJECTED → avoid that supplier; explain what changed if forced to use it.
      ✓ HUMAN APPROVED → prefer that supplier if conditions are still comparable.
  - For emergencies, prefer the fastest-lead-time supplier even if fill rate is slightly lower.

Do NOT calculate order quantities. Do NOT write rationale for the human approver.
Return ONLY the JSON object — no markdown fences, no extra keys, no text outside the JSON.
"""

# Backward-compatibility alias — existing code that imports SYSTEM_PROMPT still works.
SYSTEM_PROMPT = ANALYSIS_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Recommendation agent — quantity calculation and approver rationale
# ---------------------------------------------------------------------------

RECOMMENDATION_SYSTEM_PROMPT = """\
You are a procurement specialist for a healthcare distribution network.
Your job is to draft the final purchase order recommendation that a human will review.

You receive an analyst's supplier recommendation, inventory data, and the coverage gap.
Produce a JSON object with EXACTLY these fields:
{
  "supplier_id":   "<use the analyst's recommended supplier_id unless you have strong reason>",
  "supplier_name": "<supplier name>",
  "quantity":      <integer — calculated using the formula below>,
  "rationale":     "<2-3 sentence plain English explanation for the approver>",
  "confidence":    "<use the analyst's confidence — only override with explanation>"
}

Quantity calculation rules (apply in priority order):
  1. If coverage_gap = 0  → quantity = 0 (existing orders are sufficient, no new stock needed)
  2. If coverage_gap > 0  → quantity = ceil(coverage_gap / MOQ) × MOQ
  3. If gap unavailable   → quantity = ceil((avg_daily × 30 + safety_stock) / MOQ) × MOQ

Rationale guidelines:
  - Cite the analyst's confidence level and key risk flags.
  - Reference relevant past orders or human decisions if applicable.
  - If overriding the analyst's supplier choice, explain why clearly.
  - Use plain English — the approver is not a technical user.
  - If a human REJECTED a recent order, explicitly state what changed in this recommendation.

Patient safety: healthcare items must NEVER reach zero stock.

Return ONLY the JSON object — no markdown fences, no extra keys, no text outside the JSON.
"""


# ---------------------------------------------------------------------------
# build_analysis_prompt — user message for analysis_agent
# ---------------------------------------------------------------------------

def build_analysis_prompt(state: dict) -> str:
    """Build the structured context prompt for the analysis agent."""
    alert          = state["alert"]
    inventory      = state.get("inventory", {})
    consumption    = state.get("consumption", {})
    suppliers      = state.get("suppliers", [])
    search_results = state.get("supplier_search_results", [])
    similar_orders = state.get("similar_orders", [])
    short_term     = state.get("short_term_memories", [])
    long_term      = state.get("long_term_memories", [])

    # ── Supplier list (fill-rate sorted) ─────────────────────────────────
    supplier_lines = "\n".join(
        f"  {i+1}. {s['supplier_name']} (ID: {s['supplier_id']}) — "
        f"lead {s['lead_time_days']}d, MOQ {s['moq']}, "
        f"${s['unit_price']:.2f}/unit, "
        f"fill {s['fill_rate_pct']}%, on-time {s['on_time_delivery_pct']}%"
        for i, s in enumerate(suppliers)
        if "error" not in s
    ) or "  No supplier data available."

    # ── Atlas Search results ──────────────────────────────────────────────
    search_lines = "\n".join(
        f"  • {r['supplier_name']} (score {r.get('search_score', 0):.3f}) — "
        f"{r.get('notes', '')[:120]}…"
        for r in search_results
        if "error" not in r and "info" not in r
    ) or "  Atlas Search index not yet available — use fill-rate ranking only."

    # ── Similar past orders ───────────────────────────────────────────────
    similar_lines = "\n".join(
        f"  • {r['sku']} @ {r.get('location','?')}: "
        f"ordered {r.get('quantity','?')} units from {r.get('supplier_name','?')}, "
        f"{r.get('days_of_stock_at_order','?')}d remaining, "
        f"trend={r.get('trend_at_order','?')}, outcome={r.get('outcome','?')}, "
        f"score={r.get('similarity_score', 0):.3f}\n"
        f"    Rationale: \"{r.get('rationale','')[:200]}\""
        for r in similar_orders
        if "error" not in r and "info" not in r
    ) or "  No similar historical orders found."

    # ── Short-term memory ─────────────────────────────────────────────────
    short_term_lines = "\n".join(
        _fmt_short_term(m) for m in short_term if "error" not in m
    ) or "  No recent orders for this SKU in the last 24 h."

    # ── Long-term memory ──────────────────────────────────────────────────
    long_term_lines = "\n".join(
        f"  • [relevance {m.get('memory_score', 0):.3f}] {m.get('content','')[:250]}"
        for m in long_term
        if "error" not in m
    ) or "  No long-term memories yet — collection grows as the agent makes decisions."

    return f"""\
INVENTORY ALERT
---------------
SKU:              {alert['sku']}
Name:             {inventory.get('name', 'N/A')}
Location:         {alert['location']}
On Hand:          {inventory.get('on_hand', alert.get('on_hand', 0))} units
On Order:         {inventory.get('on_order', alert.get('on_order', 0))} units
Reorder Point:    {inventory.get('reorder_point', alert.get('reorder_point', 0))} units
Safety Stock:     {inventory.get('safety_stock', 'N/A')} units
Days of Stock:    {alert.get('days_of_stock_remaining', 0)}
Urgency:          {'CRITICAL (<2d)' if alert.get('days_of_stock_remaining', 99) < 2
                   else 'ELEVATED (2-5d)' if alert.get('days_of_stock_remaining', 99) < 5
                   else 'STANDARD (≥5d)'}

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
(Semantically nearest historical orders — use outcomes to guide your assessment)
{similar_lines}

SHORT-TERM MEMORY — RECENT DECISIONS FOR THIS SKU (last 24 h)
--------------------------------------------------------------
(⚠ HUMAN REJECTED = human overrode last recommendation — avoid same approach)
(✓ HUMAN APPROVED = strong signal to prefer that supplier again)
{short_term_lines}

LONG-TERM MEMORY — LEARNED PATTERNS
-------------------------------------
(Patterns learned across past runs — weight alongside current context)
{long_term_lines}

TASK
----
Evaluate the above data and return your risk assessment as a JSON object.
Focus on: supplier reliability, urgency level, historical outcomes, and memory signals.
"""


# ---------------------------------------------------------------------------
# build_recommendation_prompt — user message for recommendation_agent
# ---------------------------------------------------------------------------

def build_recommendation_prompt(state: dict) -> str:
    """Build the structured context prompt for the recommendation agent.

    Includes the analysis agent's output so the recommendation agent can
    build directly on the risk assessment rather than re-deriving it.
    """
    alert          = state["alert"]
    inventory      = state.get("inventory", {})
    consumption    = state.get("consumption", {})
    suppliers      = state.get("suppliers", [])
    short_term     = state.get("short_term_memories", [])
    long_term      = state.get("long_term_memories", [])
    analysis       = state.get("analysis", {})
    episodes   = state.get("retrieval_results", {}).get("get_episode_history", [])
    procedures = state.get("retrieval_results", {}).get("get_applicable_procedures", [])

    # ── Supplier list (for MOQ lookup) ────────────────────────────────────
    supplier_lines = "\n".join(
        f"  {i+1}. {s['supplier_name']} (ID: {s['supplier_id']}) — "
        f"lead {s['lead_time_days']}d, MOQ {s['moq']}, "
        f"${s['unit_price']:.2f}/unit, "
        f"fill {s['fill_rate_pct']}%, on-time {s['on_time_delivery_pct']}%"
        for i, s in enumerate(suppliers)
        if "error" not in s
    ) or "  No supplier data available."

    # ── Short-term memory ─────────────────────────────────────────────────
    short_term_lines = "\n".join(
        _fmt_short_term(m) for m in short_term if "error" not in m
    ) or "  No recent orders for this SKU in the last 24 h."

    # ── Long-term memory ──────────────────────────────────────────────────
    long_term_lines = "\n".join(
        f"  • [relevance {m.get('memory_score', 0):.3f}] {m.get('content','')[:250]}"
        for m in long_term
        if "error" not in m
    ) or "  No long-term memories yet."

    # ── Past lifecycle episodes ───────────────────────────────────────────
    def _fmt_episode(ep: dict) -> str:
        events = ep.get("events", [])
        lines = [f"  Episode for {ep.get('sku','?')} @ {ep.get('location','?')}:"]
        for ev in events:
            ev_type = ev.get("type", "?")
            if ev_type == "agent_decision":
                lines.append(
                    f"    • agent decided: {ev.get('quantity','?')} units from "
                    f"{ev.get('supplier','?')} (confidence={ev.get('confidence','?')}, "
                    f"auto_approved={ev.get('auto_approved','?')})"
                )
            elif ev_type in ("human_approved", "human_rejected"):
                lines.append(f"    • {ev_type.replace('_', ' ')}")
            elif ev_type == "stock_recovered":
                lines.append(
                    f"    • stock recovered: on_hand={ev.get('on_hand','?')}"
                )
        return "\n".join(lines)

    valid_episodes = [e for e in (episodes or []) if "error" not in e and "info" not in e]
    episode_lines = "\n".join(_fmt_episode(e) for e in valid_episodes) \
        or "  No episode history yet for this SKU."

    # ── Procedural rules ─────────────────────────────────────────────────
    valid_procs = [p for p in (procedures or []) if "error" not in p and "info" not in p]
    procedure_lines = "\n".join(
        f"  • Prefer {p.get('preferred_supplier_name','?')} "
        f"(ID: {p.get('preferred_supplier_id','?')}) — "
        f"confirmed by {p.get('evidence_count',0)} past approvals"
        for p in valid_procs
    ) or "  No confirmed procedural rules for this category+location yet."

    # ── Token budget — trim before assembling the final prompt ──────────
    from agent.context_budget import trim_to_budget
    trimmed = trim_to_budget({
        "supplier_lines":  supplier_lines,
        "short_term_lines": short_term_lines,
        "long_term_lines":  long_term_lines,
        "episode_lines":    episode_lines,
        "procedure_lines":  procedure_lines,
    })
    supplier_lines   = trimmed["supplier_lines"]
    short_term_lines = trimmed["short_term_lines"]
    long_term_lines  = trimmed["long_term_lines"]
    episode_lines    = trimmed["episode_lines"]
    procedure_lines  = trimmed["procedure_lines"]

    # ── Coverage gap section ─────────────────────────────────────────────
    existing_order_qty = state.get("existing_order_qty", 0)
    coverage_gap       = state.get("coverage_gap")
    reorder_point      = inventory.get("reorder_point", alert.get("reorder_point", 0))
    on_hand            = inventory.get("on_hand", alert.get("on_hand", 0))

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
            f"  → Order only enough to close this gap (rounded up to nearest MOQ)."
        )

    # ── Analysis agent output ─────────────────────────────────────────────
    risk_flags_str = (
        "\n".join(f"  ⚠ {f}" for f in analysis.get("risk_flags", []))
        or "  None identified"
    )
    analysis_section = (
        f"  Recommended supplier: {analysis.get('best_supplier_name', 'N/A')} "
        f"(ID: {analysis.get('best_supplier_id', 'N/A')})\n"
        f"  Confidence:           {analysis.get('confidence', 'N/A')}\n"
        f"  Risk flags:\n{risk_flags_str}\n"
        f"  Reasoning:            {analysis.get('reasoning_trace', 'N/A')}"
    ) if analysis else "  Analysis not available — derive supplier choice from suppliers list."

    return f"""\
INVENTORY ALERT
---------------
SKU:           {alert['sku']}
Name:          {inventory.get('name', 'N/A')}
Location:      {alert['location']}
Days of Stock: {alert.get('days_of_stock_remaining', 0)}

CONSUMPTION TREND (last 14 days)
---------------------------------
Average Daily: {consumption.get('avg_daily', 'N/A')} units/day
Trend:         {consumption.get('trend', 'unknown')}
Days Sampled:  {consumption.get('days_sampled', 0)}

ANALYST ASSESSMENT (from analysis agent)
-----------------------------------------
{analysis_section}

APPROVED SUPPLIERS (for MOQ and pricing lookup)
------------------------------------------------
{supplier_lines}

SHORT-TERM MEMORY — RECENT DECISIONS (last 24 h)
-------------------------------------------------
(⚠ HUMAN REJECTED entries require you to change approach and explain what changed)
(✓ HUMAN APPROVED entries are strong signals to reuse that supplier)
{short_term_lines}

LONG-TERM MEMORY — LEARNED PATTERNS
-------------------------------------
{long_term_lines}

PAST EPISODES — FULL LIFECYCLE FOR THIS SKU (most recent first)
---------------------------------------------------------------
(Shows what actually happened: agent decision → human action → stock outcome)
{episode_lines}

PROCEDURAL RULES — LEARNED SUPPLIER PREFERENCES (human-confirmed)
------------------------------------------------------------------
(Strong priors derived from repeated approval patterns — use these when conditions match)
{procedure_lines}

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
  "rationale":     "<2-3 sentence plain English explanation for the approver>",
  "confidence":    "<high | medium | low>"
}}

Calculation hint:
  - coverage_gap = 0  → quantity = 0
  - coverage_gap > 0  → quantity = ceil(coverage_gap / MOQ) × MOQ
  - gap unavailable   → quantity = ceil((avg_daily × 30 + safety_stock) / MOQ) × MOQ

Return ONLY valid JSON — no markdown fences, no extra keys, no text outside the JSON.
"""


# ---------------------------------------------------------------------------
# Shared prompt helpers
# ---------------------------------------------------------------------------

def _fmt_short_term(m: dict) -> str:
    """Format a short-term memory entry with a human-decision tag."""
    from datetime import datetime as dt, timezone as tz

    decided_by = m.get("decided_by", "agent")
    human_dec  = m.get("human_decision")

    if decided_by == "human" and human_dec == "rejected":
        decision_tag = " ⚠ HUMAN REJECTED"
    elif decided_by == "human" and human_dec == "approved":
        decision_tag = " ✓ HUMAN APPROVED"
    elif m.get("auto_approved"):
        decision_tag = " AUTO-APPROVED"
    else:
        decision_tag = " awaiting/agent"

    decided = m.get("decided_at")
    if decided:
        decided = decided.replace(tzinfo=tz.utc) if decided.tzinfo is None else decided
        delta   = dt.now(tz.utc) - decided
        age_str = f"{int(delta.total_seconds() // 3600)}h {int((delta.total_seconds() % 3600) // 60)}m"
    else:
        age_str = "?"

    days     = m.get("days_of_stock_at_decision")
    days_str = f"{days}d stock at decision" if days is not None else "stock level unknown"

    return (
        f"  • {age_str} ago: {m.get('quantity','?')} units from "
        f"{m.get('supplier_name','?')} "
        f"(confidence: {m.get('confidence','?')}, {days_str}{decision_tag})"
    )
