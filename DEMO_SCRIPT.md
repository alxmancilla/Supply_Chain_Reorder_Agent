# Demo Script — Supply Chain Reorder Alert Agent
**Target runtime: ~5 minutes**

---

## 0 · Before you start (setup, not spoken)

- Click **🔄 Reset Demo** in the sidebar to clear any state from a previous run — inventory and suppliers are preserved, operational collections are wiped
- `docker compose up --build` is running (Python 3.12 image); seeder has completed
- Browser open at **http://localhost:8501**
- Simulator is **running** (▶ Start pressed in sidebar — Status shows 🟢 Running)
- Drain Speed slider is at **1× Normal** for a measured start
- Confirm the agent is active: within ~30 seconds of starting you should see proposed orders appearing in the right column — if not, check `docker compose logs agent`
- **Optional — Explain Mode:** run the agent with `EXPLAIN_MODE=1` to print plain-English node descriptions alongside the JSON logs. Useful if your audience wants to see the pipeline narrated in real time without reading structured log output: `EXPLAIN_MODE=1 python agent/graph.py`
- Atlas UI open in a second tab (optional — useful for showing the Search / Vector Search indexes if the audience asks)
- **Watch for these SKUs** — they have a seeded rising demand trend and will produce MEDIUM confidence even with stock above 5 days:
  - `MED-3017` (Insulin Glargine · DC-Texas · pharmaceutical)
  - `SURG-0084` (Nitrile Gloves · DC-Ohio · surgical)
  - `DIAG-0331` (COVID/Flu Test Kit · DC-Ohio · diagnostic)
- **Pharmaceutical SKUs** (`MED-*` and `LAB-0112`) trigger FDA regulatory enforcement — the agent will reject any supplier not marked `fda_registered: true`

---

## 1 · Set the scene `[0:00 – 0:30]`

> "Healthcare supply chains have zero tolerance for stockouts — a missing medication
> can be a patient safety event. But most reorder decisions are still manual:
> someone checks a spreadsheet, picks a supplier, estimates a quantity, and routes
> it for approval. Across hundreds of SKUs and multiple distribution centres,
> done manually under time pressure, things get missed.
> This demo automates that entire loop — an AI agent that responds in seconds,
> not hours, using MongoDB Atlas as the backbone."

**👉 Point broadly at the dashboard.**

> "Two live columns: inventory grid on the left — critical items sort to the top
> automatically. Alerts and agent decisions stacked on the right.
> Seven KPI cards across the top: stock below reorder point, active alerts,
> orders awaiting approval, ROP health, auto-approved count, deliveries, and escalations.
> The dashboard polls every 10 seconds; the agent reacts the moment an alert lands."

---

## 2 · Watch an alert arrive `[0:30 – 1:30]`

**👉 Point at the left column — find a red 🔴 CRITICAL card near the top.**

> "Critical items sort to the top automatically. The progress bar shows how far
> below the reorder point we are. Each card shows On Hand, On Order, Reorder Point,
> Effective stock — and that small indicator at the bottom is the ROP health check,
> which I'll come back to."

**👉 Point at the Active Alerts section when a new alert card appears.**

> "The simulator drains stock every few seconds, mimicking what Atlas Stream
> Processing would do in production off a Kafka feed. The moment on-hand drops
> below the reorder point, an alert document is inserted into MongoDB.
> The agent watches that collection via a Change Stream — not a cron job, not polling —
> so it triggers the instant the write lands."

> "Notice the urgency badge: CRITICAL means less than 2 days of stock.
> The time-ago label tells us exactly when this fired."

*Contingency: if no new alert appears within 20 seconds, move the Drain Speed slider to 3× to accelerate consumption — alerts will fire within seconds.*

**👉 Move the ⚡ Drain Speed slider to 10× Chaos.**

> "Now watch what happens under load."

*(pause 10–15 seconds while alerts fire across all SKUs)*

> "Ten times the normal drain rate — alerts firing across every SKU simultaneously.
> The agent processes them concurrently. Each one runs the full pipeline independently,
> with a distributed MongoDB lock preventing two workers from writing a duplicate order
> for the same SKU. Slide it back to 1× once you've made the point."

**👉 Return slider to 1× Normal. Point at the `PROCESSING…` badge on an alert card.**

> "The agent is already working on these. Let's look at what it produces."

---

## 3 · Show the agent decision `[1:30 – 3:00]`

**👉 Wait for an order card to appear in the Agent Decisions section, then walk through it.**

*Contingency: LLM calls can take 20–40 seconds. If nothing appears, keep talking through the pipeline description — the order will arrive before you finish. If it takes longer than 60 seconds, point at a completed order from an earlier alert and use that.*

> "The agent ran a multi-agent LangGraph pipeline — four specialist agents,
> each with a scoped responsibility.
> A ReAct retrieval agent decides which MongoDB queries to run: supplier list,
> consumption trend, Atlas Search, Vector Search — and only fetches what it needs.
> An analysis agent evaluates the options and assigns a confidence level.
> A recommendation agent calculates the quantity and writes a plain-English rationale.
> An audit agent validates the output against schema and business rules before
> anything is persisted. If validation fails, the pipeline retries automatically —
> up to twice — before escalating to the human queue."

**👉 Point at the confidence and status badges.**

> "Confidence gates the auto-approve decision. HIGH requires all four conditions
> simultaneously: at least 5 days of stock, stable or decreasing trend, a supplier
> with 95%+ fill rate, and at least 14 days of consumption history on record.
> Any single red flag — rising trend, thin history, weak supplier — drops it to
> MEDIUM or LOW."
>
> "If confidence is HIGH and the total cost is under $5,000, the order is approved
> instantly — you'll see the ⚡ AUTO badge.
> Orders at or above $5,000 are always held for human review regardless of confidence.
> The card shows a ⚠ REVIEW badge with the specific reason."

**👉 If the alert is for MED-3017, SURG-0084, or DIAG-0331 — call this out.**

> "This one shows MEDIUM confidence despite having more than 5 days of stock.
> That's because consumption has been climbing steadily over the last 30 days —
> the trend is a genuine gate. An order sized for today's demand may already be
> too small by the time it delivers."

**👉 If the alert is for a pharmaceutical SKU (MED-* or LAB-0112) — add this.**

> "Notice which supplier was chosen. The agent is legally prevented from selecting
> a non-FDA-registered wholesale distributor for pharmaceutical SKUs —
> that's a hard constraint in the audit agent, not a soft preference.
> If it had picked the cheaper option, the audit step would have rejected it
> and forced a retry with a different supplier."

**👉 Expand `💬 Rationale / 🧠 Vector Search`.**

> "Plain-English rationale — ready for a non-technical approver to read.
> Stock level, trend, supplier fill rate, cost, all in one paragraph."

> "Below that are the Vector Search results. The agent embedded the current
> situation with Voyage AI and retrieved the most semantically similar historical
> orders from the same product category — using Atlas Vector Search pre-filtering
> so cross-category noise is eliminated before ANN scoring.
> Similarity scores above 0.78. These past outcomes feed directly into the
> LLM prompt as precedent, so the agent learns from what worked before."

---

## 4 · Show memory growing + reject / escalation flow `[3:00 – 4:00]`

**👉 Point at the sidebar Demo Status panel.**

> "Long-term Memories count starts low but grows with every decision.
> After each order the agent writes a natural-language summary, embeds it,
> and stores it in `agent_memory`. From the next alert for this SKU onwards,
> the agent retrieves its own past learnings via Vector Search and factors them
> into the recommendation — supplier preferences, quantity adjustments, what a
> human approved or rejected."

**👉 Find a `MEDIUM` confidence order with the AWAITING badge and click ❌ Reject.**

> "Watch what happens when I click Reject. This isn't a simple MongoDB write.
> The graph actually paused at a node called `save_order` — LangGraph serialised
> the entire agent state to the `checkpoints` collection via MongoDBSaver and waited.
> When I click Reject, the dashboard calls `graph.invoke(Command(resume={'approved': False}))`.
> LangGraph reloads that checkpoint from MongoDB, re-enters the node right where it
> stopped, and runs the rejection logic — updating the order, the alert, the memory —
> all inside the graph."
>
> "Open the `checkpoints` collection in Atlas while an order is awaiting review.
> You'll see the complete agent state — suppliers, retrieval trace, recommendation,
> memories — frozen in one document. That's the entire LangGraph + MongoDB story
> in a single Atlas query."
>
> "After the rejection: the alert resets to pending, the Change Stream fires again,
> and the agent reprocesses with a ⚠ HUMAN REJECTED tag visible in short-term memory.
> The prompt requires the agent to change its approach — different supplier,
> adjusted quantity — and explain what changed in the rationale."
>
> "The rejection is also written to long-term memory as an embedded document.
> Human oversight becomes a persistent training signal, not just a one-time gate."

*Contingency: if the reprocessed order takes more than 30 seconds, point at the Demo Status panel and watch the Long-term Memories counter increment — that's the memory write completing in the background.*

**👉 Watch the new order appear and point at the rationale.**

> "The rationale now explicitly acknowledges the prior rejection and explains
> what changed. The supplier choice or quantity will be different."

**👉 Mention the escalation safety net briefly.**

> "There's a safety net. If the same alert is rejected three times, the agent
> stops retrying and escalates — writes to an escalation queue, marks the alert
> 'escalated', and optionally fires a webhook. Those alerts surface in a dedicated
> section on the dashboard. Nothing falls through the cracks."

**👉 Click ✅ Approve on any order.**

> "Approvals are recorded the same way — ✓ HUMAN APPROVED in short-term memory
> and an embedded entry in long-term memory. That supplier gets a preference signal
> for future alerts on this SKU."

---

## 5 · MongoDB Atlas features — rapid fire `[4:00 – 4:45]`

**👉 Scroll down to Supplier Search in the sidebar, type `cold chain insulin`, press Search.**

> "The search uses two independent pipelines — one finds exact supplier name matches,
> one finds fuzzy keyword matches in capability notes — then combines their rankings
> using Reciprocal Rank Fusion. Think of it as asking two experts to independently
> shortlist suppliers and then intelligently merging their lists. The source label
> shows 🔀 $rankFusion. If the Atlas Search index isn't available it falls back
> to a plain regex search automatically."

**👉 Point at an inventory card's ✅ / ⚠️ / 🔴 indicator at the bottom.**

> "Each inventory card has a ROP health check. Green means the reorder point fires
> with enough runway for the supplier to deliver. Amber means safety stock barely
> bridges the gap. Red means the reorder point is set too low — by the time the
> alert fires, the supplier can't deliver before stock hits zero.
> The 📐 ROP Health KPI card at the top counts how many SKUs are currently at risk."

**👉 Open Admin Panel → 🔁 Agent Recovery Log.**

> "LangGraph checkpoints every node transition to MongoDB via MongoDBSaver.
> That's the same collection that stores the pause-point when `interrupt()` fires —
> one collection, two jobs: crash recovery and human-in-the-loop state.
> If the agent process crashes mid-pipeline, the next invocation resumes from
> the last checkpoint rather than starting from scratch.
> This panel reads those checkpoints directly and shows each pipeline run:
> SKU, last node executed, step count, and outcome — completed, re-queued,
> escalated, or recovered mid-pipeline. Crash recovery and human review,
> both made visible through the same MongoDB collection."

---

## 6 · Close `[4:45 – 5:00]`

> "Everything you just saw — semantic retrieval, regulatory enforcement,
> crash recovery, real-time event streaming — runs on a single MongoDB Atlas cluster.
> No separate vector database. No external search service. No dedicated streaming
> platform. The same store that holds your inventory data also carries the agent's
> memory, the compliance checks, and the full audit trail.
> In production the simulator is replaced by an Atlas Stream Processing pipeline
> off Kafka — the agent code doesn't change, it only sees the `reorder_alerts`
> collection. That's the architectural bet: when AI is core to your operations,
> your data platform has to carry it natively."

---

## Quick-reference timings

| Segment | Time | Key talking points |
|---|---|---|
| Set the scene | 0:00 – 0:30 | Pain point (manual reorder = risk), two-column layout, 7 KPI cards, Change Stream vs auto-refresh distinction |
| Alert arrives + Chaos Mode | 0:30 – 1:30 | Change Stream trigger, urgency badge, 10× Chaos to show concurrent load, distributed SKU lock |
| Agent decision + rationale | 1:30 – 3:00 | Four-agent pipeline, 4-criteria confidence, $5,000 ceiling, ⚠ REVIEW badge, rising-trend SKUs, FDA enforcement (if pharma SKU), Vector Search precedent |
| Memory + reject / escalation | 3:00 – 4:00 | Human decisions in both memory layers, ⚠ REJECTED tag forces change, 3-rejection escalation safety net |
| Atlas features rapid-fire | 4:00 – 4:45 | $rankFusion "two experts" analogy, ROP health indicator, Agent Recovery Log (checkpoint recovery) |
| Close | 4:45 – 5:00 | No separate infrastructure — one cluster for storage, search, vector, streaming, agent state |

---

## Contingency quick-reference

| Situation | Response |
|---|---|
| No new alert appears | Bump Drain Speed to 3× — alerts fire within seconds; return to 1× after |
| Agent decision takes > 45 s | Continue talking through the pipeline description; point at a completed order from an earlier alert |
| Reprocessed order after reject takes > 30 s | Point at Long-term Memories counter in Demo Status — watch it increment; mention `write_memories` node running in background |
| Atlas Search / $rankFusion error | The UI falls back to regex automatically; mention "graceful degradation" as a feature |
| Agent Recovery Log is empty | Demo was just reset; process one alert first, then reopen the expander |

---

## Sidebar controls — quick reference

| Control | Location | What it does |
|---|---|---|
| ▶ / ⏸ / ⏹ | Simulator Controls | Start / pause / stop inventory drain |
| ⚡ Drain Speed | Below start/stop buttons | 1× Normal → 10× Chaos; multiplies units consumed per tick |
| 🔍 Supplier Search | Mid-sidebar | $rankFusion hybrid search (name + capability) with regex fallback |
| 📊 Demo Status | Lower sidebar | Live counts: alerts, decisions, auto-approved %, escalations, failed memory writes, long-term memories |
| 🔄 Reset Demo | Bottom of sidebar | Clears operational data; resets speed to 1×; clears resume token — run this before every demo |
| 🛠 Admin Panel | Below Reset Demo (admin only) | Circuit breaker reset, Extract Rules, Compact Memory, Procedure candidate review (✅/🗑), Agent Recovery Log (🔁) |
