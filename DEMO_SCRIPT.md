# Demo Script — Supply Chain Reorder Alert Agent
**Target runtime: ~5 minutes**

---

## 0 · Before you start (setup, not spoken)

- `docker compose up --build` is running; seeder has completed.
- Optional detached start: `docker compose up --build -d`.
- Verify `docker compose ps` shows `app`, `agent`, `simulator`, and `memory-retry-worker` as `Up`.
- Verify `docker compose logs seeder` ends with `Seed complete.` and the Atlas Search / Vector Search indexes are ready.
- Browser open at **http://localhost:8501**.
- Startup may already create awaiting-approval orders from seeded below-ROP alerts. Click **🔄 Reset Demo** if you need a clean baseline.
- Open **Admin Panel → 🎯 Prepare Context Scenario**. This pauses the simulator and creates a deterministic `MED-3017 @ DC-Texas` alert with:
  - live inventory below reorder point
  - rising demand trend from seeded time-series data
  - similar historical insulin orders for Vector Search
  - a recent `⚠ HUMAN REJECTED` short-term memory
  - a confirmed procedural rule preferring BioPharm Global
- Confirm the new alert appears in **Active Alerts**.
- Wait for a new MED-3017 proposed order, then use that order card for the Context Packet walkthrough.
- Optional: keep Atlas UI open to show `reorder_alerts`, `proposed_orders`, `checkpoints`, `short_term_memory`, and `order_history`.
- Optional: run the agent with `EXPLAIN_MODE=1` if you want narrated node logs.
- Stop the stack after rehearsal with `docker compose down`.

---

## 1 · Set the scene `[0:00 – 0:30]`

> "The demo is not just an LLM writing a purchase order. The important part is
> context engineering: deciding what operational facts, memories, policies, and
> historical precedents the model should see before it makes a recommendation."

**Point broadly at the dashboard.**

> "Healthcare inventory decisions are high stakes. A reorder agent needs live stock,
> active orders, supplier constraints, demand trend, past outcomes, human feedback,
> and compliance rules. MongoDB is acting as the context substrate for all of that."

---

## 2 · Alert arrives: live operational context `[0:30 – 1:10]`

**Point at the `MED-3017 @ DC-Texas` alert in Active Alerts.**

> "This alert is a live MongoDB document. The agent watches `reorder_alerts` with a
> Change Stream. When a pending alert appears, the graph claims it atomically and
> computes the real coverage gap from current inventory plus active orders."

**Point at the inventory card.**

> "The first context layer is live state: on hand, reorder point, active orders,
> days of stock, and ROP health. This prevents the model from ordering against
> stale dashboard data or duplicating an in-flight order."

> "The SKU-level lock makes this safe to run with multiple workers. In the default
> demo stack we use one agent worker for clarity, but the lock prevents duplicate
> orders if you scale workers out."

---

## 3 · Open the Context Packet `[1:10 – 2:30]`

**Open the latest order card → `💬 Rationale / 📦 Context Packet`.**

If older seeded orders are visible, use the newest `MED-3017 @ DC-Texas` order created after **🎯 Prepare Context Scenario**.

> "This is the center of the demo. The agent persisted a context manifest with the
> order, so we can inspect what context was used, why it was included, and how it
> was budgeted before the final recommendation was written."

**Point at the three metrics.**

> "Coverage gap tells us the precise quantity problem. Tools Called shows which
> retrieval tools were actually used. Context Tokens shows the final context size
> after budget management."

**Point at 'What context was included and why'.**

> "The context packet is deliberately structured into categories: live state,
> retrieved context, memory, policy rules, and validation. This is the difference
> between a generic chatbot and an operational agent."

Call out these rows if present:

- **Inventory position**: grounds the recommendation in live stock.
- **Active order coverage**: prevents duplicate purchasing.
- **Approved suppliers**: constrains the model to real supplier IDs, MOQ, cost, and lead time.
- **Consumption trend**: `MED-3017` has rising demand, so confidence should drop.
- **Vector-similar past orders**: similar insulin reorder precedents from `order_history`.
- **Short-term memory**: recent `⚠ HUMAN REJECTED` signal.
- **Procedural rules**: confirmed preference for BioPharm Global.
- **Episode history**: prior alert lifecycle outcomes when available.

**Point at Retrieval trace.**

> "The retrieval trace is the agent's context acquisition path. The ReAct stage
> decides which MongoDB-backed tools to call: supplier options, time-series trend,
> Atlas Search, Vector Search, recent decisions, long-term memory, procedures, and
> episode history."

**Point at Token budget trimming.**

> "As memory grows, context cannot grow forever. The budget manager trims low-value
> context first: debug traces, lower-score long-term memories, lower-similarity
> past orders, then less critical sections. High-value facts like coverage gap and
> supplier constraints are kept as long as possible."

---

## 4 · Recommendation and guardrails `[2:30 – 3:10]`

**Point at confidence, status, supplier, quantity, and rationale.**

> "The model does not decide in isolation. The analysis stage evaluates supplier
> reliability, lead time, compliance, trend, and memory. The recommendation stage
> calculates quantity from coverage gap and MOQ, then writes plain-English rationale."

**Point at Validation guardrails in the Context Packet.**

> "After generation, validation checks the output. It rejects malformed JSON,
> unknown supplier IDs, non-FDA suppliers for pharmaceutical SKUs, invalid zero-gap
> quantities, and anything that violates the approval policy. Context engineering
> includes retrieval, prompt structure, and post-generation verification."

**If the order is awaiting approval:**

> "This order is held because confidence or budget policy requires a human. The
> graph pauses with LangGraph `interrupt()` and stores the checkpoint in MongoDB."

**If the order auto-approved:**

> "This path still writes the same context manifest and memory records. For the
> human-feedback segment, use another awaiting-review order or reset and rerun
> the prepared scenario."

---

## 5 · Human feedback changes future context `[3:10 – 4:00]`

**Click ❌ Reject on the awaiting order.**

> "A rejection is not just a status update. The graph resumes from checkpoint,
> marks the order rejected, writes short-term and long-term memory, then requeues
> the alert. The next run sees a different context packet."

**Wait for the next order and reopen `💬 Rationale / 📦 Context Packet`.**

> "Now the short-term memory row contains the human rejection. The prompt renders
> it as `⚠ HUMAN REJECTED`, and the recommendation is required to change approach
> or explain why it cannot."

**Point at rationale.**

> "This is the key context-engineering loop: human feedback becomes context, not
> just an audit log. The next recommendation is conditioned on what the human just
> taught the agent."

**Click ✅ Approve on the revised order if available.**

> "Approval is written the same way: short-term memory for immediate reuse and
> long-term semantic memory for future vector retrieval."

---

## 6 · MongoDB Atlas features through the context lens `[4:00 – 4:45]`

**Supplier Search: type `cold chain insulin`.**

> "Atlas Search is used for supplier capability retrieval. The app combines supplier
> name matching and capability-note matching with `$rankFusion`, then falls back
> gracefully if Search is unavailable."

**Point back to the Context Packet.**

> "Vector Search supplies historical precedents. Time Series supplies trend.
> TTL memory supplies fresh human feedback. Checkpoints freeze the graph state.
> Documents hold the final order and its context manifest. These are all context
> engineering primitives, not separate demo tricks."

**Open Admin Panel → Agent Recovery Log.**

> "The same checkpoint store supports human-in-the-loop pause and crash recovery.
> If the agent crashes after claiming an alert, startup recovery requeues stale
> `processing` alerts. If it crashes around approval, inventory side effects are
> idempotent so stock is not double-counted."

---

## 7 · Close `[4:45 – 5:00]`

> "The takeaway: production agents are only as good as their context pipeline.
> This demo engineers the context packet, validates the output, persists the
> decision, and turns human feedback into future context. MongoDB Atlas is the
> operational substrate for live state, search, vector retrieval, memory,
> checkpoints, and auditability."

---

## Quick-reference timings

| Segment | Time | Key point |
|---|---:|---|
| Setup | before | Use 🎯 Prepare Context Scenario for deterministic MED-3017 flow |
| Set scene | 0:00–0:30 | Context engineering, not just PO generation |
| Alert/live state | 0:30–1:10 | Change Stream, coverage gap, active orders, SKU lock |
| Context Packet | 1:10–2:30 | Sources, retrieval trace, token budget, why included |
| Recommendation | 2:30–3:10 | Staged reasoning, validation guardrails, HITL |
| Human feedback | 3:10–4:00 | Reject → memory → changed context → revised recommendation |
| Atlas features | 4:00–4:45 | Search, Vector Search, Time Series, TTL, checkpoints |
| Close | 4:45–5:00 | MongoDB as context substrate |

---

## Contingency quick-reference

| Situation | Response |
|---|---|
| No order appears | Confirm agent container is running: `docker compose logs agent`; the prepared alert should process without waiting for simulator drain. |
| Services are not running | Run `docker compose ps`; start with `docker compose up --build -d`; stop with `docker compose down`. |
| Context Packet is missing | Use a newly generated order; older orders created before this feature do not have `context_manifest`. |
| Latest order is not MED-3017 | Use the newest MED-3017 order created after Prepare Context Scenario, or click Reset Demo and prepare the scenario again. |
| Vector Search has no hits | Mention graceful degradation; the Context Packet still shows supplier, trend, memory, and validation context. |
| Reprocessed order takes time | Explain that the graph writes memory before requeueing so the next run sees the rejection. |
| Agent Recovery Log is empty | Process one alert first; checkpoints appear after graph execution. |
| Too many old cards | Click Reset Demo, then Prepare Context Scenario again. |
| LangGraph deprecation warning in logs | Non-blocking for this demo; continue unless the agent logs an exception or exits. |

---

## Sidebar controls — quick reference

| Control | What it does |
|---|---|
| ▶ / ⏸ / ⏹ | Start, pause, or stop simulator-driven inventory drain |
| ⚡ Drain Speed | Multiplies simulated consumption rate |
| 🔍 Supplier Search | Demonstrates Atlas Search / `$rankFusion` supplier capability retrieval |
| 📊 Demo Status | Shows alert, decision, auto-approval, escalation, failure, and memory counts |
| 🔄 Reset Demo | Clears operational data and checkpoint state |
| 🎯 Prepare Context Scenario | Creates deterministic MED-3017 context-engineering demo path |
| 🛠 Admin Panel | Circuit reset, procedure extraction, memory compaction, recovery log |
