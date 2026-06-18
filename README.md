# Supply Chain Reorder Alert Agent

A demonstration supply chain reorder alert agent built with **MongoDB Atlas**, **LangChain**, **LangGraph**, **GPT-4o** (via Grove API gateway), and **Voyage AI** embeddings.

The agent monitors simulated inventory events, detects when stock drops below a reorder point, and coordinates a multi-agent pipeline to evaluate suppliers, draft purchase orders with plain-English rationale, validate output, and escalate edge cases — all in real time.

> **This is a demo.** Data is simulated. Prioritises clarity and MongoDB feature visibility over production robustness.

---

## Architecture

```
stream_simulator.py          agent/graph.py                              app.py
──────────────────     →     ────────────────────────────────     →     ──────────────────
Simulates Atlas              LangGraph multi-agent state machine          Streamlit dashboard
Stream Processing            watches reorder_alerts via Change            shows live inventory,
                             Stream, coordinates specialist               alerts, and agent
Writes reorder_alerts        agents, saves proposed_orders,              decisions with
to MongoDB every 5 s         reads/writes dual-layer memory              Approve/Reject buttons
```

For a Mermaid data-flow diagram see [DASHBOARD_ARCHITECTURE.md](DASHBOARD_ARCHITECTURE.md).
For node-by-node agent diagrams see [WORKFLOW.md](WORKFLOW.md).

---

## Multi-Agent Graph Topology

```
START
  → assess_alert
  → route_by_urgency (conditional)
      ↘ save_order    (zero-gap fast path — no LLM)
      ↘ recommend     (retrieval + analysis + recommendation + validation)
          ↘ escalate  (validation exhausted after all retries)
          ↘ save_order
              → write_memories    (short-term, long-term, history)
END
```

| Node | What it does |
|---|---|
| `assess_alert` | Fetches live inventory position from MongoDB; sums active orders to compute `coverage_gap`; sets `expedite` flag when `< 2 days` remaining |
| `route_by_urgency` | Conditional edge — zero-gap alerts skip the LLM entirely; everything else goes to `recommend` |
| `recommend` | **The core reasoning node.** Runs a ReAct tool-calling loop to gather context (Atlas Search, Vector Search, time series, memory), then calls the LLM twice — once for supplier analysis (confidence score) and once for the order recommendation (quantity + rationale). Validates inline; retries up to 3× before setting `escalate_flag` |
| `save_order` | Writes `proposed_orders`; **auto-approves** if confidence is `high` and cost `< $5,000`; otherwise calls **`interrupt()`** — LangGraph freezes the graph state to the `checkpoints` collection (MongoDBSaver) and waits for a human decision; the dashboard resumes with `Command(resume={...})` |
| `write_memories` | Runs after `save_order` completes; writes short-term memory (24 h TTL), long-term semantic embedding (`agent_memory`), and order history in parallel |
| `escalate` | Writes to `escalation_queue`; sets alert status to `"escalated"`; optionally POSTs a webhook notification |

Every graph invocation is checkpointed to MongoDB via **LangGraph `MongoDBSaver`** (`checkpoints` collection).

### ReAct Retrieval Agent — available tools

| Tool | MongoDB feature | Purpose |
|---|---|---|
| `get_inventory_position` | Basic `find` | Live on-hand, on-order, reorder-point |
| `get_supplier_options` | `find`, sort by fill rate | Approved suppliers for the SKU |
| `get_consumption_trend` | **Time Series** aggregation | 14-day avg daily demand + trend direction |
| `search_suppliers_by_capability` | **Atlas Search** (`$rankFusion`) | RRF hybrid search combining a `name_match` pipeline (0.4 weight) and a `capability_match` pipeline (0.6 weight); falls back to compound `$search` on M0 tier |
| `find_similar_past_orders` | **Atlas Vector Search** (`$vectorSearch`) on `order_history` | Semantically similar historical orders as precedent (score ≥ 0.78) |
| `get_episode_history` | `find` on `alert_lifecycle` | Full event timeline for the 3 most recent past alerts for this SKU+location |

Short-term and long-term memories are injected directly into the recommendation agent's prompt, not fetched via tools.

---

## Real-Time Coordination

### The problem

Between the moment `assess_alert` reads the inventory gap and the moment `save_order` writes the order (~20–30 s of LLM pipeline), a second alert for the same SKU+location could arrive, pass the same gap check, and result in a duplicate order. This TOCTOU (time-of-check/time-of-use) race is real when multiple workers run concurrently.

### The fix — two layers of mutual exclusion

**1. Atomic alert claim (`pending → processing`)**

In `_process_one_alert`, before invoking the graph, the alert status is transitioned atomically:

```python
claimed = await _db_async.reorder_alerts.find_one_and_update(
    {"_id": alert_oid, "status": "pending"},
    {"$set": {"status": "processing"}},
)
```

Only one worker can win this update. If `claimed` is `None`, the alert was already taken — skip without processing.

**2. SKU-level distributed lock (`agent/sku_lock.py`)**

After claiming the alert, the worker acquires an exclusive MongoDB-backed lock for `(sku, location)`:

- A unique compound index on `sku_processing_locks.{sku, location}` makes the `insert_one` atomic — only one `insert_one` can succeed.
- If the lock is held, the worker retries with exponential backoff (1 s → 2 s → 4 s → 8 s → 16 s, up to ~47 s total).
- If the lock cannot be acquired after all retries, the alert is reset to `"pending"` — the Change Stream re-fires it.
- A TTL index on `expires_at` (5-minute window) auto-expires stale locks from crashed workers.
- `release()` scopes its `delete_one` to `held_by` — a late release from a crashed worker can never evict a lock held by a different alert.

```
Worker A                          Worker B
─────────────────────────────     ─────────────────────────────
claim alert-1 ✓                   claim alert-2 ✓
acquire SKU lock ✓                acquire SKU lock ✗ (retrying…)
assess_alert → coverage_gap=155
[LLM pipeline ~20 s]
save_order → order written
release SKU lock                  acquire SKU lock ✓
                                  assess_alert → coverage_gap=0
                                  zero-gap fast path → no order
                                  release SKU lock
```

### Auto-approve

Orders are automatically set to `status: approved` when **both** conditions are met:

| Condition | Value |
|---|---|
| Agent confidence | `high` |
| Total order cost | `< $5,000.00` |

Zero-quantity orders (zero-gap fast path) are always auto-approved regardless of cost.

Orders that do not meet both conditions are routed to human review. `save_order` calls LangGraph's **`interrupt()`**, which serialises the full graph state to the `checkpoints` collection (MongoDBSaver) and pauses the graph. The dashboard detects `status: awaiting_approval`, surfaces the review card, and on button click calls `graph.invoke(Command(resume={"approved": True}))` to reload the checkpoint and continue the graph from where it stopped. The order document carries a `review_reason` field (`"budget_threshold ($X ≥ $5,000 limit)"` or `"confidence=medium"`) shown as a `⚠ REVIEW: <reason>` badge in the dashboard.

> **Teaching moment:** while a graph is paused you can open the `checkpoints` collection in Atlas and see the entire agent state frozen in a single document — suppliers, retrieval trace, recommendation, memories — all serialised by MongoDBSaver.

### Confidence level criteria

| Level | Conditions |
|---|---|
| `high` | **ALL of:** ≥5 days remaining · trend stable or decreasing · preferred supplier fill rate ≥95% confirmed by Atlas Search · ≥14 days of consumption history sampled |
| `medium` | Does not qualify for `high` but no critical flags |
| `low` | **ANY of:** <2 days remaining · <7 days of history · preferred supplier fill rate <85% · trend unknown · no supplier confirmed by Atlas Search |

### Escalation Policy

An alert is **escalated** (status = `"escalated"`, written to `escalation_queue`) if:
- A human rejects the same alert **3 or more times** — detected in `_process_one_alert` before the graph runs
- The **audit agent** reaches max retries (2) without a valid recommendation

Escalated alerts appear in the dashboard "Escalated Alerts" section. An optional `ESCALATION_WEBHOOK_URL` environment variable triggers an HTTP POST on escalation.

### Circuit Breaker

The LLM client tracks consecutive failures. After **3 consecutive failures** the circuit opens:
- New alerts are written to `human_review_queue` instead of being processed
- The dashboard sidebar shows a warning banner with a **Reset Circuit Breaker** button (admin only)
- The circuit resets automatically on the next successful LLM call

---

## Memory Layers

### Short-term (`short_term_memory`)

One record per `write_memories` run (after auto-approval or after a human decision is received via `interrupt()` resume). Includes `decided_by`, `human_decision`, supplier, quantity, confidence. Rendered in the recommendation agent prompt with visual tags:
- `⚠ HUMAN REJECTED` — agent must change approach
- `✓ HUMAN APPROVED` — strong reuse signal
- `AUTO-APPROVED` — agent's own prior decisions

TTL index auto-purges entries older than 24 h.

### Long-term (`agent_memory`)

Natural-language summaries embedded with Voyage AI `voyage-4-large` (1024 dims). Retrieved via `$vectorSearch` (score ≥ 0.72, location pre-filter). Human override summaries are prefixed distinctly so they surface as separate precedents.

`memory_compactor.py` deduplicates near-duplicate entries (cosine similarity > 0.95) and summarises groups older than 30 days to prevent unbounded growth.

### Procedural (`procedures`)

Candidate rules derived from repeated approval patterns (≥5 approvals of same supplier for same category+location in 30 days). Rules are written with `human_confirmed: false` and must be confirmed via the Admin Panel before the agent uses them.

### Episodic (`alert_lifecycle`)

Full event timeline per alert: `alert_created` → `agent_decision` → `human_approved` / `human_rejected` → `order_placed` → `stock_recovered`. The `get_episode_history` tool surfaces the 3 most recent complete episodes to the recommendation agent.

---

## MongoDB Collections

| Collection | Type | Purpose |
|---|---|---|
| `inventory` | Standard | Current stock positions per SKU + location |
| `suppliers` | Standard | Approved suppliers with lead times and performance metrics |
| `consumption_history` | **Time Series** | 90 days of simulated dispensing events; 3 SKUs have a rising trend |
| `order_history` | Standard | Seeded historical orders with Voyage AI embeddings (Vector Search source) |
| `reorder_alerts` | Standard | Simulator output; triggers the agent via **Change Stream** |
| `proposed_orders` | Standard | Agent output; human approves/rejects via dashboard (or auto-approved) |
| `short_term_memory` | Standard + **TTL** | Agent and human decisions per SKU, auto-deleted after 24 h |
| `agent_memory` | Standard | Long-term semantic summaries of decisions; embedded with Voyage AI |
| `checkpoints` | Standard | LangGraph `MongoDBSaver` state snapshots |
| `alert_lifecycle` | Standard | Full event timeline per alert (created → decision → approval → stock recovery) |
| `confidence_outcomes` | Standard | Predicted confidence vs actual outcome — powers the Confidence Calibration tab |
| `escalation_queue` | Standard | Alerts that could not be resolved and were escalated |
| `sku_processing_locks` | Standard + **TTL** | Distributed lock per `(sku, location)`; unique index enforces exclusivity; TTL expires stale locks after 5 min |
| `failed_memory_writes` | Standard | Memory write failures that could not be committed; retried by `memory_retry_worker.py` |
| `human_review_queue` | Standard | Alerts queued for human processing when the LLM circuit breaker is open |
| `dead_letter_events` | Standard | Validation-rejected Kafka events and malformed alert documents |
| `procedures` | Standard | Procedural preference rules derived from repeated approval patterns |
| `agent_state` | Standard | Change Stream resume token — persisted after every event |

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (recommended)
  _or_ Python 3.12+ with a virtual environment
- [MongoDB Atlas](https://www.mongodb.com/atlas) cluster (M0 free tier works; Atlas Search and Vector Search must be enabled)
- Grove API key + base URL (Azure OpenAI gateway providing GPT-4o)
- [Voyage AI](https://www.voyageai.com) API key (for `voyage-4-large` embeddings)

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd Supply_Chain_Reorder_Agent
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/
GROVE_API_KEY=<your-grove-api-key>
GROVE_API_BASE_URL=https://grove-gateway-prod.azure-api.net/grove-foundry-prod/openai/v1
VOYAGE_API_KEY=<your-voyage-api-key>

# Optional
FALLBACK_MODEL=gpt-4o
FALLBACK_API_BASE_URL=<alternate-openai-base>
ESCALATION_WEBHOOK_URL=<webhook-url>
LOG_LEVEL=INFO
EXPLAIN_MODE=1           # print plain-English node descriptions (learning/demo aid)
```

### 3. (Optional) Configure dashboard authentication

```bash
cp auth/users.yaml.example auth/users.yaml
```

Edit `auth/users.yaml` with hashed credentials. When `auth/users.yaml` is present, the dashboard requires login and enforces three roles:

| Role | Can do |
|---|---|
| `viewer` | Read-only: view dashboard, inventory, orders |
| `approver` | All above + approve/reject orders |
| `admin` | All above + Admin Panel (Extract Rules, Compact Memory, reset circuit breaker) |

If `auth/users.yaml` is absent, the dashboard runs unauthenticated with full admin access (demo mode).

### 4. Seed MongoDB

Run once to create and populate all collections, generate Voyage AI embeddings, and create Atlas Search and Vector Search indexes.

```bash
python data/seed.py
```

> **Re-running seed:** drops and recreates all operational collections for a clean demo state.

---

## Running the Demo

### Option A — Docker Compose (recommended)

```bash
docker compose up --build
```

Opens at **http://localhost:8501**. Stop with `docker compose down`.

Services started:
| Service | What it does |
|---|---|
| `seeder` | Seeds MongoDB once; exits 0 on success |
| `agent` | Runs the LangGraph pipeline; watches Change Stream |
| `memory-retry-worker` | Polls `failed_memory_writes` every 30 s; retries up to 3× |
| `simulator` | Writes a reorder alert every 5 s |
| `app` | Streamlit dashboard on port 8501 |

### Option B — Kafka mode

Uses a local Kafka broker instead of the simulator. Atlas Stream Processing would replace the consumer in production.

```bash
export HOST_IP=$(curl -s ifconfig.me)   # public IP so Atlas ASP can reach port 9094
docker compose --profile kafka up --build
```

Additional services added:
| Service | What it does |
|---|---|
| `kafka` | Bitnami KRaft broker (no ZooKeeper); ports 9092 (internal) + 9094 (external) |
| `kafka-init` | Creates `wms-inventory-events` topic (3 partitions) |
| `kafka-consumer` | Reads from Kafka; writes `reorder_alerts` to MongoDB |

Send a test event:
```bash
python kafka/producer.py
```

Stop the simulator from the dashboard before switching to Kafka mode to avoid duplicate alerts.

### Option C — Manual (multiple terminals)

```bash
source .venv/bin/activate

# Terminal 1 — stream simulator
python simulator/stream_simulator.py

# Terminal 2 — agent (add EXPLAIN_MODE=1 for plain-English node descriptions)
python agent/graph.py
# or: EXPLAIN_MODE=1 python agent/graph.py

# Terminal 3 — dashboard
streamlit run app.py
```

Within ~30 seconds you should see inventory levels ticking down, reorder alerts appearing, and proposed orders being drafted.

---

## Dashboard (`app.py`)

The dashboard auto-refreshes every **5 seconds**.

### Column 1 — Live Inventory

- Critical items (below reorder point) sort to the top.
- Stock level progress bar with `⚠️ Below reorder` warning.

### Column 2 — Active Alerts

- Urgency tags: 🔴 Critical (`< 2 days`), 🟡 Low stock (`< 7 days`), 🟠 Watch (`≥ 7 days`).
- Time-ago labels. Processing indicator while the agent is working.

### Column 3 — Agent Decisions

- Awaiting-approval orders sort to the top.
- Confidence progress bar. `⚡ Auto-approved` badge for auto-approved orders; `⚠ REVIEW: <reason>` badge on orders held for human review (budget-threshold or low-confidence routing).
- Expandable Agent Rationale and Similar Past Orders sections.
- Approve / Reject buttons (requires `approver` or `admin` role when auth is enabled).

Each inventory card also shows a per-SKU ROP health indicator:
- ✅ green: ROP fires at X days, lead time Y days — covered
- ⚠️ amber: safety stock bridges the gap
- 🔴 red: undercalibrated by X days

A **📐 ROP Health** KPI card (7th card in the KPI row) counts the total number of SKUs where the reorder point fires before the preferred supplier can deliver.

### Escalated Alerts section

Appears when any alert has been escalated. Shows SKU, location, rejection count, and escalation timestamp.

### Confidence Calibration expander

Aggregation table showing how well predicted confidence (`high`/`medium`/`low`) correlates with actual outcomes (`resolved`/`pending`/`escalated`). Populated as the agent processes alerts and humans make decisions.

### Sidebar

**Simulator Controls** — Start / pause / stop + drain speed slider (1× to 10×).

**Supplier Search** — `$rankFusion` hybrid search combining two independent `$search` sub-pipelines: `name_match` (weight 0.4) for exact supplier-name hits and `capability_match` (weight 0.6) for fuzzy capability-notes matching, merged via Reciprocal Rank Fusion. Falls back to compound `$search` on M0 tier. Source label shows **🔀 $rankFusion**.

**Demo Status** — Live counters: decisions, auto-approved %, escalations, failed memory writes, long-term memories.

**Reset Demo** — Clears all operational collections and resets simulator to 1× speed. Inventory and suppliers are preserved.

**Admin Panel** (admin role only)

| Control | What it does |
|---|---|
| Circuit Breaker status / Reset | Shows failure count; reset button clears the counter and resumes LLM calls |
| Extract Rules | Runs `procedure_extractor.py` — scans `short_term_memory` for patterns (same supplier approved ≥5× in 30 days for same category+location) and writes candidate rules to `procedures` |
| Compact Memory | Runs `memory_compactor.py` — deduplicates near-identical `agent_memory` entries (cosine similarity > 0.95) and summarises old entries |
| Procedure candidate review | Expander listing `procedures` docs where `human_confirmed: False`; per-rule ✅ Confirm and 🗑 Dismiss buttons; confirmed rules are passed to the agent via `get_applicable_procedures` tool |
| 🔁 Agent Recovery Log | Expander that reads the `checkpoints` collection (LangGraph `MongoDBSaver`), groups by `thread_id` (= alert `_id`), and shows each pipeline run: SKU, location, last node executed, step count, and run outcome (completed ✅ / re-queued 🔄 / escalated 🔺 / recovered mid-pipeline ⚡) |

### Human-in-the-Loop flow

When `save_order` determines an order needs human review it calls `interrupt()`. LangGraph writes the full graph state to `checkpoints` (MongoDBSaver) and the graph pauses. The dashboard surfaces the review card.

**Approve:**
1. Dashboard calls `graph.invoke(Command(resume={"approved": True, "approver": "jsmith"}))`.
2. LangGraph reloads the checkpoint, re-enters `save_order` after the `interrupt()` call.
3. `proposed_orders.status` → `"approved"`; `inventory.on_order` incremented.
4. Graph continues to `write_memories` — short-term and long-term memory reflect the human approval.

**Reject:**
1. Dashboard calls `graph.invoke(Command(resume={"approved": False, "reason": "wrong supplier"}))`.
2. `proposed_orders.status` → `"rejected"`; `rejection_count` on the alert incremented.
3. Alert resets to `"pending"` — the Change Stream fires again and the agent reprocesses with the rejection visible in short-term memory (`⚠ HUMAN REJECTED` tag).
4. `write_memories` is still called, writing the rejection to both memory layers as a persistent learning signal.
5. If `rejection_count >= 3`, the next processing cycle escalates directly without running the graph.

---

## Production Hardening

### Input Validation (Pydantic)

All system boundaries validate incoming data with `agent/schemas.py`:
- `ReorderAlertSchema` — validates alert documents before graph invocation
- `KafkaInventoryEventSchema` — validates Kafka messages in `consumer.py`
- `RecommendationOutputSchema` — used by the audit agent

Validation failures are written to `dead_letter_events` and logged at ERROR level.

### Failed Memory Write Recovery

If a memory write fails, `agent/tools.py` records the failure to `failed_memory_writes` instead of silently dropping it. `agent/memory_retry_worker.py` polls every 30 s and retries failed writes up to 3 times with backoff, marking them `resolved: true` on success. The dashboard header shows the count of unresolved failures.

### Structured Logging

All `print()` calls replaced with `agent/logger.py` — emits JSON lines with fields: `timestamp`, `level`, `module`, `phase`, `alert_id`, `sku`, `location`, `event`, `duration_ms`. Log level controlled by `LOG_LEVEL` env var (default `INFO`).

### Token Budget Management

`agent/context_budget.py` enforces a 6,000-token budget on context sections before they are injected into the recommendation agent prompt. Sections are trimmed in priority order (lowest-priority dropped first):

1. `retrieval_trace` (dropped entirely)
2. `long_term_memories` (lowest-score entries dropped)
3. `similar_orders` (lowest similarity dropped)
4. `short_term_memories` (oldest entries summarised)
5. `supplier_search_results` (capped at 3)

Token counting uses `tiktoken` when available, falling back to a word-count heuristic.

---

## Tests

```bash
python3 -m pytest tests/ -v
```

Tests that require a live MongoDB Atlas connection are automatically skipped when the cluster is unreachable (`requires_db` marker). All other tests run without any external dependencies.

```bash
# Only offline unit tests (no DB, no LLM)
python3 -m pytest tests/ -v -m "not requires_db"

# Only the coordination / lock tests
python3 -m pytest tests/test_coordination/ -v
```

### Test coverage

| Module | Location | What is tested |
|---|---|---|
| Coverage gap logic | `test_nodes/test_assess_alert.py` | Gap calculation, expedite flag, boundary conditions |
| Auto-approve thresholds | `test_nodes/test_save_order.py` | High-confidence auto-approval, zero-gap path, cost threshold, interrupt() pause |
| Recommend validation | `test_nodes/test_recommend.py` | Validation rules, retry logic, escalation flag |
| Consumption trend | `test_tools/test_consumption_trend.py` | Avg daily calculation, trend direction |
| Memory read/write | `test_tools/test_memory_read_write.py` | Short-term insert, read, TTL filtering |
| Context budget | `test_tools/test_context_budget.py` | Token counting, trim priority order |
| Prompt rendering | `test_prompts/test_build_prompt.py`, `test_memory_tags.py` | Tag formatting, prompt structure |
| SKU lock | `test_coordination/test_sku_lock.py` | Acquire/release logic, expired-lock cleanup, concurrent exclusivity (real DB) |
| End-to-end pipeline | `integration/test_full_pipeline.py` | Insert alert → graph → assert proposed_order saved (requires Docker) |

### Coordination tests in detail

`tests/test_coordination/test_sku_lock.py` covers three layers:

**Unit (mocked DB — always runs):**
- `acquire` returns `True` on first insert; writes required fields with future `expires_at`
- Returns `False` when a live lock is held
- Removes expired locks and retries; handles back-to-back race
- `release` scopes `delete_one` to `held_by`; silent no-op when nothing deleted

**DB-backed (real MongoDB):**
- `ensure_indexes` is idempotent
- Sequential acquire / release / reacquire flow
- Different SKUs acquire independently
- Late `release` with wrong `held_by` leaves the live lock intact

**Concurrency (real MongoDB + `asyncio.gather`):**
- 2 concurrent acquires for the same SKU → exactly 1 winner
- 5 concurrent acquires for the same SKU → exactly 1 winner
- The lock document in MongoDB matches the winner's `held_by`

---

## Project Structure

```
Supply_Chain_Reorder_Agent/
├── Dockerfile
├── docker-compose.yml               # simulator mode (default)
├── docker-compose.kafka.yml         # kafka mode overlay
├── .env.example
├── requirements.txt
├── README.md
├── DASHBOARD_ARCHITECTURE.md        # Mermaid data-flow diagram (simulator → Atlas → agent → dashboard)
├── WORKFLOW.md                      # node-by-node agent diagrams (system flow + agent cards + state schema)
├── DEMO_SCRIPT.md                   # 5-minute demo walkthrough with talking points
├── auth/
│   └── users.yaml.example           # copy to users.yaml and fill credentials
├── data/
│   └── seed.py                      # DB setup — run by seeder service
├── simulator/
│   └── stream_simulator.py          # simulates Atlas Stream Processing
├── kafka/
│   ├── consumer.py                  # Kafka → MongoDB reorder_alerts bridge
│   └── producer.py                  # sends test events to wms-inventory-events
├── agent/
│   ├── graph.py                     # LangGraph multi-agent state machine + Change Stream watcher
│   ├── tools.py                     # LangChain @tool functions (inventory, suppliers, memory)
│   ├── prompts.py                   # LLM prompt templates
│   ├── schemas.py                   # Pydantic validation schemas
│   ├── logger.py                    # structured JSON logger
│   ├── context_budget.py            # token budget trimming for prompt assembly
│   ├── sku_lock.py                  # distributed SKU-level lock (unique index + TTL)
│   ├── procedure_extractor.py       # derives procedural rules from approval patterns
│   ├── memory_compactor.py          # deduplicates and summarises agent_memory
│   └── memory_retry_worker.py       # retries failed memory writes
├── tests/
│   ├── conftest.py                  # fixtures: test_db, mock LLM, mock Voyage AI
│   ├── test_nodes/
│   │   ├── test_assess_alert.py     # coverage gap + expedite logic
│   │   ├── test_save_order.py       # auto-approve thresholds, zero-gap path, interrupt() pause
│   │   ├── test_write_memories.py   # memory writes after approval / rejection
│   │   └── test_recommend.py        # validation rules, retry logic, escalation flag
│   ├── test_tools/
│   │   ├── test_consumption_trend.py
│   │   ├── test_memory_read_write.py
│   │   └── test_context_budget.py
│   ├── test_prompts/
│   │   ├── test_build_prompt.py
│   │   └── test_memory_tags.py
│   ├── test_coordination/
│   │   └── test_sku_lock.py         # acquire/release, expired-lock cleanup, concurrency proof
│   └── integration/
│       └── test_full_pipeline.py    # end-to-end: insert alert → run graph → assert order saved
└── app.py                           # Streamlit dashboard
```

---

## SKUs in the Demo

| SKU | Name | Category | Location | Reorder Point | Consumption pattern |
|---|---|---|---|---|---|
| MED-2041 | Amoxicillin 500mg Capsules | pharmaceutical | DC-Ohio | 200 | Stable |
| MED-3017 | Insulin Glargine | pharmaceutical | DC-Texas | 1,000 | **↗ Rising** |
| MED-4490 | Metformin 1000mg | pharmaceutical | DC-Ohio | 1,500 | Stable |
| MED-5502 | Vancomycin 1g IV | pharmaceutical | DC-California | 200 | Stable |
| MED-6201 | Heparin Sodium 5000U/mL | pharmaceutical | DC-Texas | 600 | Stable |
| SURG-0084 | Nitrile Gloves (box) | surgical | DC-Ohio | 100 | **↗ Rising** |
| SURG-1122 | IV Bags 1L | surgical | DC-Texas | 500 | Stable |
| SURG-2244 | N95 Respirator Mask | surgical | DC-California | 400 | Stable |
| DIAG-0331 | Rapid COVID/Flu Combo Test Kit | diagnostic | DC-Ohio | 500 | **↗ Rising** |
| LAB-0112 | Aerobic Blood Culture Bottle | laboratory | DC-Texas | 400 | Stable |

`MED-3017`, `SURG-0084`, and `DIAG-0331` have a linearly rising demand trend in their last 30 days of seeded history. The trend detector reliably returns `trend: increasing` for these SKUs, which prevents `high` confidence even when stock days are adequate.

---

## MongoDB Atlas Features Demonstrated

| Feature | Where used |
|---|---|
| **Document model** | Flexible schema across pharmaceutical, surgical, diagnostic, and laboratory SKUs |
| **Time Series Collection** | `consumption_history` — 90 days of dispensing events; 3 SKUs seeded with a rising trend |
| **Change Streams** | Agent triggers on every insert to `reorder_alerts`; resume token persisted to `agent_state` |
| **Aggregation Pipeline** | 14-day consumption trend, `$facet` for dashboard KPIs, Time Series group-by-day |
| **TTL Index** | `short_term_memory.decided_at` (24 h); `sku_processing_locks.expires_at` (5 min crash-recovery) |
| **Unique Index** | `sku_processing_locks.{sku, location}` — enforces distributed lock exclusivity atomically |
| **Compound Indexes** | `(sku, location, status)` on `proposed_orders`; `(status, created_at)` on `reorder_alerts`; `(sku, location, decided_at)` on `short_term_memory` |
| **Atlas Search** | Supplier capability search via `$rankFusion` (two named `$search` sub-pipelines merged via Reciprocal Rank Fusion); falls back to compound `$search` on M0 tier; powers sidebar Supplier Search |
| **`$rankFusion` / Reciprocal Rank Fusion** | Two independent named `$search` pipelines (`name_match` at 0.4 weight, `capability_match` at 0.6 weight) merged via RRF so a supplier ranking highly in either path rises to the top regardless of raw score scale |
| **Atlas Vector Search** | Category-filtered search on `order_history`; location-filterable search on `agent_memory` |
| **FDA regulatory hard filter** | `recommend` calls `validate_recommendation()` in `agent/tools.py` after each LLM call; pharmaceutical and laboratory SKUs are rejected and retried if the supplier does not have `fda_registered: True`; constraint is also stated in `ANALYSIS_SYSTEM_PROMPT` |
| **ROP health check** | `app.py` aggregates 30-day average consumption from `consumption_history` and best lead time from `suppliers` in two batch queries; each inventory card shows a ✅ / ⚠️ / 🔴 ROP health indicator; a **📐 ROP Health** KPI card counts at-risk SKUs |
| **LangGraph MongoDBSaver** | Checkpoints full graph state to `checkpoints` per alert invocation; also the pause-point when `interrupt()` fires — open Atlas during a human-review pause to see the entire agent state frozen in one document |
| **`interrupt()` / `Command(resume=...)`** | LangGraph native HITL primitive used in `save_order`; pauses the graph at the node level; resumed by the dashboard calling `graph.invoke(Command(resume={...}))` with the human's decision |
| **Atomic `findOneAndUpdate`** | Alert claim (pending → processing) prevents two workers racing on the same document |

> **Note on Atlas Stream Processing:** In production, `stream_simulator.py` would be replaced by an Atlas Stream Processing pipeline consuming from Apache Kafka. The agent code is identical regardless — it only sees the `reorder_alerts` collection.

---

## Supply Chain Domain Features

1. **`$rankFusion` hybrid search** — supplier search uses two independent `$search` pipelines (name-match and capability-notes) merged via Reciprocal Rank Fusion, giving more robust ranking than a single compound query. Falls back to compound `$search` on M0 tier.

2. **Budget authority thresholds** — the auto-approve ceiling is `$5,000` (module constant `_AUTO_APPROVE_MAX_USD = 5_000.00` in `agent/graph.py`). Orders at or above this threshold are routed to human review: `save_order` calls `interrupt()`, the graph pauses, and the dashboard resumes it with `Command(resume={"approved": True})`. The `review_reason` field on the order document is surfaced as a `⚠ REVIEW` badge in the dashboard.

3. **FDA regulatory enforcement** — `validate_recommendation()` in `agent/tools.py` rejects any recommendation that assigns a non-FDA-registered supplier to a pharmaceutical or laboratory SKU. The `recommend` node retries inline on violation. Three suppliers (SUP-004, SUP-010, SUP-014) have `fda_registered: False` in the seeded data.

4. **ROP health monitoring** — each inventory card shows a per-SKU reorder-point health indicator (✅ covered / ⚠️ safety-stock bridge / 🔴 undercalibrated). A KPI card counts at-risk SKUs across the full inventory. Computed via two O(1) batch aggregations per dashboard refresh.

5. **Agent Recovery Log** — the Admin Panel exposes a 🔁 Agent Recovery Log that reads LangGraph `MongoDBSaver` checkpoints, groups them by alert, and shows each pipeline run's outcome (completed ✅ / re-queued 🔄 / escalated 🔺 / recovered mid-pipeline ⚡), making crash-recovery tangible in the UI.
