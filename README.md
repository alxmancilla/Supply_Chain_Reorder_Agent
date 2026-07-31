# Supply Chain Reorder Alert Agent

An AI agent that monitors healthcare inventory in real time, detects stockouts, and autonomously drafts purchase orders — with human approval for anything above the auto-approve threshold.

**Stack:** MongoDB Atlas · LangGraph · LangChain · Grove OpenAI-compatible LLM · Voyage AI (embeddings + reranking, both pluggable)

> **Demo project.** All inventory data is simulated. Designed to showcase MongoDB Atlas capabilities in an agentic workflow. Production-hardening sections demonstrate patterns, not a complete production procurement system.

---

## How it works

```
Simulator → reorder_alerts (MongoDB) → Agent (LangGraph) → proposed_orders → Dashboard
```

The simulator writes a reorder alert whenever stock falls below the reorder point. The agent picks it up via a **Change Stream**, runs a 5-node pipeline, and either auto-approves the order or pauses for human review.

### Agent pipeline

| Node | What it does |
|---|---|
| `assess_alert` | Reads live inventory; calculates coverage gap and urgency |
| `route_by_urgency` | Zero-gap alerts skip the LLM; all others start the recommendation flow |
| `retrieve_context` | ReAct loop: queries Atlas Search, Vector Search, time-series trend, and memory layers |
| `analyze_suppliers` | Ranks suppliers by fill rate, lead time, FDA compliance, and cost |
| `draft_recommendation` | Calculates order quantity and selects best supplier |
| `save_order` | Writes the order. **Auto-approves** if confidence is `high` and cost < $5,000; otherwise pauses for human review via LangGraph `interrupt()` |
| `write_memories` | Persists the decision to short-term memory (TTL 24h), long-term semantic memory, and order history |
| `escalate` | Writes an escalation record if the agent cannot produce a valid recommendation |

Full diagrams: [diagrams/agent_diagrams.md](diagrams/agent_diagrams.md)

### Human-in-the-Loop

When an order needs review, `save_order` calls `interrupt()` — LangGraph serialises the full graph state to MongoDB (`checkpoints` collection) and pauses. The dashboard shows a review card with an Approve / Reject button.

- **Approve** → `asyncio.run(_agent_graph.ainvoke(Command(resume={"approved": True, ...})))` resumes from the exact pause point.
- **Reject** → alert resets to `"pending"`; agent reprocesses with the rejection visible in memory (`⚠ HUMAN REJECTED` tag).
- **3 rejections** → alert escalates automatically.

> Open the `checkpoints` collection in Atlas while a graph is paused to see the entire agent state serialised in a single document.

### Auto-approve criteria

| Condition | Threshold |
|---|---|
| Confidence | `high` |
| Order cost | < $5,000 |

`high` confidence requires: ≥5 days stock remaining, stable/falling trend, preferred supplier fill rate ≥95%, and ≥14 days of consumption history. Anything below that routes to human review.

---

## Memory Layers

| Layer | Collection | How it works |
|---|---|---|
| **Short-term** (episodic) | `short_term_memory` | One record per decision; TTL 24h. Tags injected into the next prompt: `⚠ HUMAN REJECTED`, `✓ HUMAN APPROVED`, `AUTO-APPROVED`. Also acts as a shared coordination bus between concurrent workers. |
| **Long-term** (semantic) | `agent_memory` | Natural-language summaries embedded via the configured provider (default: `voyage-4-large`). Retrieved via `$vectorSearch` and optionally refined via the reranking layer. `memory_compactor.py` deduplicates and summarises old entries. |
| **Semantic precedents** | `order_history` | Historical order archive with vector embeddings. Retrieved by `find_similar_past_orders` via `$vectorSearch` and optionally refined via the reranking layer (see `RERANKER_ENABLED`). |
| **Procedural** | `procedures` | Rules extracted by `procedure_extractor.py` when the same supplier is approved ≥5× for the same category+location in 30 days. Require human confirmation before the agent uses them. |
| **Episodic timeline** | `alert_lifecycle` | Full event log per alert: `alert_created → agent_decision → human_approved/rejected → order_placed → stock_recovered`. Surfaced to the agent via `get_episode_history`. |

---

## MongoDB Collections

| Collection | Type | Purpose |
|---|---|---|
| `inventory` | Standard | Current stock positions per SKU + location |
| `suppliers` | Standard | Approved suppliers with lead times and performance metrics |
| `consumption_history` | **Time Series** | 90 days of simulated dispensing events; 3 SKUs have a rising trend |
| `order_history` | Standard | Seeded historical orders with embeddings from the configured provider (Vector Search source) |
| `reorder_alerts` | Standard | Simulator output; triggers the agent via **Change Stream** |
| `proposed_orders` | Standard | Agent output; human approves/rejects via dashboard (or auto-approved) |
| `short_term_memory` | Standard + **TTL** | Agent and human decisions per SKU, auto-deleted after 24 h |
| `agent_memory` | Standard | Long-term semantic summaries of decisions; embedded via the configured provider |
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
- Grove API key + base URL (OpenAI-compatible gateway; code defaults to `gpt-5.4`)
- [Voyage AI](https://www.voyageai.com) API key (default provider for embeddings and, optionally, reranking — see `agent/embeddings.py` and `agent/rerank.py`)

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

# Optional — override the embedding model/dimensions (see agent/embeddings.py)
EMBEDDING_MODEL=voyage-4-large
EMBEDDING_DIMS=1024

# Optional — enable the reranking layer (see agent/rerank.py)
RERANKER_ENABLED=0
RERANKER_MODEL=rerank-2
```

`agent/embeddings.py` wraps the embedding provider behind LangChain's `Embeddings` interface, and `agent/rerank.py` provides a pluggable reranking layer. Changing models is a configuration change in `.env`. Swapping SDKs (e.g., away from Voyage AI) involves implementing new subclasses in those files. If you change the embedding dimensions, re-run `python data/seed.py` to regenerate the Atlas Vector Search indexes.

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

If `auth/users.yaml` is absent, the dashboard runs unauthenticated with full admin access (demo mode). If the file is present but invalid, the dashboard fails closed instead of granting admin access.

### 4. Seed MongoDB

Run once to create and populate all collections, generate embeddings via the configured provider, and create Atlas Search and Vector Search indexes.

```bash
python data/seed.py
```

> **Re-running seed:** drops and recreates all operational collections for a clean demo state.

---

## Running the Demo

### Option A — Docker Compose (recommended)

```bash
docker compose up --build
# or run in the background:
docker compose up --build -d
```

Opens at **http://localhost:8501**. Stop with `docker compose down`.

Quick smoke checks after startup:

```bash
docker compose ps
docker compose logs agent
```

Expected healthy state: `app`, `agent`, `simulator`, and `memory-retry-worker` are `Up`; `seeder` has exited successfully; the dashboard responds at `http://localhost:8501`. The agent may immediately process seeded below-ROP alerts and create awaiting-approval orders before you prepare the deterministic MED-3017 scenario.

Services started:
| Service | What it does |
|---|---|
| `seeder` | Seeds MongoDB once; exits 0 on success |
| `agent` | Runs the LangGraph pipeline; watches Change Stream |
| `memory-retry-worker` | Polls `failed_memory_writes` every 30 s; retries up to 3× |
| `simulator` | Drains inventory on a timer and writes reorder alerts when stock falls below ROP |
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
| Prepare Context Scenario | Pauses the simulator and creates the deterministic `MED-3017 @ DC-Texas` demo path with live-state, memory, vector-precedent, and procedural-rule context |
| Extract Rules | Runs `procedure_extractor.py` — scans `proposed_orders` for patterns (same supplier approved ≥5× in 30 days for same category+location) and writes candidate rules to `procedures` |
| Compact Memory | Runs `memory_compactor.py` — deduplicates near-identical `agent_memory` entries (cosine similarity > 0.95) and summarises old entries |
| Procedure candidate review | Expander listing `procedures` docs where `human_confirmed: False`; per-rule ✅ Confirm and 🗑 Dismiss buttons; confirmed rules are passed to the agent via `get_applicable_procedures` tool |
| 🔁 Agent Recovery Log | Expander that reads the `checkpoints` collection (LangGraph `MongoDBSaver`), groups by `thread_id` (= alert `_id`), and shows each pipeline run: SKU, location, last node executed, step count, and run outcome (completed ✅ / re-queued 🔄 / escalated 🔺 / recovered mid-pipeline ⚡) |

### Human-in-the-Loop flow

When `save_order` determines an order needs human review it calls `interrupt()`. LangGraph writes the full graph state to `checkpoints` (MongoDBSaver) and the graph pauses. The dashboard surfaces the review card.

**Approve:**
1. Dashboard calls `asyncio.run(_agent_graph.ainvoke(Command(resume={"approved": True, "approver": "jsmith"})))`.
2. LangGraph reloads the checkpoint, re-enters `save_order` after the `interrupt()` call.
3. `proposed_orders.status` → `"approved"`; `inventory.on_order` incremented.
4. Graph continues to `write_memories` — short-term and long-term memory reflect the human approval.

**Reject:**
1. Dashboard calls `asyncio.run(_agent_graph.ainvoke(Command(resume={"approved": False, "reason": "wrong supplier"})))`.
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
- `RecommendationOutputSchema` — schema reference for recommendation validation

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

Current local baseline in this workspace: `97 passed` when `MONGODB_URI` points to a reachable Atlas cluster (includes live DB, integration, and reranking-mock coverage). `requires_db`-marked tests are skipped automatically when Atlas is unreachable.

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
| Auto-approve thresholds | `test_nodes/test_save_and_notify.py` | Production `agent.order_policy` behavior for high-confidence approval, zero-gap path, cost threshold, and approval idempotency gate |
| Inline validation | `test_nodes/test_audit_agent.py` | Recommendation validation rules, including supplier allow-list and FDA hard filter |
| Consumption trend | `test_tools/test_consumption_trend.py` | Avg daily calculation, trend direction |
| Memory read/write | `test_tools/test_memory_read_write.py` | Short-term insert, read, TTL filtering |
| Memory payloads | `test_tools/test_memory_payloads.py` | Human rejection fields persisted for future context |
| Context manifest | `test_tools/test_context_manifest.py` | Context source summary and guardrail manifest shape |
| Context budget | `test_tools/test_context_budget.py` | Token counting, trim priority order |
| Prompt rendering | `test_prompts/test_build_prompt.py`, `test_memory_tags.py` | Tag formatting, prompt structure |
| SKU lock | `test_coordination/test_sku_lock.py` | Acquire/release logic, expired-lock cleanup, concurrent exclusivity (real DB) |
| Offline graph path | `test_nodes/test_graph_offline.py` | Compiles the real topology without MongoDBSaver and verifies zero-gap routing skips recommendation |
| Alert helpers | `test_tools/test_alert_helpers.py` | Canonical alert shape, lifecycle append, dead-letter behavior |
| Reranking | `test_tools/test_reranking.py` | `find_similar_past_orders` / `get_long_term_memories` call the configured reranker and respect its ordering/limit |
| End-to-end pipeline | `integration/test_full_pipeline.py` | Insert alert → graph → assert proposed_order saved (requires MongoDB and graph dependencies) |

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
├── DEMO_SCRIPT.md                   # 5-minute demo walkthrough with talking points
├── diagrams/
│   └── agent_diagrams.md            # workflow and dashboard architecture diagrams
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
│   ├── embeddings.py                # embedding provider abstraction (swap models/SDKs here)
│   ├── rerank.py                    # pluggable reranking layer (Voyage Rerank, LLM, etc.)
│   ├── alerts.py                    # shared reorder-alert build/insert/lifecycle helpers
│   ├── order_policy.py              # pure order-decision policy (auto-approve, budget, zero-gap)
│   ├── prompts.py                   # LLM prompt templates
│   ├── schemas.py                   # Pydantic validation schemas
│   ├── logger.py                    # structured JSON logger
│   ├── context_budget.py            # token budget trimming for prompt assembly
│   ├── sku_lock.py                  # distributed SKU-level lock (unique index + TTL)
│   ├── procedure_extractor.py       # derives procedural rules from approval patterns
│   ├── memory_compactor.py          # deduplicates and summarises agent_memory
│   └── memory_retry_worker.py       # retries failed memory writes
├── tests/
│   ├── conftest.py                  # fixtures: test_db, mock LLM, mock embedding provider, mock reranker
│   ├── test_nodes/
│   │   ├── test_assess_alert.py     # coverage gap + expedite logic
│   │   ├── test_save_and_notify.py  # production order-policy tests
│   │   ├── test_audit_agent.py      # inline recommendation validation
│   │   └── test_graph_offline.py    # mocked graph topology path without Atlas
│   ├── test_tools/
│   │   ├── test_consumption_trend.py
│   │   ├── test_context_budget.py
│   │   ├── test_context_manifest.py
│   │   ├── test_memory_payloads.py
│   │   ├── test_memory_read_write.py
│   │   ├── test_alert_helpers.py
│   │   └── test_reranking.py
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
| **Atlas Vector Search** | Category-filtered search on `order_history`; location-filterable search on `agent_memory`; both over-fetch candidates and optionally refine via the pluggable reranking layer (`agent/rerank.py`, `RERANKER_ENABLED`) |
| **FDA regulatory hard filter** | `recommend` calls `validate_recommendation()` in `agent/tools.py` after each LLM call; pharmaceutical and laboratory SKUs are rejected and retried if the supplier does not have `fda_registered: True`; constraint is also stated in `ANALYSIS_SYSTEM_PROMPT` |
| **ROP health check** | `app.py` aggregates 30-day average consumption from `consumption_history` and best lead time from `suppliers` in two batch queries; each inventory card shows a ✅ / ⚠️ / 🔴 ROP health indicator; a **📐 ROP Health** KPI card counts at-risk SKUs |
| **LangGraph MongoDBSaver** | Checkpoints full graph state to `checkpoints` per alert invocation; also the pause-point when `interrupt()` fires — open Atlas during a human-review pause to see the entire agent state frozen in one document |
| **`interrupt()` / `Command(resume=...)`** | LangGraph native HITL primitive used in `save_order`; pauses the graph at the node level; resumed by the dashboard calling `asyncio.run(_agent_graph.ainvoke(Command(resume={...})))` with the human's decision |
| **Atomic `findOneAndUpdate`** | Alert claim (pending → processing) prevents two workers racing on the same document |

> **Note on Atlas Stream Processing:** In production, `stream_simulator.py` would be replaced by an Atlas Stream Processing pipeline consuming from Apache Kafka. The agent code is identical regardless — it only sees the `reorder_alerts` collection.

---

## Supply Chain Domain Features

1. **`$rankFusion` hybrid search** — supplier search uses two independent `$search` pipelines (name-match and capability-notes) merged via Reciprocal Rank Fusion, giving more robust ranking than a single compound query. Falls back to compound `$search` on M0 tier.

2. **Budget authority thresholds** — the auto-approve ceiling is `$5,000` (module constant `_AUTO_APPROVE_MAX_USD = 5_000.00` in `agent/graph.py`). Orders at or above this threshold are routed to human review: `save_order` calls `interrupt()`, the graph pauses, and the dashboard resumes it with `Command(resume={"approved": True})`. The `review_reason` field on the order document is surfaced as a `⚠ REVIEW` badge in the dashboard.

3. **FDA regulatory enforcement** — `validate_recommendation()` in `agent/tools.py` rejects any recommendation that assigns a non-FDA-registered supplier to a pharmaceutical or laboratory SKU. The `recommend` node retries inline on violation. Three suppliers (SUP-004, SUP-010, SUP-014) have `fda_registered: False` in the seeded data.

4. **ROP health monitoring** — each inventory card shows a per-SKU reorder-point health indicator (✅ covered / ⚠️ safety-stock bridge / 🔴 undercalibrated). A KPI card counts at-risk SKUs across the full inventory. Computed via two O(1) batch aggregations per dashboard refresh.

5. **Agent Recovery Log** — the Admin Panel exposes a 🔁 Agent Recovery Log that reads LangGraph `MongoDBSaver` checkpoints, groups them by alert, and shows each pipeline run's outcome (completed ✅ / re-queued 🔄 / escalated 🔺 / recovered mid-pipeline ⚡), making crash-recovery tangible in the UI.
