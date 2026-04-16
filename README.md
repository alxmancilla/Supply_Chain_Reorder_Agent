# Supply Chain Reorder Alert Agent

A demonstration supply chain reorder alert agent built with **MongoDB Atlas**, **LangChain**, **LangGraph**, **GPT-5.4** (via Grove API gateway), and **Voyage AI** embeddings.

The agent monitors simulated inventory events, detects when stock drops below a reorder point, reasons about supplier options and consumption trends, and drafts a purchase order with a plain-English rationale — all in real time.

> **This is a demo.** Data is simulated. Prioritises clarity and MongoDB feature visibility over production robustness.

---

## Architecture

```
stream_simulator.py          agent/graph.py                        app.py
──────────────────     →     ──────────────────────────     →     ──────────────
Simulates Atlas              LangGraph state machine                Streamlit dashboard
Stream Processing            watches reorder_alerts                 shows live inventory,
                             via Change Stream,                     alerts, and agent
Writes reorder_alerts        calls LLM, saves proposed_orders,     decisions with
to MongoDB every 10 s        reads/writes memory layers            Approve/Reject buttons
```

### LangGraph State Machine

```
START → assess_alert → gather_context → reason_and_draft → save_and_notify → END
```

| Node | What it does |
|---|---|
| `assess_alert` | Fetches live inventory position from MongoDB; sums active orders to compute coverage gap |
| `gather_context` | Runs **6 queries concurrently**: supplier list, consumption trend, Atlas Search, Vector Search on order history, short-term memory, long-term memory |
| `reason_and_draft` | Single GPT-5.4 call with full context + memory → produces quantity, supplier, rationale, confidence; retries up to 3× on JSON parse errors (self-correction loop) |
| `save_and_notify` | Writes `proposed_orders`; **auto-approves** if confidence is `high` and cost `< $2,500`; writes short-term and long-term memory entries |

Every graph invocation is checkpointed to MongoDB via **LangGraph `MongoDBSaver`**, giving each alert a persistent state snapshot (collection: `checkpoints`).

### `gather_context` — six concurrent MongoDB queries

| Query | MongoDB feature | Purpose |
|---|---|---|
| `get_supplier_options` | Basic `find`, sort by fill rate | Approved suppliers for the SKU |
| `get_consumption_trend` | **Time Series** aggregation | 14-day avg daily demand + trend direction |
| `search_suppliers_by_capability` | **Atlas Search** (`$search`) | Rank suppliers across the full catalog by text relevance to product name + urgency |
| `find_similar_past_orders` | **Atlas Vector Search** (`$vectorSearch`) on `order_history` | Semantically similar historical orders as LLM precedent (score ≥ 0.78) |
| `get_short_term_memories` | `find` with compound `(sku, location, decided_at)` index | Recent decisions for this SKU in the last 24 h — includes both agent decisions and **human approve/reject overrides** |
| `get_long_term_memories` | **Atlas Vector Search** (`$vectorSearch`) on `agent_memory` | Semantic summaries of agent decisions **and human overrides**, embedded with Voyage AI; grows over time |

### Auto-approve

Orders are automatically set to `status: approved` when **both** conditions are met:

| Condition | Value |
|---|---|
| Agent confidence | `high` |
| Total order cost | `< $2,500.00` |

Any order that fails either condition (medium/low confidence, or high-value spend) is set to `awaiting_approval` for human review. The log clearly states the reason:
- `awaiting approval (confidence not high)`
- `awaiting approval (cost $X ≥ $2,500 threshold)`

#### Confidence level criteria

| Level | Conditions |
|---|---|
| `high` | **ALL of:** ≥5 days remaining · trend stable or decreasing · preferred supplier fill rate ≥95% confirmed by Atlas Search · ≥14 days of consumption history sampled |
| `medium` | Does not qualify for `high` but no critical flags: 2–4 days remaining, OR increasing trend, OR two suppliers closely matched, OR only 7–13 days of history sampled |
| `low` | **ANY of:** <2 days remaining · <7 days of consumption history · preferred supplier fill rate <85% · trend unknown · no supplier confirmed by Atlas Search |

### MongoDB Collections

| Collection | Type | Purpose |
|---|---|---|
| `inventory` | Standard | Current stock positions per SKU + location |
| `suppliers` | Standard | Approved suppliers with lead times and performance metrics |
| `consumption_history` | **Time Series** | 90 days of simulated dispensing events; 3 SKUs have a linearly rising trend |
| `order_history` | Standard | Seeded historical orders with Voyage AI embeddings (Vector Search source); pre-filtered by `category` |
| `reorder_alerts` | Standard | Simulator output; triggers the agent via **Change Stream** |
| `proposed_orders` | Standard | Agent output; human approves/rejects via dashboard (or auto-approved) |
| `short_term_memory` | Standard + **TTL** | Agent and human decisions per SKU, auto-deleted after 24 h |
| `agent_memory` | Standard | Long-term semantic summaries of agent and human decisions; embedded with Voyage AI; grows as agent runs |
| `checkpoints` | Standard | LangGraph `MongoDBSaver` state snapshots — one per alert invocation |
| `agent_state` | Standard | Change Stream resume token — persisted after every event so the agent survives restarts without missing alerts |

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (recommended — runs everything in one command)
  _or_ Python 3.12+ with a virtual environment (container image uses `python:3.12-slim`)
- [MongoDB Atlas](https://www.mongodb.com/atlas) cluster (M0 free tier works; Atlas Search and Vector Search must be enabled)
- Grove API key + base URL (Azure OpenAI gateway providing GPT-5.4)
- [Voyage AI](https://www.voyageai.com) API key (for `voyage-4-large` embeddings)

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd Reorder_Alert_Agent
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/
GROVE_API_KEY=<your-grove-api-key>
GROVE_API_BASE_URL=https://grove-gateway-prod.azure-api.net/grove-foundry-prod/openai/v1
VOYAGE_API_KEY=<your-voyage-api-key>
```

### 3. Seed MongoDB

Run once to create and populate all collections (including 90 days of consumption history), generate Voyage AI embeddings for order history, and create the Atlas Search and Vector Search indexes (including the new `agent_memory_vector_index`). Also creates a TTL index on `short_term_memory` for automatic 24 h expiry.

```bash
python data/seed.py
```

> **Re-running seed:** `seed.py` drops and recreates `reorder_alerts`, `proposed_orders`, `short_term_memory`, and `agent_memory` on each run — giving you a clean demo state. The Atlas Vector Search indexes are also dropped and recreated to reflect any embedding model changes.

---

## Running the Demo

### Option A — Docker Compose (recommended)

One command starts all four services in the correct order:

```bash
docker compose up --build
```

Docker Compose will:

1. **Build** a single shared image from the `Dockerfile`
2. **Start `seeder`** — seeds MongoDB and creates Atlas indexes, then exits
3. **Start `agent`, `simulator`, and `app`** — only after the seeder completes successfully

Open **http://localhost:8501** in your browser.

To stop everything:

```bash
docker compose down
```

To re-seed and restart from a clean state:

```bash
docker compose down && docker compose up --build
```

> **Note:** The seeder creates Atlas Search and Vector Search indexes which may take 1–3 minutes to become active. The agent and dashboard will start immediately but will show 0 Atlas/Vector Search hits until the indexes are ready.

---

### Option B — Manual (multiple terminals)

```bash
# Activate the virtual environment first (each terminal)
source .venv/bin/activate

# Terminal 1 — stream simulator (generates reorder alerts every 10 s)
python simulator/stream_simulator.py

# Terminal 2 — agent (listens for alerts via Change Stream, drafts orders)
python agent/graph.py

# Terminal 3 — Streamlit dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

Within ~30 seconds you should see:
1. **Column 1** — inventory levels ticking down every 10 s as the simulator runs
2. **Column 2** — reorder alerts appearing with an "Agent processing…" spinner
3. **Column 3** — proposed purchase orders drafted by the agent; `high` confidence orders are **auto-approved** instantly

Use the **Approve / Reject** buttons for `medium` and `low` confidence orders.

As the agent processes more alerts the memory layers fill up. From the second alert onwards for a given SKU, the agent prompt includes recent decisions (short-term) and learned patterns (long-term), visibly influencing the rationale and quantity choices.

---

## Dashboard (`app.py`)

The Streamlit dashboard auto-refreshes every **10 seconds** and is split into three columns plus a sidebar.

### Column 1 — Live Inventory

- **Critical items sort to the top** — SKUs below their reorder point always appear first.
- Each card shows on-hand, on-order, and reorder-point metrics.
- A **stock level progress bar** shows on-hand as a percentage of the reorder point; items below threshold display an `⚠️ Below reorder` warning.

### Column 2 — Active Alerts

- **Urgency colour tag** on every alert: 🔴 Critical (`< 2 days`), 🟡 Low stock (`< 7 days`), 🟠 Watch (`≥ 7 days`).
- **Time-ago label** (`🕐 Xs ago`) instead of a raw timestamp — easier to read during a live demo.
- A `⏳ Agent processing…` indicator appears while the agent is working on an alert.

### Column 3 — Agent Decisions

- **Awaiting-approval orders sort to the top** so actionable items are always visible.
- **Confidence progress bar** — High = 100 %, Medium = 60 %, Low = 30 %.
- **⚡ Auto-approved** badge on orders the agent approved automatically (confidence: `high` **and** total cost `< $10,000`).
- **🔍 Atlas Search** badge when supplier capability search influenced the decision.
- Time-ago label in the card header.
- Expandable **💬 Agent Rationale** and **🧠 Similar Past Orders (Vector Search)** sections.
- **Approve / Reject** buttons for `awaiting_approval` orders.

### Reject flow

When an order is **rejected**:
1. `proposed_orders.status` is set to `rejected`.
2. The linked `reorder_alert` is **reset to `pending`** and its `order_id` is cleared.
3. A **human-decision memory entry** (`decided_by: "human"`, `human_decision: "rejected"`) is written to both `short_term_memory` and `agent_memory` (embedded) in a background thread.
4. The Change Stream fires again — the agent re-processes the same alert immediately. The rejection is visible in short-term memory with a `⚠ HUMAN REJECTED` tag; the system prompt instructs the agent to change supplier or quantity and explain what it changed.

When an order is **approved** by a human:
1. `proposed_orders.status` is set to `approved` and `inventory.on_order` is incremented.
2. A **human-decision memory entry** (`decided_by: "human"`, `human_decision: "approved"`) is written to both memory layers — a positive signal that the agent can learn from on future alerts for the same SKU.

### Sidebar

**Simulator Controls**
Start / pause / stop inventory drain plus an **⚡ Drain Speed** select slider:

| Setting | Multiplier | Use case |
|---|---|---|
| 1× Normal | 1× | Measured walkthrough |
| 2× Fast | 2× | Accelerated demo |
| 3× Faster | 3× | — |
| 5× Demo | 5× | Live presentation |
| 10× Chaos | 10× | Rapid alert surge |

**Supplier Search**
Full-text Atlas Search using a `compound` operator: `supplier_name` matches score **3× higher** than `notes` matches, so exact name hits always outrank fuzzy capability hits. SKU strings are excluded from the search path. Fuzzy matching handles typos within one edit distance. Falls back to a regex search if Atlas Search is unavailable.

**Demo Status**
Live counters: Last Alert / Last Decision, Total Decisions, Auto-Approved %, Long-term Memories.

**🔄 Reset Demo**
Clears `reorder_alerts`, `proposed_orders`, `short_term_memory`, `agent_memory`, `checkpoints`, `checkpoint_writes`, and the Change Stream resume token from `agent_state`. Resets `on_order` counters to 0 and restores simulator to 1× speed. Inventory and suppliers are preserved — no reseeding required.

### Footer metrics

| Metric | Description |
|---|---|
| SKUs Monitored | Total inventory items |
| Below Reorder Point | Count of SKUs currently in alert state |
| Pending Alerts | Open alerts awaiting agent processing |
| Awaiting Approval | Orders needing human review |
| Auto-Approved | Orders the agent approved autonomously |
| Long-term Memories | Documents in `agent_memory` (grows each run) |

---

## Project Structure

```
Reorder_Alert_Agent/
├── Dockerfile                    # single image for all services
├── docker-compose.yml            # orchestrates seeder → agent + simulator + app
├── .env                          # credentials (not committed)
├── .env.example                  # template
├── requirements.txt
├── data/
│   └── seed.py                   # one-time DB setup (run by seeder service)
├── simulator/
│   └── stream_simulator.py       # simulates Atlas Stream Processing
├── agent/
│   ├── tools.py                  # LangChain @tool functions (MongoDB reads)
│   ├── prompts.py                # LLM system + user prompt templates
│   └── graph.py                  # LangGraph state machine + Change Stream watcher
└── app.py                        # Streamlit dashboard
```

---

## SKUs in the Demo

| SKU | Name | Category | Location | Reorder Point | Consumption pattern |
|---|---|---|---|---|---|
| MED-2041 | Amoxicillin 500mg Capsules | pharmaceutical | DC-Ohio | 200 | Stable |
| MED-3017 | Insulin Glargine | pharmaceutical | DC-Texas | 1,000 | **↗ Rising** |
| MED-4490 | Metformin 1000mg | pharmaceutical | DC-Ohio | 1,500 ✅ above reorder | Stable |
| MED-5502 | Vancomycin 1g IV | pharmaceutical | DC-California | 200 | Stable |
| MED-6201 | Heparin Sodium 5000U/mL | pharmaceutical | DC-Texas | 600 | Stable |
| SURG-0084 | Nitrile Gloves (box) | surgical | DC-Ohio | 100 | **↗ Rising** |
| SURG-1122 | IV Bags 1L | surgical | DC-Texas | 500 | Stable |
| SURG-2244 | N95 Respirator Mask | surgical | DC-California | 400 | Stable |
| DIAG-0331 | Rapid COVID/Flu Combo Test Kit | diagnostic | DC-Ohio | 500 | **↗ Rising** |
| LAB-0112 | Aerobic Blood Culture Bottle | laboratory | DC-Texas | 400 | Stable |

**Notes:**
- `MED-4490` is intentionally seeded above its reorder point to show the agent correctly ignoring healthy stock.
- `MED-3017`, `SURG-0084`, and `DIAG-0331` have a **linearly rising demand trend** in their last 30 days of seeded history (consumption ramps from 1× to 2× the base average). The trend detector reliably returns `trend: increasing` for these SKUs (second-half / first-half ratio ≈ 1.15 — above the 1.10 threshold), which prevents HIGH confidence even when stock days are adequate. This demonstrates the confidence model doing real work rather than always approving.

---

## MongoDB Atlas Features Demonstrated

| Feature | Where used |
|---|---|
| **Document model** | Flexible schema across pharmaceutical, surgical, diagnostic, and laboratory SKUs |
| **Time Series Collection** | `consumption_history` — 90 days of dispensing events; 3 SKUs seeded with a rising trend |
| **Change Streams** | Agent triggers on every insert to `reorder_alerts`; resume token persisted to `agent_state` for crash recovery |
| **Aggregation Pipeline** | 14-day consumption trend, `$facet` for dashboard KPI counts, Time Series group-by-day |
| **TTL Index** | `short_term_memory.decided_at` — auto-deletes entries older than 24 h |
| **Compound Indexes** | `(sku, location, status)` on `proposed_orders`; `(status, created_at)` on `reorder_alerts`; `(sku, location, decided_at)` on `short_term_memory` |
| **Atlas Search** | `gather_context` — compound operator with 3× name boost, fuzzy notes matching; also powers sidebar Supplier Search |
| **Atlas Vector Search** | `gather_context` — category pre-filtered search on `order_history`; location-filterable search on `agent_memory` |
| **LangGraph MongoDBSaver** | Checkpoints full graph state to `checkpoints` collection per alert invocation |

### Atlas Search — `suppliers_text_search` index

Created automatically by `seed.py` on the `suppliers` collection.
Uses a `compound` operator with boosted scoring: `supplier_name` matches score **3× higher** than `notes` matches, so name hits always outrank fuzzy capability hits. The `sku` field is intentionally excluded from the search path to prevent SKU strings from polluting capability searches. Fuzzy matching (`maxEdits: 1`) handles typos.
Query is built from the **specific product name + urgency phrase** (e.g. `"Vancomycin 1g IV urgent emergency expedited"`) so only genuinely relevant suppliers score highly.
Hit count varies (0–5) depending on how many suppliers' notes match. Results include a `search_score` stored on each proposed order.

### Atlas Vector Search — `order_history_vector_index` index

Created automatically by `seed.py` on the `order_history` collection.
Each historical order's `rationale` text is embedded with **Voyage AI `voyage-4-large`** (1024 dims, cosine similarity) at seed time.
The index includes a **`category` pre-filter field**, allowing the `$vectorSearch` stage to scope ANN search to same-category orders before scoring — eliminating cross-category noise (e.g. surgical order precedents won't influence pharmaceutical decisions).
At alert time the agent embeds the current situation and retrieves past orders with `similarity_score ≥ 0.78`.
Result count varies (0–5): critical alerts (`<2 days` stock) fetch up to 5 candidates; routine alerts up to 3. Similar orders — including outcomes — are passed to GPT-5.4 and stored on the proposed order.

### Agent Memory — short-term and long-term

The agent maintains two memory layers that grow as it processes alerts. Both layers record **agent decisions and human approve/reject overrides**, making recommendations progressively more informed by both machine and human signal.

**Short-term memory (`short_term_memory` collection)**
After every `save_and_notify` and every human approve/reject, a concise decision record is written for the SKU and location. Each record includes `decided_by` (`"agent"` or `"human"`) and `human_decision` (`"approved"` / `"rejected"`) fields. On the next alert for the same SKU, `gather_context` reads these records (last 24 h, up to 5) and renders them in the LLM prompt with visual tags:
- `⚠ HUMAN REJECTED` — agent **must** change supplier or quantity and explain what changed
- `✓ HUMAN APPROVED` — strong signal to reuse that supplier if conditions are still comparable
- `AUTO-APPROVED` — agent's own past decisions already in flight

A **TTL index** on `decided_at` (86 400 s) automatically purges entries older than 24 h.
A **compound index** on `(sku, location, decided_at)` covers the per-SKU time-range query efficiently.

**Long-term memory (`agent_memory` collection)**
After every `save_and_notify` and every human approve/reject, a natural-language summary is embedded with **Voyage AI `voyage-4-large`** and stored in `agent_memory`. Human override summaries are prefixed with `"Human APPROVED/REJECTED order for…"` so Vector Search retrieval surfaces them as distinct precedents.
On each new alert, `gather_context` runs a **`$vectorSearch`** against `agent_memory` (index: `agent_memory_vector_index`, pre-filter on `location`, score ≥ 0.72) and injects the most relevant past learnings into the prompt. The collection starts empty and becomes richer with every run and every human decision.

### LangGraph `MongoDBSaver` checkpointer

Every graph invocation (one per alert) is checkpointed to the `checkpoints` collection in `supply_chain_demo`.
Each alert's MongoDB `_id` is used as the `thread_id`, so snapshots are fully isolated and inspectable in Atlas.
This enables crash recovery (the graph can resume mid-run) and provides a full audit trail of every state transition the agent went through.

---

> **Note on Atlas Stream Processing:** In production, `stream_simulator.py` would be replaced by an Atlas Stream Processing pipeline consuming from Apache Kafka. The ASP pipeline performs windowed aggregation and reorder-point checks before writing to `reorder_alerts`. The agent code is identical regardless — it only sees the `reorder_alerts` collection.
