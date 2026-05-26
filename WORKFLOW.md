# Agent Workflow Diagrams

---

## Diagram 1 — System Workflow

End-to-end flow from inventory event to resolved order, including trigger layer, agent pipeline, memory reads/writes, and persistence.

```mermaid
flowchart TD
    %% ── Trigger Layer ────────────────────────────────────────────────────────
    subgraph TRIGGER["⚡ Trigger Layer"]
        SIM["🔄 Simulator\nconsumption event every 10 s\n1 min = 1 demo day"]
        KAFKA["📨 Kafka Consumer\nwms-inventory-events topic"]
        ALERTS[("reorder_alerts\ncollection")]
        CS["📡 MongoDB Change Stream\nwatches reorder_alerts\nfor status: pending"]
    end

    SIM -->|"on_hand drops below\nreorder_point"| ALERTS
    KAFKA -->|"on_hand drops below\nreorder_point"| ALERTS
    ALERTS --> CS
    CS -->|"alert doc injected\ninto AgentState"| START

    %% ── Agent Pipeline ───────────────────────────────────────────────────────
    START(["▶ START"]) --> N1

    subgraph PIPELINE["🤖 LangGraph Agent Pipeline  ·  MongoDB Checkpointer"]
        N1["1 · assess_alert\nFetch live inventory · compute coverage gap\ncheck active in-flight orders"]

        N1 -->|"coverage_gap == 0\nzero-gap fast path"| N6
        N1 -->|"coverage_gap > 0"| N2

        N2["2 · retrieval_agent\nReAct tool-calling loop\nLLM decides which tools to call\nmax 20 iterations"]

        N2 --> N3

        N3["3 · analysis_agent\nEvaluate suppliers\nAssign confidence · flag risks\nNo quantity output"]

        N3 --> N4

        N4["4 · recommendation_agent\nCalculate order quantity\nWrite plain-English rationale\nReceives prior audit errors on retry"]

        N4 --> N5

        N5{"5 · audit_agent\nPydantic schema validation\nbusiness rule checks"}

        N5 -->|"valid"| N6
        N5 -->|"invalid\nretries < 2"| N4
        N5 -->|"invalid\nretries ≥ 2"| N7

        N6["6 · save_and_notify\nPersist proposed order\nWrite memory layers\nAuto-approve or queue for human"]

        N7["7 · escalate\nDead-letter alert\nFire optional webhook\nStatus → escalated"]
    end

    N6 --> END1(["⏹ END"])
    N7 --> END2(["⏹ END"])

    %% ── Memory Layer (reads) ─────────────────────────────────────────────────
    subgraph MEMORY["🧠 Memory Layer  ·  MongoDB Atlas"]
        ST[("short_term_memory\n24 h rolling window")]
        LT[("agent_memory\nAtlas Vector Search\nVoyage AI embeddings")]
        EP[("alert_lifecycle\nepisodic store")]
        PR[("procedures\nhuman-confirmed rules")]
    end

    N2 -->|reads| ST
    N2 -->|reads| LT
    N2 -->|reads| EP
    N2 -->|reads| PR

    N6 -->|writes| ST
    N6 -->|writes| LT

    %% ── Persistence Layer (writes) ───────────────────────────────────────────
    subgraph PERSIST["💾 Persistence Layer  ·  MongoDB"]
        PO[("proposed_orders\nstatus: awaiting_approval\nor auto-approved")]
        OH[("order_history\n+ Voyage AI embedding\nfor future Vector Search")]
        CO[("confidence_outcomes\ncalibration tracking")]
        LC[("alert_lifecycle\nfull episode log")]
        EQ[("escalation_queue\nhuman review required")]
    end

    N6 -->|writes| PO
    N6 -->|writes| OH
    N6 -->|writes| CO
    N6 -->|writes| LC
    N7 -->|writes| EQ

    %% ── Delivery Loop ────────────────────────────────────────────────────────
    subgraph DELIVERY["📦 Delivery Simulation  ·  Simulator tick every 10 s"]
        DL["deliver_pending_orders\nChecks approved orders\n95% on-time · 5% delayed 1–3 days\n1 real min = 1 demo day"]
    end

    PO -->|"status: approved\ndelivered_at missing"| DL
    DL -->|"status → received\non_hand += qty\non_order -= qty"| ALERTS
    DL -->|"outcome update"| OH
    DL -->|"outcome → resolved"| CO
    DL -->|"order_delivered event"| LC

    %% ── Styles ───────────────────────────────────────────────────────────────
    style START fill:#1a1a2e,color:#e2e8f0,stroke:#4a5568
    style END1  fill:#1a1a2e,color:#e2e8f0,stroke:#4a5568
    style END2  fill:#1a1a2e,color:#e2e8f0,stroke:#4a5568
    style N1    fill:#2d3748,color:#e2e8f0,stroke:#4a90d9
    style N2    fill:#1a365d,color:#e2e8f0,stroke:#4a90d9
    style N3    fill:#1a365d,color:#e2e8f0,stroke:#4a90d9
    style N4    fill:#1a365d,color:#e2e8f0,stroke:#4a90d9
    style N5    fill:#744210,color:#fefcbf,stroke:#d69e2e
    style N6    fill:#1c4532,color:#c6f6d5,stroke:#38a169
    style N7    fill:#742a2a,color:#fed7d7,stroke:#e53e3e
    style DL    fill:#2c3e50,color:#e2e8f0,stroke:#319795
```

---

## Diagram 2 — Agent Cards

Detailed view of each agent node: responsibility, LLM usage, state inputs and outputs, tools, and routing behaviour.

```mermaid
classDiagram
    direction TB

    class assess_alert {
        ROLE: Compute coverage gap · no LLM
        ─────────────────────────────────────
        LLM: none · pure DB queries
        ─────────────────────────────────────
        INPUT state fields
        alert: reorder alert doc
        ─────────────────────────────────────
        READS from MongoDB
        inventory: live on_hand · on_order
        proposed_orders: active order qty
        ─────────────────────────────────────
        OUTPUT state fields
        inventory: enriched inventory doc
        existing_order_qty: units already ordered
        coverage_gap: units still needed
        expedite: true if days_remaining < 2
        ─────────────────────────────────────
        ROUTING
        gap == 0 → save_and_notify fast path
        gap  > 0 → retrieval_agent
    }

    class retrieval_agent {
        ROLE: ReAct tool loop · gather context
        ─────────────────────────────────────
        LLM: GPT-5.4 · max 20 iterations
        ─────────────────────────────────────
        INPUT state fields
        alert · coverage_gap · expedite
        ─────────────────────────────────────
        TOOLS available
        get_inventory_position
        get_supplier_options
        get_consumption_trend
        search_suppliers_by_capability [Atlas Search]
        find_similar_past_orders [Vector Search]
        get_recent_decisions [short-term memory]
        get_learned_patterns [Vector Search]
        ─────────────────────────────────────
        OUTPUT state fields
        suppliers · consumption · similar_orders
        short_term_memories · long_term_memories
        supplier_search_results · retrieval_trace
    }

    class analysis_agent {
        ROLE: Supplier evaluation · confidence score
        ─────────────────────────────────────
        LLM: GPT-5.4 · up to 3 JSON retries
        ─────────────────────────────────────
        INPUT state fields
        suppliers · consumption
        short_term_memories · long_term_memories
        similar_orders · supplier_search_results
        ─────────────────────────────────────
        CONSTRAINT: no quantity calculation
        ─────────────────────────────────────
        OUTPUT state fields
        analysis.best_supplier_id
        analysis.best_supplier_name
        analysis.confidence: high · medium · low
        analysis.risk_flags: list of strings
        analysis.reasoning_trace
    }

    class recommendation_agent {
        ROLE: Order quantity · plain-English rationale
        ─────────────────────────────────────
        LLM: GPT-5.4 · up to 3 JSON retries
        ─────────────────────────────────────
        INPUT state fields
        analysis · coverage_gap · consumption
        suppliers · short and long-term memories
        audit_result.errors on retry
        ─────────────────────────────────────
        QUANTITY FORMULA
        gap + safety_stock - on_order
        rounded up to supplier MOQ
        ─────────────────────────────────────
        OUTPUT state fields
        recommendation.supplier_id
        recommendation.supplier_name
        recommendation.quantity
        recommendation.rationale
        recommendation.confidence
    }

    class audit_agent {
        ROLE: Schema + business rule gate
        ─────────────────────────────────────
        LLM: none · Pydantic validation
        ─────────────────────────────────────
        INPUT state fields
        recommendation · alert · coverage_gap
        ─────────────────────────────────────
        VALIDATES
        required fields present and typed
        quantity > 0
        confidence in high · medium · low
        supplier_id non-empty
        ─────────────────────────────────────
        OUTPUT state fields
        audit_result.valid: bool
        audit_result.errors: list of strings
        audit_retries: incremented on failure
        ─────────────────────────────────────
        ROUTING
        valid → save_and_notify
        invalid · retries < 2 → recommendation_agent
        invalid · retries ≥ 2 → escalate
    }

    class save_and_notify {
        ROLE: Persist order · write all memory
        ─────────────────────────────────────
        LLM: none · async DB writes
        ─────────────────────────────────────
        INPUT state fields
        alert · recommendation · analysis
        inventory · consumption · similar_orders
        ─────────────────────────────────────
        AUTO-APPROVE condition
        confidence == high AND cost < $2500
        ─────────────────────────────────────
        WRITES to MongoDB
        proposed_orders: full order document
        inventory.on_order: incremented if auto-approved
        reorder_alerts: status → processed
        alert_lifecycle: agent_decision · order_placed
        short_term_memory: rolling 24 h window
        agent_memory: vectorised rationale
        order_history: rationale + Voyage AI embedding
        confidence_outcomes: calibration record
        ─────────────────────────────────────
        OUTPUT state fields
        order_id: _id of saved proposed_order
    }

    class escalate {
        ROLE: Dead-letter · human escalation
        ─────────────────────────────────────
        LLM: none
        ─────────────────────────────────────
        INPUT state fields
        alert · recommendation · audit_result
        ─────────────────────────────────────
        TRIGGERS when
        audit_retries ≥ 2 after recommendation
        rejection_count ≥ 3 from human reviewer
        LLM circuit breaker open
        ─────────────────────────────────────
        WRITES to MongoDB
        escalation_queue: full alert context
        reorder_alerts: status → escalated
        alert_lifecycle: escalated event
        ─────────────────────────────────────
        FIRES optional webhook
        ESCALATION_WEBHOOK_URL env var
    }

    assess_alert         --> retrieval_agent      : coverage_gap > 0
    assess_alert         --> save_and_notify      : coverage_gap == 0
    retrieval_agent      --> analysis_agent
    analysis_agent       --> recommendation_agent
    recommendation_agent --> audit_agent
    audit_agent          --> save_and_notify      : valid
    audit_agent          --> recommendation_agent : retry
    audit_agent          --> escalate             : max retries
```

---

## State Schema Reference

Fields carried through `AgentState` across all nodes.

```mermaid
classDiagram
    class AgentState {
        ALERT CONTEXT
        alert: dict
        inventory: dict
        ─────────────────────────────────────
        RETRIEVAL RESULTS
        suppliers: list
        consumption: dict
        supplier_search_results: list
        similar_orders: list
        short_term_memories: list
        long_term_memories: list
        retrieval_results: dict
        retrieval_trace: list
        ─────────────────────────────────────
        ROUTING FLAGS
        existing_order_qty: int
        coverage_gap: int
        expedite: bool
        ─────────────────────────────────────
        AGENT OUTPUTS
        analysis: dict
        recommendation: dict
        audit_result: dict
        audit_retries: int
        ─────────────────────────────────────
        PERSISTENCE
        order_id: str
    }
```
