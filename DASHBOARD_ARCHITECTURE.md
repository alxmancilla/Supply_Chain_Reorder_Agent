# Dashboard Architecture — Operating Model

High-level data flow across the simulator, MongoDB Atlas, LangGraph agent pipeline, and Streamlit dashboard.

```mermaid
flowchart LR

  subgraph SEED ["seed.py - Data Setup"]
    S1["10 SKUs - 20 suppliers\n90-day consumption history\nHistorical order archive"]
  end

  subgraph DB ["MongoDB Atlas - Central Data Store"]
    direction TB
    D1[("inventory - reorder_alerts\nproposed_orders")]
    D2[("suppliers\nAtlas Search + rankFusion index")]
    D3[("order_history - agent_memory\nVector Search - Persistent")]
    D4[("consumption_history - short_term_memory\nTime Series - TTL 24h")]
  end

  SIM["Stream Simulator\nSimulates SKU consumption\nevery 5 seconds\nCreates reorder alerts\nDelivers approved orders\nafter lead time"]

  subgraph AGENT ["LangGraph Agent - Change Stream listener"]
    direction TB
    A1["assess_alert\ncoverage gap"]
    A2["retrieval_agent ReAct\nrankFusion + Vector Search\nconsumption trend + memory"]
    A3["analysis_agent\nsupplier selection + confidence\nFDA regulatory enforcement"]
    A4["recommendation_agent\norder quantity + rationale"]
    A5["audit_agent retry x2\nsave_and_notify\norder + memory writes"]
    A1 --> A2 --> A3 --> A4 --> A5
  end

  DASH["Streamlit Dashboard\nLive KPIs - inventory grid\nROP health indicators\nAlert and order feed\nHuman approve / reject\nSimulator controls\nrankFusion supplier search\nConfidence Calibration\nAgent Recovery Log"]

  SEED   -->|"seeds collections"| DB
  SIM    -->|"update on_hand\ninsert time-series"| DB
  DB     -->|"Change Stream insert"| A1
  A2    <-->|"search and recall"| DB
  A5     -->|"proposed order\nmemory writes"| DB
  DB     -->|"auto-refresh 5s"| DASH
  DASH   -->|"approve / reject"| DB
  DB     -->|"deliver orders"| SIM
```

For full node-by-node detail see [WORKFLOW.md](WORKFLOW.md).
