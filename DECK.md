# Agentic Workflows in Practice: Giving AI the Memory to Solve Real Problems

> **Format:** Google Slides script — each section contains: Headline, Body, Speaker Script
>
> **Runtime:** 22–28 min with live demo · 16–20 min without

---

## Slide 1 — Title

**Headline:** Agentic Workflows in Practice

**Subhead:** Giving AI the Memory to Solve Real Problems

**Body:**
- A live demonstration: reactive multi-agent supply chain reordering
- Built on LangGraph · MongoDB Atlas · Voyage AI · GPT-5.4

**Speaker script:**
> "The subtitle is a promise I intend to keep in the next 25 minutes. Not 'giving AI memory' as a metaphor — literally: a persistent memory architecture that sits behind a multi-agent system and allows it to reason from accumulated experience rather than starting from zero every time. By the end of this talk you will have seen it working on a real problem, understood why it works architecturally, and have a framework for applying the same pattern to your own workflows. Let's start with the problem."

---

## Slide 2 — Every Supply Chain Has a 2am Moment

**Headline:** Every Supply Chain Has a 2am Moment

**Body:**
- A critical SKU drops below its reorder threshold overnight
- The analyst on call opens four dashboards — none of them share context
- She spends 90 minutes reconstructing what the system should already know
- She makes a decision. She documents it. Next month, it happens again. The system has forgotten.
- This is not a staffing problem. It is not a tooling problem.
- **It is a memory problem.**

**Three signals that your agents have the same problem:**
- They re-derive context across sessions that they computed correctly last time
- They ignore human overrides made in a previous run
- Two agents working the same workflow reach different conclusions because neither knows what the other learned

**Speaker script:**
> "I'm not going to start by explaining what an AI agent is. You're building with them. What I want to establish first is that the problem I'm going to solve today is one you recognise — either from your own systems or from your customers'. Healthcare supply chains experience critical stockouts that take an average of six hours to resolve manually. Not because the people handling them are slow, but because the information required to make a good decision is scattered, disconnected, and — this is the part that matters — it has to be reconstructed from scratch every single time. If that sounds familiar in a different domain, it's because it's the same architectural failure. Stateful problems running on stateless infrastructure. Let me show you what that costs."

---

## Slide 3 — The Cost Is Measurable. The People Paying It Have Names.

**Headline:** The Cost Is Measurable. The People Paying It Have Names.

**Body — four statistics (large type):**

| | |
|---|---|
| **6 hours** | average time to resolve a single supply chain disruption manually today |
| **40–80%** | failure rate in multi-agent systems operating under production load *(Stanford HAI, 2024)* |
| **41.77%** | of those failures traced directly to context and specification loss *(AgentBench analysis)* |
| **50 tool calls** | per complex task at real token costs — wasted when the agent re-derives what it already knew |

**Two human stories:**

> **The analyst, 2:17am**
>
> She's paged because Vancomycin 1g IV — a critical antibiotic — has dropped to 42% of its reorder threshold at the California distribution centre. She opens the inventory dashboard, the supplier portal, the consumption history tool, and the order history spreadsheet. None of them know what the others know. She spends 90 minutes reading four disconnected systems to answer one question: *what should we order, from whom, and how fast?* The answer existed in those systems. The system just couldn't hold it together.

> **The nurse, 7:04am**
>
> She picks up the chart for the patient in bed 4. He was admitted 18 hours ago. She spends 20 minutes asking colleagues, reviewing notes, and reconstructing a clinical picture that a connected system already held — but couldn't surface in a useful form. The information existed. The memory didn't.

**Speaker script:**
> "Two faces on the numbers. The analyst at 2am isn't making bad decisions because she's undertrained or under-resourced. She's spending 90 minutes doing cognitive labour that a system with memory should be doing for her. And she'll do it again next month, for the same SKU, at the same location, because the resolution she reached this month will not be available to the system that wakes up next month. The nurse at 7am is in the same position. Not missing information — missing a system capable of holding information across time and presenting what's relevant without being asked. That is the problem memory solves. Now let me show you why today's agents don't solve it on their own."

---

## Slide 4 — Four Premises. One Conclusion.

**Headline:** Four Premises You Already Accept. One Conclusion You Can't Avoid.

**Body:**

**Premise 1**
LLMs are stateless by design. Each invocation begins with no knowledge of any previous invocation. This is not a limitation to be patched — it is the architecture.

**Premise 2**
Production workflows span sessions, shifts, and days. A supply chain disruption that begins Monday night is resolved Tuesday morning. A clinical episode that opens at admission closes at discharge. Statefulness is not optional.

**Premise 3**
Agents without shared persistent memory fail systematically when the workflow exceeds a single context window — they re-derive, contradict, and ignore prior decisions because those decisions do not exist in their context.

**Premise 4**
Context engineering — deciding what information the agent receives, from where, in what form, at what moment — is now the highest-leverage skill in production agentic systems. More than prompt engineering. More than model selection.

---

> ### ∴ Conclusion
>
> **Giving an agent memory is not a feature you add when you have time.**
> **It is the engineering decision that determines whether your agent**
> **is solving real problems or performing them.**

---

**Speaker script:**
> "Here is the logical case, as compressed as I can make it. LLMs are stateless — that's the architecture, not a bug. Production workflows are stateful — that's reality, not a choice. Close that gap without an explicit memory layer and you get systematic failure: agents that re-derive what they already knew, ignore what humans told them last week, and contradict each other because they're reasoning from different incomplete pictures. Context engineering — what you give the agent, when, from where — is the skill that determines whether the agent actually solves the problem. Those four premises have one conclusion: giving an agent memory is not optional. It is the engineering decision that separates systems that perform agentic workflows from systems that solve them."

---

## Slide 5 — Perceive → Remember → Plan → Act → Learn

**Headline:** Perceive → Remember → Plan → Act → Learn

**Body:**

**The loop:**

```
PERCEIVE   Change Stream fires on stock drop
    ↓         (no polling; synchronous trigger)
REMEMBER   Short-term + vector + episodic memory loaded
    ↓         (what we knew before this call begins)
PLAN       ReAct agent selects tools; Analysis agent evaluates
    ↓         (LLM reasons from a full picture, not a blank slate)
ACT        Order written or escalated; inventory adjusted
    ↓         (decision executed with explainable rationale)
LEARN      Outcome recorded; memory updated; rules extracted
    ↓         (next invocation starts smarter than this one)
    ↻ loops
```

**The memory stack:**

| Layer | What it holds | Technology |
|---|---|---|
| **Short-term** | Last 24h decisions for this SKU + location | MongoDB, rolling window |
| **Long-term** | Semantic order history across all SKUs | Atlas Vector Search · Voyage AI |
| **Episodic** | Full alert lifecycle: alert → decision → delivery | `alert_lifecycle` collection |
| **Procedural** | Confirmed rules: *"For surgical SKUs in DC-Ohio, prefer SUP-005"* | `procedures` collection |
| **Checkpoint** | Full LangGraph state, resumable across restarts | MongoDB Checkpointer |

**Speaker script:**
> "Here is the concrete architecture. The loop has five phases — and the one that makes everything else work is Learn. Every other phase exists so that Learn can improve it. Perceive fires the agent without polling — a MongoDB Change Stream delivers the alert the moment it's written. Remember loads what the agent already knows before a single LLM call is made: the last 24 hours of decisions for this SKU, the semantically similar past orders from vector search, any procedural rule an operator has confirmed for this category. Plan and Act are where the LLM operates — but it's operating from a full picture, not a blank slate. And then Learn writes the outcome back: the human's decision, the delivery result, the confidence accuracy. The next event starts with all of that in context. The system compounds. A stateless system resets."

---

## Slide 6 — Live Demo: Watch It Solve the Problem

**Headline:** The Same Problem. Solved in Under 10 Seconds.

**Presenter checklist** *(not shown to audience):*

1. Open dashboard → KPI row: 3 of 10 SKUs below reorder (30% — the realistic starting state)
2. Point to simulator status: "This is the Change Stream in action — consumption events every 10 seconds"
3. Alert fires for MED-5502 (Vancomycin) → agent pipeline activates
4. Proposed order card appears — open rationale expander: show vector search surfacing past orders
5. Point to memory: "This is episodic memory. The agent knows what happened last time."
6. If awaiting approval — approve manually; show `on_order` increment in the inventory grid
7. Wait ~3 min → 📦 RECEIVED badge appears → inventory grid updates
8. Open Confidence Calibration expander → show % resolved per confidence tier
9. Admin panel → Extract Rules → point to procedural memory formation

**Speaker script:**
> "Let me switch to the live system. Real MongoDB Atlas cluster. Three SKUs already below their reorder threshold — about 30% of the catalogue. The simulator is generating consumption events every ten seconds, which in this demo represents one day of inventory movement. Watch the Active Alerts panel."
>
> *(narrate as events happen)*
>
> "An alert fired for Vancomycin 1g IV at the California distribution centre — the same SKU from our 2am story. The agent woke up. Open the rationale. It has retrieved two semantically similar past orders — same category, comparable urgency — and is using their outcomes to weight its supplier selection. That is episodic memory surfaced through vector search. The agent didn't query for it. It knew to look. The proposed order is high confidence, under the auto-approval threshold — the analyst is never paged. In about three minutes of demo time — three days of simulated time — you'll see the 📦 RECEIVED badge and the inventory grid update. The loop is closed. The outcome will be written to memory before the next alert fires."

---

## Slide 7 — Same Problem. Different Experience of It.

**Headline:** Same Problem. Different Experience of It.

**The analyst:**

| Before | After |
|---|---|
| Paged at 2:17am | Not paged |
| 90 minutes reconstructing context | 0 minutes — agent held it |
| Decision made on incomplete information | Decision made from 90 days of history |
| Next month: starts from zero again | Next month: agent has this episode in memory |
| **Resolution time: ~6 hours** | **Resolution time: under 10 seconds** |

**The doctor:**

| Before | After |
|---|---|
| 20 minutes reconstructing patient episode | Walks in already knowing |
| Asks colleagues what the night shift learned | Agent held the night shift's observations |
| Has to search for comparable cases | Agent surfaced one without being asked |
| — | *"Similar tachycardia on day 2 resolved with IV fluids in 3 comparable cases. Confidence: medium."* |

**The number that matters:**

> Not the 10 seconds. The trajectory.
> Every resolved event makes the next one faster.
> A system without memory resets. A system with memory compounds.

**Speaker script:**
> "Let me put the before and after side by side. The analyst is not paged. The context she would have spent 90 minutes reconstructing was held by the system. The decision was made from 90 days of order history and three directly comparable past episodes. And here's the detail that doesn't show up in a benchmark: next month, when this happens again, the agent will have this episode in its memory. It won't start from zero. It will start from an informed baseline. The same compounding applies to the clinical case. The doctor walks into rounds already knowing. She doesn't ask colleagues what the night shift observed — the agent held those observations. And it surfaced a comparable case without being asked, because it recognised the pattern from episodic memory. The number I want you to hold onto is not the 10 seconds. It's the trajectory. A stateless system resets every time. A system with memory compounds every time."

---

## Slide 8 — A 90-Day Path From Stateless to Memory-Enabled

**Headline:** A 90-Day Path From Stateless to Memory-Enabled

**Body:**

### Days 1–30 — Diagnose: find your memory failure

> Where in your current agentic workflows does context loss cause the most damage?
>
> Map three things: where agents re-derive context they've already computed; where human overrides are ignored in the next session; where two agents working the same problem reach different conclusions.
>
> *That intersection is where you build first.*

### Days 31–60 — Design: match the memory layer to the failure mode

> - Re-derivation of recent decisions? A MongoDB checkpointer + short-term memory collection is enough.
> - Ignoring human overrides? Write approvals and rejections back as tagged memory entries.
> - Agents contradicting across sessions? Add episodic memory: a per-workflow lifecycle collection.
> - Recurring patterns you want the agent to apply automatically? Build procedural memory with human confirmation.
>
> *Don't over-engineer. Match the layer to the problem.*

### Days 61–90 — Deploy: instrument for five signals

> 1. Human override rate is falling as the agent learns from past overrides
> 2. Auto-approval rate is rising without quality degradation
> 3. Time-to-decision is decreasing on scenarios the agent has seen before
> 4. Agents are no longer being paged for situations they have resolved before
> 5. A new engineer can read the rationale and understand why the agent decided what it decided

---

### Closing question

> *If your agents forgot everything they learned today —*
> *every decision, every human override, every outcome, every pattern —*
> *would your workflows still solve real problems tomorrow?*
>
> *If yes: you've built capable tools.*
> *If no: you know exactly what to build next.*

**Speaker script:**
> "The 90-day framework is a diagnostic-first approach, not a technology-first one. Start by finding where memory failure is the root cause in your specific workflows — not where you think agents might be useful, but where you can trace a real failure back to context loss. Then design the memory layer that matches that failure mode exactly. Not every workflow needs vector search and procedural memory. Some need a checkpointer and a short-term collection. Know which problem you have before you build the solution. Then deploy and watch the five signals. If override rates are falling and auto-approval rates are rising, the memory layer is working. If they're not moving, something in the architecture is wrong. And I'll close with the question that drove the design of everything you saw today. If your agents forgot everything they learned right now — would your workflows still solve real problems tomorrow? If yes, you've built capable tools. If no, you now know exactly what to build next."

---

## Reference

| Property | Value |
|---|---|
| Slide count | 8 |
| Runtime with demo | 22–28 min |
| Runtime without demo | 16–20 min |
| Narrative arc | Problem → human cost → logical case → architecture → proof → human outcome → your action |
| Through-line question | *If your agents forgot everything they learned today, would your workflows still solve real problems tomorrow?* |
| Demo slot | Slide 6 — budget 5–8 minutes |
