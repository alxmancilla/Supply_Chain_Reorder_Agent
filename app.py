"""
Streamlit dashboard — WMS-style operations view, auto-refreshes every 5 seconds.

Layout:
  Top bar  — brand, nav tabs, live indicator
  KPI row  — 4 summary cards (below-reorder, alerts, auto-approved, awaiting)
  Left panel  — inventory grid (SKU cards with stock-level bars)
  Right panel — active alerts + agent decisions (severity-badged cards)
"""

import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
from bson import ObjectId
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
from pymongo import MongoClient

from agent.tools import append_lifecycle_event, write_human_decision_memory

load_dotenv()

# ---------------------------------------------------------------------------
# Authentication — optional; activated when auth/users.yaml exists.
# Without the file the dashboard runs unauthenticated (demo / local mode).
# ---------------------------------------------------------------------------

_AUTH_CONFIG_PATH = Path(__file__).parent / "auth" / "users.yaml"
_auth_enabled = _AUTH_CONFIG_PATH.exists()
_current_role  = "admin"   # default when auth is disabled

if _auth_enabled:
    try:
        import yaml
        import streamlit_authenticator as stauth

        with open(_AUTH_CONFIG_PATH) as f:
            _auth_config = yaml.safe_load(f)

        _authenticator = stauth.Authenticate(
            _auth_config["credentials"],
            _auth_config["cookie"]["name"],
            _auth_config["cookie"]["key"],
            _auth_config["cookie"]["expiry_days"],
        )
    except Exception as _auth_exc:
        _auth_enabled = False
        _current_role  = "admin"
        import warnings
        warnings.warn(f"Auth config load failed — running unauthenticated: {_auth_exc}")


def _check_role(required_role: str) -> bool:
    """Return True if the current user has at least the required access level."""
    _hierarchy = {"viewer": 0, "approver": 1, "admin": 2}
    return _hierarchy.get(_current_role, 0) >= _hierarchy.get(required_role, 0)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Reorder Alert Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# MongoDB connection
# ---------------------------------------------------------------------------
@st.cache_resource
def get_db():
    c = MongoClient(
        os.environ["MONGODB_URI"],
        serverSelectionTimeoutMS=30_000,
        connectTimeoutMS=10_000,
        socketTimeoutMS=30_000,
    )
    return c["supply_chain_demo"]


db = get_db()

# ---------------------------------------------------------------------------
# Persist human decisions across reruns (autorefresh race-condition fix)
# ---------------------------------------------------------------------------
# st.button() returns True only in the single rerun triggered by the click.
# st_autorefresh fires every 5 s and can preempt that rerun, discarding the
# click.  We store the action in session_state so it survives any rerun.
_pending_procedure = st.session_state.pop("_pending_procedure_action", None)
if _pending_procedure:
    _pp_id     = ObjectId(_pending_procedure["rule_id_str"])
    _pp_action = _pending_procedure["action"]
    if _pp_action == "confirm":
        db.procedures.update_one(
            {"_id": _pp_id},
            {"$set": {"human_confirmed": True, "last_updated": datetime.now(timezone.utc)}},
        )
        st.toast("✅ Rule confirmed — agent will now use this supplier preference.")
    elif _pp_action == "dismiss":
        db.procedures.delete_one({"_id": _pp_id})
        st.toast("🗑 Rule dismissed.")

_pending_decision = st.session_state.pop("_pending_human_decision", None)
if _pending_decision:
    _pd_oid        = ObjectId(_pending_decision["oid_str"])
    _pd_action     = _pending_decision["action"]
    _pd_alert_str  = _pending_decision.get("alert_id_str", "")
    _pd_alert_id   = ObjectId(_pd_alert_str) if _pd_alert_str else None
    _pd_order      = db.proposed_orders.find_one({"_id": _pd_oid})
    if _pd_order and _pd_order.get("status") == "awaiting_approval":
        if _pd_action == "approve":
            db.proposed_orders.update_one({"_id": _pd_oid}, {"$set": {"status": "approved"}})
            db.inventory.update_one(
                {"sku": _pd_order["sku"], "location": _pd_order["location"]},
                {"$inc": {"on_order": _pd_order["quantity_recommended"]}},
            )
            db.confidence_outcomes.update_one(
                {"order_id": _pd_oid},
                {"$set": {"human_decision": "approved", "outcome": "resolved"}},
            )
            db.order_history.update_one(
                {"proposed_order_id": _pd_oid},
                {"$set": {"outcome": "human_approved"}},
            )
            threading.Thread(
                target=write_human_decision_memory,
                args=(dict(_pd_order), "approved"),
                daemon=True,
            ).start()
            threading.Thread(
                target=append_lifecycle_event,
                args=(
                    str(_pd_order.get("alert_id", "")), "human_approved",
                    {"order_id": str(_pd_oid)},
                    _pd_order.get("sku", ""), _pd_order.get("location", ""),
                ),
                daemon=True,
            ).start()
            st.toast(
                f"✅ Approved — {_pd_order.get('quantity_recommended', 0):,} units"
                f" from {_pd_order.get('supplier_name', 'supplier')}"
                f" ({_pd_order.get('sku', '')} @ {_pd_order.get('location', '')})"
            )
        elif _pd_action == "reject":
            db.proposed_orders.update_one({"_id": _pd_oid}, {"$set": {"status": "rejected"}})
            db.confidence_outcomes.update_one(
                {"order_id": _pd_oid},
                {"$set": {"human_decision": "rejected", "outcome": "escalated"}},
            )
            db.order_history.update_one(
                {"proposed_order_id": _pd_oid},
                {"$set": {"outcome": "rejected"}},
            )
            if _pd_alert_id:
                db.reorder_alerts.update_one(
                    {"_id": _pd_alert_id},
                    {
                        "$set":   {"status": "pending"},
                        "$unset": {"order_id": ""},
                        "$inc":   {"rejection_count": 1},
                    },
                )
            threading.Thread(
                target=write_human_decision_memory,
                args=(dict(_pd_order), "rejected"),
                daemon=True,
            ).start()
            threading.Thread(
                target=append_lifecycle_event,
                args=(
                    str(_pd_order.get("alert_id", "")), "human_rejected",
                    {"order_id": str(_pd_oid)},
                    _pd_order.get("sku", ""), _pd_order.get("location", ""),
                ),
                daemon=True,
            ).start()
            st.toast(
                f"❌ Rejected — {_pd_order.get('sku', '')} @ {_pd_order.get('location', '')}"
                " reset for reprocessing."
            )

# ---------------------------------------------------------------------------
# Auto-refresh interval
# ---------------------------------------------------------------------------
REFRESH_INTERVAL = 10  # seconds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ago(dt) -> str:
    """Return a human-readable 'Xs/Xm/Xh ago' string for a UTC datetime."""
    if not dt:
        return "?"
    try:
        d    = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        secs = max(0, int((datetime.now(timezone.utc) - d).total_seconds()))
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m ago"
        return f"{secs // 3600}h ago"
    except Exception:
        return "?"


def _delivery_timing(order: dict) -> str:
    """Return a delivery status string for the order card (1 min = 1 demo day)."""
    status = order.get("status")
    try:
        if status == "received":
            dt = order.get("delivered_at")
            label = "on time" if order.get("on_time", True) else "delayed"
            return f"📬 Delivered {_ago(dt)} ({label})"
        if status == "approved":
            created_at = order.get("created_at")
            if not created_at:
                return ""
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            expected_days = order.get("expected_delivery_days", 3)
            delay_days    = order.get("demo_delivery_delay_days", 0)
            total_secs    = (expected_days + delay_days) * 60   # 1 min = 1 day
            elapsed       = (datetime.now(timezone.utc) - created_at).total_seconds()
            remaining     = int(total_secs - elapsed)
            if remaining > 0:
                return f"🚚 Due in ~{remaining}s ({expected_days + delay_days} demo days)"
            return "🚚 Delivering imminently…"
    except Exception:
        pass
    return ""

# ---------------------------------------------------------------------------
# Authentication gate — renders login page when auth is enabled and user is not
# logged in; sets _current_role for the rest of the script on success.
# ---------------------------------------------------------------------------

if _auth_enabled:
    try:
        _name, _auth_status, _username = _authenticator.login("Login", "main")
    except Exception:
        _auth_status = None
        _name = ""
        _username = ""

    if _auth_status is False:
        st.error("Incorrect username or password.")
        st.stop()
    elif _auth_status is None:
        st.warning("Please enter your username and password.")
        st.stop()
    else:
        _current_role = _auth_config["credentials"]["usernames"].get(
            _username, {}
        ).get("role", "viewer")
        _authenticator.logout("Logout", "sidebar")

# ---------------------------------------------------------------------------
# CSS — Professional WMS-style theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
[data-testid="block-container"] { padding-top: 0.75rem; }

/* Top bar — MongoDB Forest → Evergreen gradient */
.top-bar {
    background: linear-gradient(135deg, #001E2B 0%, #023430 100%);
    border-radius: 10px; padding: 14px 20px; margin-bottom: 14px;
    display: flex; align-items: center; justify-content: space-between;
}
.top-bar h1 { color: #fff; font-size: 1.25rem; font-weight: 700; margin: 0; display: inline; }
.tech-pill {
    display: inline-block; font-size: 0.6rem; font-weight: 700;
    padding: 2px 8px; border-radius: 20px; margin-left: 6px;
    background: #1C2D38; color: #E8EDEB;
}
.tech-pill.mongo { background: #00684A; color: #00ED64; }
.live-indicator { color: #889397; font-size: 0.78rem; text-align: right; line-height: 1.6; }
.live-dot {
    display: inline-block; width: 8px; height: 8px;
    background: #00ED64; border-radius: 50%; margin-right: 5px;
    animation: blink 1.8s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* KPI row */
.kpi-row { display: flex; gap: 10px; margin-bottom: 16px; }
.kpi-card {
    flex: 1; background: #fff; border-radius: 10px;
    padding: 14px 18px; box-shadow: 0 1px 4px rgba(0,30,43,.08);
    border-top: 3px solid #E8EDEB;
}
.kpi-card.red   { border-top-color: #e53e3e; }
.kpi-card.amber { border-top-color: #d69e2e; }
.kpi-card.green { border-top-color: #00ED64; }
.kpi-card.blue  { border-top-color: #00684A; }
.kpi-lbl { font-size: 0.7rem; font-weight: 600; color: #5C6C75; text-transform: uppercase; letter-spacing:.06em; }
.kpi-val { font-size: 1.9rem; font-weight: 800; color: #001E2B; line-height: 1.15; }
.kpi-sub { font-size: 0.7rem; color: #889397; }

/* Panel headings */
.panel-hdr {
    font-size: 0.72rem; font-weight: 700; color: #1C2D38;
    text-transform: uppercase; letter-spacing: .08em;
    padding-bottom: 6px; border-bottom: 2px solid #00684A; margin-bottom: 10px;
}

/* Inventory cards */
.inv-card {
    background: #fff; border-radius: 10px; padding: 12px 14px;
    margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,30,43,.07);
    border-left: 4px solid #00ED64;
}
.inv-card.crit { border-left-color: #e53e3e; }
.inv-sku  { font-size: 0.7rem; font-weight: 700; color: #5C6C75; }
.inv-name { font-size: 0.86rem; font-weight: 600; color: #001E2B; line-height: 1.2; }
.inv-loc  { font-size: 0.67rem; color: #889397; margin-bottom: 6px; }
.bar-bg   { background: #E8EDEB; border-radius: 3px; height: 5px; margin: 6px 0 2px; }
.bar-fill { height: 5px; border-radius: 3px; }
.inv-nums { display: flex; gap: 10px; margin-top: 6px; flex-wrap: wrap; }
.inv-n    { text-align: center; }
.inv-nv   { font-size: 0.85rem; font-weight: 700; color: #1C2D38; }
.inv-nl   { font-size: 0.58rem; color: #889397; text-transform: uppercase; }

/* Feed cards (alerts + decisions) */
.feed-card {
    background: #fff; border-radius: 10px; padding: 12px 14px;
    margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,30,43,.07);
    border-left: 4px solid #E8EDEB;
}
.feed-card.crit   { border-left-color: #e53e3e; }
.feed-card.high   { border-left-color: #dd6b20; }
.feed-card.medium { border-left-color: #d69e2e; }
.feed-card.ok     { border-left-color: #00ED64; }
.feed-card.teal   { border-left-color: #00684A; }
.feed-hdr  { display: flex; justify-content: space-between; align-items: flex-start; }
.feed-sku  { font-size: 0.88rem; font-weight: 700; color: #001E2B; }
.feed-loc  { font-size: 0.72rem; color: #5C6C75; }
.feed-age  { font-size: 0.68rem; color: #889397; white-space: nowrap; }
.feed-nums { display: flex; gap: 14px; margin-top: 8px; flex-wrap: wrap; }
.feed-nv   { font-size: 0.88rem; font-weight: 700; color: #1C2D38; }
.feed-nl   { font-size: 0.62rem; color: #889397; text-transform: uppercase; }

/* Badges */
.badge {
    display: inline-block; font-size: 0.58rem; font-weight: 700;
    padding: 2px 6px; border-radius: 20px; margin-right: 3px;
    text-transform: uppercase; letter-spacing: .04em;
}
.b-crit   { background:#fed7d7; color:#9b2c2c; }
.b-high   { background:#feebc8; color:#7b341e; }
.b-med    { background:#fefcbf; color:#744210; }
.b-ok     { background:#CDFEE8; color:#023430; }
.b-info   { background:#E8EDEB; color:#1C2D38; }
.b-purple { background:#e9d8fd; color:#44337a; }
.b-teal   { background:#CDFEE8; color:#00684A; }
.b-amber  { background:#feebc8; color:#7b341e; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Simulator controls + Atlas Search: supplier capability search
# ---------------------------------------------------------------------------
with st.sidebar:

    # ── Simulator Controls ────────────────────────────────────────────────
    st.subheader("⚙️ Simulator Controls")

    ctrl_doc  = db.simulator_control.find_one({"_id": "main"})
    sim_state = ctrl_doc.get("state", "stopped") if ctrl_doc else "stopped"
    sim_speed = int(ctrl_doc.get("speed", 1)) if ctrl_doc else 1

    badge = {"running": "🟢 Running", "paused": "🟡 Paused", "stopped": "🔴 Stopped"}
    st.caption(f"Status: **{badge.get(sim_state, '❓ Unknown')}**")

    sc1, sc2, sc3 = st.columns(3)
    if sc1.button("▶ Start", disabled=(sim_state == "running"), use_container_width=True):
        db.simulator_control.update_one(
            {"_id": "main"}, {"$set": {"state": "running"}}, upsert=True
        )
        st.rerun()
    if sc2.button("⏸ Pause", disabled=(sim_state != "running"), use_container_width=True):
        db.simulator_control.update_one(
            {"_id": "main"}, {"$set": {"state": "paused"}}, upsert=True
        )
        st.rerun()
    if sc3.button("⏹ Stop", disabled=(sim_state == "stopped"), use_container_width=True):
        db.simulator_control.update_one(
            {"_id": "main"}, {"$set": {"state": "stopped"}}, upsert=True
        )
        st.rerun()

    # ── Drain speed slider ────────────────────────────────────────────────
    new_speed = st.select_slider(
        "⚡ Drain Speed",
        options=[1, 2, 3, 5, 10],
        value=sim_speed,
        format_func=lambda v: {1: "1× Normal", 2: "2× Fast", 3: "3× Faster",
                               5: "5× Demo", 10: "10× Chaos"}.get(v, f"{v}×"),
        help="Multiplies units consumed per tick — higher = alerts fire faster",
    )
    if new_speed != sim_speed:
        db.simulator_control.update_one(
            {"_id": "main"}, {"$set": {"speed": new_speed}}, upsert=True
        )

    st.divider()
    st.header("🔍 Supplier Search")
    st.caption("Powered by **Atlas $rankFusion** — hybrid name + capability search via RRF")

    # ── Use st.form so the auto-refresh (st.rerun every 10 s) cannot clear ──
    # the user's mid-type input.  Form widgets buffer their state and only      ──
    # fire when the submit button is clicked or Enter is pressed.              ──
    if "supplier_search_query" not in st.session_state:
        st.session_state.supplier_search_query = ""

    with st.form("supplier_search_form", clear_on_submit=False):
        raw_query = st.text_input(
            "Search supplier capabilities",
            value=st.session_state.supplier_search_query,
            placeholder="e.g. cold chain insulin, expedited PPE, emergency antibiotics",
        )
        col_btn, col_clr = st.columns([3, 1])
        submitted = col_btn.form_submit_button("🔍 Search", use_container_width=True)
        cleared   = col_clr.form_submit_button("✕ Clear",  use_container_width=True)

    if submitted and raw_query.strip():
        st.session_state.supplier_search_query = raw_query.strip()
    if cleared:
        st.session_state.supplier_search_query = ""

    search_query = st.session_state.supplier_search_query

    if search_query:
        # ── Primary: $rankFusion — two independent $search pipelines merged via RRF ──
        # name_match (precision) + capability_match (fuzzy recall), weighted 40/60.
        # Falls back to compound $search if $rankFusion is unavailable (M0/M2/M5 tiers).
        atlas_results = []
        atlas_error   = None
        _sidebar_proj = {
            "_id": 0, "supplier_name": 1, "supplier_id": 1,
            "sku": 1, "lead_time_days": 1, "fill_rate_pct": 1,
            "unit_price": 1, "notes": 1, "score": 1,
        }
        try:
            pipeline = [
                {
                    "$rankFusion": {
                        "input": {
                            "pipelines": {
                                "name_match": [
                                    {
                                        "$search": {
                                            "index": "suppliers_text_search",
                                            "text": {
                                                "query": search_query,
                                                "path":  "supplier_name",
                                            },
                                        }
                                    }
                                ],
                                "capability_match": [
                                    {
                                        "$search": {
                                            "index": "suppliers_text_search",
                                            "text": {
                                                "query": search_query,
                                                "path":  "notes",
                                                "fuzzy": {"maxEdits": 1},
                                            },
                                        }
                                    }
                                ],
                            }
                        },
                        "combination": {
                            "weights": {
                                "name_match":       0.4,
                                "capability_match": 0.6,
                            }
                        },
                    }
                },
                {"$addFields": {"score": {"$meta": "searchScore"}}},
                {"$project": _sidebar_proj},
                {"$limit": 8},
            ]
            atlas_results = list(db.suppliers.aggregate(pipeline))
        except Exception:
            # $rankFusion not available — try compound $search
            try:
                pipeline = [
                    {
                        "$search": {
                            "index": "suppliers_text_search",
                            "compound": {
                                "should": [
                                    {
                                        "text": {
                                            "query": search_query,
                                            "path":  "supplier_name",
                                            "score": {"boost": {"value": 3}},
                                        }
                                    },
                                    {
                                        "text": {
                                            "query": search_query,
                                            "path":  "notes",
                                            "fuzzy": {"maxEdits": 1},
                                        }
                                    },
                                ]
                            },
                        }
                    },
                    {"$addFields": {"score": {"$meta": "searchScore"}}},
                    {"$project": _sidebar_proj},
                    {"$limit": 8},
                ]
                atlas_results = list(db.suppliers.aggregate(pipeline))
            except Exception as exc:
                atlas_error = exc

        # ── Fallback: regex search when Atlas Search errors or returns nothing ──
        if atlas_error or not atlas_results:
            terms = search_query.split()
            regex_pat = "|".join(terms)
            fallback = list(db.suppliers.find(
                {"$or": [
                    {"supplier_name": {"$regex": regex_pat, "$options": "i"}},
                    {"notes":         {"$regex": regex_pat, "$options": "i"}},
                ]},
                {"_id": 0, "supplier_name": 1, "supplier_id": 1, "sku": 1,
                 "lead_time_days": 1, "fill_rate_pct": 1, "unit_price": 1, "notes": 1},
            ).limit(8))
            # Attach a dummy score so the display loop is uniform
            results  = [dict(r, score=None) for r in fallback]
            src_label = "🔄 Regex fallback"
            if atlas_error:
                st.warning(f"Atlas Search unavailable — showing regex results.\n\n`{atlas_error}`")
        else:
            results   = atlas_results
            src_label = "🔀 $rankFusion"

        if results:
            st.success(f"{len(results)} supplier(s) matched  ·  {src_label}")

            for r in results:
                with st.container(border=True):
                    rc1, rc2 = st.columns([3, 1])
                    rc1.markdown(f"**{r['supplier_name']}**")
                    rc2.markdown(f"`{r['sku']}`")
                    st.caption(
                        f"Lead {r['lead_time_days']}d · "
                        f"Fill {r['fill_rate_pct']}% · "
                        f"${r['unit_price']:.2f}/unit"
                    )
                    if r["score"] is not None:
                        st.progress(
                            min(r["score"] / 3.0, 1.0),
                            text=f"Relevance score: {r['score']:.3f}",
                        )
                    with st.expander("Capability notes"):
                        st.write(r.get("notes", "No notes available."))
        else:
            st.info("No suppliers matched. Try broader terms.")
    else:
        st.info("Type a capability and press **Search** (or Enter).")

    st.divider()
    st.caption("Atlas Search index: `suppliers_text_search` · hybrid via `$rankFusion`")
    st.caption("Atlas Vector Search index: `order_history_vector_index`")

    # ── Demo Status ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Demo Status")
    _last_alert = db.reorder_alerts.find_one(sort=[("created_at", -1)])
    _last_order = db.proposed_orders.find_one(sort=[("created_at", -1)])
    _total      = db.proposed_orders.count_documents({})
    _auto       = db.proposed_orders.count_documents({"auto_approved": True})
    _mems       = db.agent_memory.count_documents({})

    sc1, sc2 = st.columns(2)
    sc1.metric("Last Alert",    _ago(_last_alert.get("created_at")) if _last_alert else "—")
    sc2.metric("Last Decision", _ago(_last_order.get("created_at")) if _last_order else "—")

    sd1, sd2 = st.columns(2)
    sd1.metric("Total Decisions", _total)
    sd2.metric("Auto-Approved",   f"{_auto} ({int(_auto / max(_total, 1) * 100)}%)")

    st.metric("Long-term Memories", _mems)

    # ── Reset Demo ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔄 Reset Demo")
    st.caption("Clears alerts, orders, short-term memory and checkpoints. Inventory, suppliers, and long-term agent memory are preserved.")
    if st.button("🔄 Reset Demo", type="primary", use_container_width=True):
        # Clear all operational collections
        db.reorder_alerts.drop()
        db.reorder_alerts.create_index([("status", 1)])
        db.reorder_alerts.create_index([("sku", 1), ("created_at", 1)])
        db.reorder_alerts.create_index([("status", 1), ("created_at", 1)])
        db.proposed_orders.drop()
        db.proposed_orders.create_index([("status", 1)])
        db.proposed_orders.create_index([("alert_id", 1)])
        db.proposed_orders.create_index([("sku", 1), ("location", 1), ("status", 1)])
        db.proposed_orders.create_index([("created_at", 1)])
        db.short_term_memory.drop()
        db.checkpoints.drop()
        db.checkpoint_writes.drop()
        # Clear the Change Stream resume token — the reorder_alerts collection
        # is being dropped, so the saved token is now invalid.  The agent will
        # re-open a fresh stream from the current head on its next reconnect.
        db.agent_state.delete_one({"_id": "change_stream_resume_token"})
        # Reset on_order counters so inventory shows clean state
        db.inventory.update_many({}, {"$set": {"on_order": 0}})
        # Reset simulator to running at normal speed
        db.simulator_control.update_one(
            {"_id": "main"}, {"$set": {"state": "running", "speed": 1}}, upsert=True
        )
        st.success("✅ Demo reset! Collections cleared — simulator restarted at 1× speed.")
        st.rerun()

    # ── Admin Panel ───────────────────────────────────────────────────────
    if _check_role("admin"):
        st.divider()
        st.subheader("🛠 Admin Panel")

        # Circuit breaker reset
        import agent.graph as _agent_graph
        cb_open = _agent_graph._circuit_failures >= _agent_graph._CIRCUIT_THRESHOLD
        if cb_open:
            st.warning(f"⚡ Circuit breaker OPEN ({_agent_graph._circuit_failures} failures)")
            if st.button("🔌 Reset Circuit Breaker", use_container_width=True):
                _agent_graph._circuit_failures = 0
                st.success("Circuit breaker reset — LLM calls resumed.")
                st.rerun()
        else:
            st.caption(f"Circuit breaker: 🟢 Closed ({_agent_graph._circuit_failures}/{_agent_graph._CIRCUIT_THRESHOLD} failures)")

        # Procedural memory extraction
        st.caption("Extract supplier preference rules from repeated approvals (≥5 same supplier in 30 days).")
        if st.button("📐 Extract Rules", use_container_width=True):
            try:
                from agent.procedure_extractor import extract_procedures
                candidates = extract_procedures()
                if candidates:
                    st.success(f"✅ Extracted {len(candidates)} candidate rule(s). Review them in the Procedures collection.")
                else:
                    st.info("No new patterns found yet — need ≥5 approvals for the same supplier/category/location.")
            except Exception as _exc:
                st.error(f"Extraction failed: {_exc}")

        # Procedure candidate review
        _pending_rules = list(db.procedures.find(
            {"human_confirmed": False},
            {"sku_category": 1, "location": 1, "preferred_supplier_name": 1,
             "preferred_supplier_id": 1, "evidence_count": 1, "created_at": 1},
        ).sort("evidence_count", -1))
        if _pending_rules:
            with st.expander(f"📋 {len(_pending_rules)} candidate rule(s) awaiting confirmation", expanded=True):
                st.caption("Rules are only passed to the agent after you confirm them.")
                for _rule in _pending_rules:
                    _rc, _rb = st.columns([5, 2])
                    _rc.markdown(
                        f"**{_rule['sku_category']}** @ `{_rule['location']}`  →  "
                        f"**{_rule['preferred_supplier_name']}**  "
                        f"*(ID: {_rule['preferred_supplier_id']}, "
                        f"{_rule['evidence_count']} approvals)*"
                    )
                    _rb1, _rb2 = _rb.columns(2)
                    _rule_id_str = str(_rule["_id"])
                    if _rb1.button("✅", key=f"confirm_rule_{_rule_id_str}",
                                   help="Confirm — agent will use this rule",
                                   use_container_width=True):
                        st.session_state["_pending_procedure_action"] = {
                            "action": "confirm", "rule_id_str": _rule_id_str,
                        }
                        st.rerun()
                    if _rb2.button("🗑", key=f"dismiss_rule_{_rule_id_str}",
                                   help="Dismiss — delete this candidate",
                                   use_container_width=True):
                        st.session_state["_pending_procedure_action"] = {
                            "action": "dismiss", "rule_id_str": _rule_id_str,
                        }
                        st.rerun()

        # Memory compaction
        st.caption("Deduplicate and summarize old agent memories (entries older than 30 days).")
        if st.button("🗜 Compact Memory", use_container_width=True):
            try:
                from agent.memory_compactor import run_compaction
                result = run_compaction()
                st.success(
                    f"✅ Compaction done — {result.get('deduped', 0)} duplicates merged, "
                    f"{result.get('compacted', 0)} old entries summarized."
                )
            except Exception as _exc:
                st.error(f"Compaction failed: {_exc}")

        # ── Agent Recovery via LangGraph Checkpoints ──────────────────────
        # MongoDB persists LangGraph graph state node-by-node. If the agent
        # process crashes mid-pipeline, the next alert re-invocation resumes
        # from the last checkpoint rather than restarting from scratch.
        st.divider()
        st.caption("LangGraph checkpoints stored in MongoDB — enables crash recovery mid-pipeline.")
        with st.expander("🔁 Agent Recovery Log", expanded=False):
            _ckpt_rows = list(db.checkpoints.aggregate([
                {"$sort": {"checkpoint_id": -1}},
                {"$group": {
                    "_id":                   "$thread_id",
                    "latest_metadata":       {"$first": "$metadata"},
                    "latest_checkpoint_id":  {"$first": "$checkpoint_id"},
                    "total_steps":           {"$sum": 1},
                }},
                {"$sort": {"latest_checkpoint_id": -1}},
                {"$limit": 10},
            ]))

            if not _ckpt_rows:
                st.info("No checkpoints yet. Process an alert to generate pipeline state.")
            else:
                # Batch-fetch alert and order data for all thread_ids
                _ck_alert_ids = []
                for _cr in _ckpt_rows:
                    try:
                        _ck_alert_ids.append(ObjectId(_cr["_id"]))
                    except Exception:
                        pass
                _ck_alert_map = {
                    str(a["_id"]): a
                    for a in db.reorder_alerts.find(
                        {"_id": {"$in": _ck_alert_ids}},
                        {"sku": 1, "location": 1, "status": 1},
                    )
                }
                _ck_order_map = {
                    str(o.get("alert_id", "")): o
                    for o in db.proposed_orders.find(
                        {"alert_id": {"$in": _ck_alert_ids}},
                        {"alert_id": 1, "status": 1, "auto_approved": 1},
                    )
                }

                _FULL_STEPS = 6  # typical completed run (assess → retrieve → analyse → recommend → audit → save)
                st.caption(f"{len(_ckpt_rows)} recent pipeline run(s) · thread ID = alert `_id`")

                for _cr in _ckpt_rows:
                    _tid          = _cr["_id"]
                    _ck_alert     = _ck_alert_map.get(_tid, {})
                    _ck_order     = _ck_order_map.get(_tid, {})
                    _ck_meta      = _cr.get("latest_metadata") or {}
                    _ck_writes    = (_ck_meta.get("writes") or {}) if isinstance(_ck_meta, dict) else {}
                    _last_node    = list(_ck_writes.keys())[0] if _ck_writes else "—"
                    _steps        = _cr.get("total_steps", 0)
                    _order_status = _ck_order.get("status")
                    _alert_status = _ck_alert.get("status", "?")

                    if _order_status in ("approved", "received", "awaiting_approval"):
                        _ri, _rl, _rc = "✅", "completed", "#00684A"
                    elif _order_status == "rejected":
                        _ri, _rl, _rc = "🔄", "re-queued", "#d69e2e"
                    elif _alert_status == "escalated":
                        _ri, _rl, _rc = "🔺", "escalated", "#e53e3e"
                    elif _alert_status in ("processing", "pending") and _steps < _FULL_STEPS:
                        _ri, _rl, _rc = "⚡", "recovered on restart", "#d69e2e"
                    else:
                        _ri, _rl, _rc = "✅", "completed", "#00684A"

                    _sku_lbl = _ck_alert.get("sku", f"{_tid[:8]}…")
                    _loc_lbl = _ck_alert.get("location", "?")
                    st.markdown(
                        f'<div style="background:#f7f9f7;border-left:3px solid {_rc};'
                        f'padding:7px 10px;border-radius:4px;margin-bottom:6px">'
                        f'<span style="font-size:0.82rem;font-weight:700">{_ri} {_sku_lbl}</span>'
                        f' <span style="font-size:0.7rem;color:#5C6C75">@ {_loc_lbl}</span>'
                        f'<span style="float:right;font-size:0.66rem;color:#889397">'
                        f'{_steps} steps · last: <code>{_last_node}</code></span>'
                        f'<div style="font-size:0.64rem;color:#889397;margin-top:3px">'
                        f'<code>{_tid[:20]}…</code> · alert: {_alert_status} · {_rl}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
inventory_docs = list(db.inventory.find({}, {"_id": 0}).sort("sku", 1))
inventory_docs.sort(key=lambda x: (0 if x["on_hand"] < x["reorder_point"] else 1, x["sku"]))

alert_docs      = list(db.reorder_alerts.find({"status": "pending"}).sort("created_at", -1).limit(20))
escalation_docs      = list(db.escalation_queue.find().sort("escalated_at", -1).limit(10))
escalated_count      = db.reorder_alerts.count_documents({"status": "escalated"})
failed_writes_count  = db.failed_memory_writes.count_documents({"resolved": False})
# Always show ALL pending approvals; cap non-pending history at 10.
_pending_orders = list(db.proposed_orders.find(
    {"status": "awaiting_approval"}
).sort("created_at", -1))
_history_orders = list(db.proposed_orders.find(
    {"status": {"$ne": "awaiting_approval"}}
).sort("created_at", -1).limit(5))
_history_orders.sort(key=lambda o: (
    {"approved": 0, "received": 1, "rejected": 2}.get(o.get("status"), 1),
    -(o["created_at"].timestamp() if hasattr(o.get("created_at"), "timestamp") else 0),
))
order_docs = _pending_orders + _history_orders

# ROP health: avg daily consumption (last 30 days) and best-supplier lead time per SKU.
# Two single-pass aggregations so the inventory loop has O(1) per-card lookup.
_rop_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
_avg_daily_map: dict = {
    r["_id"]: round(r["avg_daily"], 1)
    for r in db.consumption_history.aggregate([
        {"$match": {"timestamp": {"$gte": _rop_cutoff}}},
        {"$group": {
            "_id":     "$sku",
            "total":   {"$sum": "$quantity"},
            "day_set": {"$addToSet": {
                "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}
            }},
        }},
        {"$project": {"avg_daily": {"$divide": ["$total", {"$size": "$day_set"}]}}},
    ])
}
_lead_time_map: dict = {
    r["_id"]: r["lead_time"]
    for r in db.suppliers.aggregate([
        {"$sort": {"fill_rate_pct": -1}},
        {"$group": {"_id": "$sku", "lead_time": {"$first": "$lead_time_days"}}},
    ])
}

# KPI aggregates — single $facet round-trip instead of multiple count_documents
below_reorder  = sum(1 for i in inventory_docs if i["on_hand"] < i["reorder_point"])
awaiting_count = sum(1 for o in order_docs if o.get("status") == "awaiting_approval")

# Count SKUs whose ROP fires before the preferred supplier can deliver
_rop_issues = sum(
    1 for i in inventory_docs
    if _avg_daily_map.get(i["sku"], 0) > 0
    and _lead_time_map.get(i["sku"], 0) > 0
    and (i["reorder_point"] / _avg_daily_map[i["sku"]]) < _lead_time_map[i["sku"]]
)

_order_kpis = list(db.proposed_orders.aggregate([
    {"$facet": {
        "total":     [{"$count": "n"}],
        "auto":      [{"$match": {"auto_approved": True}}, {"$count": "n"}],
    }}
]))
_total_orders   = _order_kpis[0]["total"][0]["n"]  if _order_kpis and _order_kpis[0]["total"]  else 0
_total_auto     = _order_kpis[0]["auto"][0]["n"]   if _order_kpis and _order_kpis[0]["auto"]   else 0
received_count  = db.proposed_orders.count_documents({"status": "received"})
mem_count       = db.agent_memory.count_documents({})

_kpi_below  = "red"   if below_reorder >= 4  else ("amber" if below_reorder >= 1 else "green")
_kpi_alerts = "red"   if len(alert_docs) >= 5 else ("amber" if alert_docs else "green")
_kpi_queue  = "amber" if awaiting_count >= 1  else "green"
_kpi_esc    = "red"   if escalated_count >= 1  else "green"

# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="top-bar">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
    <span style="font-size:1.5rem">🏥</span>
    <h1>Supply Chain Reorder Agent</h1>
    <span class="tech-pill mongo">MongoDB Atlas</span>
    <span class="tech-pill">LangGraph</span>
    <span class="tech-pill">GPT-5.4</span>
    <span class="tech-pill">Voyage AI</span>
  </div>
  <div class="live-indicator">
    <span class="live-dot"></span><strong style="color:#E8EDEB">LIVE</strong>
    &nbsp;·&nbsp; {time.strftime('%H:%M:%S')}<br>
    Change Streams &nbsp;·&nbsp; Atlas Search &nbsp;·&nbsp; Vector Search
  </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card {_kpi_below}">
    <div class="kpi-lbl">📦 Below Reorder</div>
    <div class="kpi-val">{below_reorder}</div>
    <div class="kpi-sub">of {len(inventory_docs)} monitored SKUs</div>
  </div>
  <div class="kpi-card {_kpi_alerts}">
    <div class="kpi-lbl">🚨 Active Alerts</div>
    <div class="kpi-val">{len(alert_docs)}</div>
    <div class="kpi-sub">pending reorder events</div>
  </div>
  <div class="kpi-card {_kpi_queue}">
    <div class="kpi-lbl">⏳ Approval Queue</div>
    <div class="kpi-val">{awaiting_count}</div>
    <div class="kpi-sub">orders awaiting review</div>
  </div>
  <div class="kpi-card {"amber" if _rop_issues > 0 else "green"}">
    <div class="kpi-lbl">📐 ROP Health</div>
    <div class="kpi-val">{_rop_issues}</div>
    <div class="kpi-sub">SKU{"s" if _rop_issues != 1 else ""} with ROP below lead-time demand</div>
  </div>
  <div class="kpi-card blue">
    <div class="kpi-lbl">⚡ Auto-Approved</div>
    <div class="kpi-val">{_total_auto}</div>
    <div class="kpi-sub">{int(_total_auto/max(_total_orders,1)*100)}% of {_total_orders} decisions · {mem_count} memories · {failed_writes_count} failed writes</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-lbl">📬 Delivered</div>
    <div class="kpi-val">{received_count}</div>
    <div class="kpi-sub">orders received (1 min = 1 day)</div>
  </div>
  <div class="kpi-card {_kpi_esc}">
    <div class="kpi-lbl">🔺 Escalated</div>
    <div class="kpi-val">{escalated_count}</div>
    <div class="kpi-sub">alerts requiring human review</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main 2-panel layout
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1.1, 1])

# ── Left panel: Inventory Grid ─────────────────────────────────────────────
with left_col:
    st.markdown('<div class="panel-hdr">📦 Live Inventory</div>', unsafe_allow_html=True)
    if not inventory_docs:
        st.info("No inventory data. Run `python data/seed.py` first.")
    else:
        for i in range(0, len(inventory_docs), 2):
            row_items = inventory_docs[i:i + 2]
            cols = st.columns(len(row_items))
            for col, item in zip(cols, row_items):
                with col:
                    on_hand  = item["on_hand"]
                    on_order = item.get("on_order", 0)
                    reorder  = item["reorder_point"]
                    below    = on_hand < reorder
                    eff      = on_hand + on_order
                    pct      = min(on_hand / max(reorder, 1), 1.0) * 100
                    bar_clr  = "#e53e3e" if below else "#00ED64"
                    card_cls = "crit" if below else ""
                    sbadge   = (
                        '<span class="badge b-crit">CRITICAL</span>' if pct < 25
                        else '<span class="badge b-high">LOW</span>' if below
                        else '<span class="badge b-ok">OK</span>'
                    )

                    # ROP health indicator
                    _avg_d = _avg_daily_map.get(item["sku"], 0)
                    _lead  = _lead_time_map.get(item["sku"], 0)
                    if _avg_d > 0 and _lead > 0:
                        _days_at_rop = round(reorder / _avg_d, 1)
                        _gap         = round(_lead - _days_at_rop, 1)
                        if _days_at_rop >= _lead:
                            _rop_icon, _rop_clr = "✅", "#00684A"
                            _rop_txt = f"ROP fires at {_days_at_rop}d · lead {_lead}d — covered"
                        elif _days_at_rop >= _lead * 0.7:
                            _rop_icon, _rop_clr = "⚠️", "#b7791f"
                            _rop_txt = f"ROP fires at {_days_at_rop}d · lead {_lead}d — safety stock bridges {_gap}d"
                        else:
                            _rop_icon, _rop_clr = "🔴", "#c53030"
                            _rop_txt = f"ROP fires at {_days_at_rop}d · lead {_lead}d — undercalibrated by {_gap}d"
                        _rop_html = (
                            f'<div style="font-size:0.6rem;margin-top:5px;'
                            f'padding:2px 6px;border-radius:4px;'
                            f'background:{"#f0fff4" if _days_at_rop >= _lead else "#fffbeb" if _days_at_rop >= _lead * 0.7 else "#fff5f5"};'
                            f'color:{_rop_clr};border:1px solid {_rop_clr}22">'
                            f'{_rop_icon} {_rop_txt}</div>'
                        )
                    else:
                        _rop_html = ""

                    st.markdown(f"""
<div class="inv-card {card_cls}">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span class="inv-sku">{item['sku']}</span>{sbadge}
  </div>
  <div class="inv-name">{item['name']}</div>
  <div class="inv-loc">{item['location']} · {item['category']}</div>
  <div class="bar-bg"><div class="bar-fill" style="width:{pct:.0f}%;background:{bar_clr}"></div></div>
  <div style="font-size:0.62rem;color:#5C6C75;margin-bottom:4px">{pct:.0f}% of reorder level</div>
  <div class="inv-nums">
    <div class="inv-n"><div class="inv-nv">{on_hand:,}</div><div class="inv-nl">On Hand</div></div>
    <div class="inv-n"><div class="inv-nv" style="color:#00684A">{on_order:,}</div><div class="inv-nl">On Order</div></div>
    <div class="inv-n"><div class="inv-nv">{reorder:,}</div><div class="inv-nl">Reorder Pt</div></div>
    <div class="inv-n"><div class="inv-nv" style="color:#023430">{eff:,}</div><div class="inv-nl">Effective</div></div>
  </div>
  {_rop_html}
</div>""", unsafe_allow_html=True)

# ── Right panel: Alerts + Decisions ───────────────────────────────────────
with right_col:

    # Active Alerts
    st.markdown('<div class="panel-hdr">🚨 Active Alerts</div>', unsafe_allow_html=True)
    if not alert_docs:
        st.success("✅ No pending alerts — all stock levels OK.")
    else:
        for alert in alert_docs:
            days = alert.get("days_of_stock_remaining", 0)
            if days < 2:
                sev_cls, sev_b = "crit",   '<span class="badge b-crit">CRITICAL</span>'
            elif days < 7:
                sev_cls, sev_b = "high",   '<span class="badge b-high">HIGH</span>'
            else:
                sev_cls, sev_b = "medium", '<span class="badge b-med">WATCH</span>'
            has_order = any(str(o.get("alert_id")) == str(alert["_id"]) for o in order_docs)
            agent_b   = (
                '<span class="badge b-ok">ORDER DRAFTED</span>' if has_order
                else '<span class="badge b-info">PROCESSING…</span>'
            )
            st.markdown(f"""
<div class="feed-card {sev_cls}">
  <div class="feed-hdr">
    <div><span class="feed-sku">{alert['sku']}</span>
    <span class="feed-loc"> @ {alert['location']}</span></div>
    <span class="feed-age">{_ago(alert.get('created_at'))}</span>
  </div>
  <div style="margin:4px 0">{sev_b}{agent_b}</div>
  <div class="feed-nums">
    <div><div class="feed-nv">{alert['on_hand']:,}</div><div class="feed-nl">On Hand</div></div>
    <div><div class="feed-nv">{float(days):.1f}d</div><div class="feed-nl">Days Left</div></div>
    <div><div class="feed-nv">{alert['reorder_point']:,}</div><div class="feed-nl">Reorder Pt</div></div>
    <div><div class="feed-nv">{alert['avg_daily_consumption']} u/d</div><div class="feed-nl">Avg Daily</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    # Agent Decisions
    st.markdown('<div class="panel-hdr" style="margin-top:18px">🤖 Agent Decisions</div>',
                unsafe_allow_html=True)
    if not order_docs:
        st.info("No proposed orders yet.")
    else:
        for order in order_docs:
            status        = order.get("status", "awaiting_approval")
            confidence    = order.get("confidence", "medium")
            auto_approved = order.get("auto_approved", False)
            used_search   = order.get("atlas_search_used", False)

            review_reason = order.get("review_reason")
            conf_b   = {"high": '<span class="badge b-ok">HIGH</span>',
                        "medium": '<span class="badge b-med">MEDIUM</span>',
                        "low":    '<span class="badge b-crit">LOW</span>'}.get(confidence, "")
            status_b = {"awaiting_approval": '<span class="badge b-info">AWAITING</span>',
                        "approved":          '<span class="badge b-ok">APPROVED</span>',
                        "rejected":          '<span class="badge b-crit">REJECTED</span>',
                        "received":          '<span class="badge b-teal">📦 RECEIVED</span>'}.get(status, "")
            auto_b    = '<span class="badge b-purple">⚡ AUTO</span>'   if auto_approved else ""
            srch_b    = '<span class="badge b-info">🔍 SEARCH</span>'  if used_search   else ""
            review_b  = (
                f'<span class="badge b-amber" title="{review_reason}">⚠ REVIEW: {review_reason}</span>'
                if review_reason and status == "awaiting_approval" else ""
            )
            card_cls = ("teal" if status == "received"
                        else "ok" if status == "approved"
                        else "crit" if status == "rejected"
                        else "medium")

            timing_line = _delivery_timing(order)
            timing_html = (
                f'<div style="font-size:0.7rem;color:#5C6C75;margin:3px 0 2px">{timing_line}</div>'
                if timing_line else ""
            )
            with st.container():
                st.markdown(f"""
<div class="feed-card {card_cls}">
  <div class="feed-hdr">
    <div><span class="feed-sku">{order['sku']}</span>
    <span class="feed-loc"> @ {order['location']}</span></div>
    <span class="feed-age">{_ago(order.get('created_at'))}</span>
  </div>
  <div style="margin:4px 0">{conf_b}{status_b}{auto_b}{srch_b}{review_b}</div>
  {timing_html}<div class="feed-nums">
    <div><div class="feed-nv">{order['quantity_recommended']:,}</div><div class="feed-nl">Qty</div></div>
    <div><div class="feed-nv">${order['total_cost']:,.0f}</div><div class="feed-nl">Cost</div></div>
    <div><div class="feed-nv">{order['expected_delivery_days']}d</div><div class="feed-nl">Lead</div></div>
    <div style="flex:2"><div class="feed-nv" style="font-size:0.78rem">{order['supplier_name']}</div><div class="feed-nl">Supplier</div></div>
  </div>
</div>""", unsafe_allow_html=True)

                similar = [s for s in order.get("similar_orders", [])
                           if "error" not in s and "info" not in s]
                with st.expander("💬 Rationale  /  🧠 Vector Search", expanded=False):
                    st.write(order.get("rationale", "N/A"))
                    if similar:
                        st.caption(f"🧠 {len(similar)} similar past order(s) via Vector Search")
                        for s in similar:
                            st.markdown(
                                f"**{s['sku']}** @ {s.get('location','?')} · "
                                f"{s.get('days_of_stock_at_order','?')}d remaining · "
                                f"trend: *{s.get('trend_at_order','?')}* · "
                                f"outcome: **{s.get('outcome','?')}** · "
                                f"score: `{s.get('similarity_score',0):.3f}`"
                            )

                if status == "awaiting_approval":
                    btn1, btn2 = st.columns(2)
                    oid = order["_id"]
                    _can_approve = _check_role("approver")
                    if not _can_approve:
                        st.caption("🔒 Viewer access — approval requires Approver or Admin role.")
                    if _can_approve:
                        # Detect which button (if any) is currently in flight for this order
                        _inflight = st.session_state.get("_pending_human_decision", {})
                        _inflight_oid = _inflight.get("oid_str") == str(oid)
                        _inflight_action = _inflight.get("action") if _inflight_oid else None

                        _approve_primary = _inflight_action == "approve"
                        _reject_primary  = _inflight_action == "reject"

                        if btn1.button(
                            "✅ Approve",
                            key=f"approve_{oid}",
                            use_container_width=True,
                            type="primary" if _approve_primary else "secondary",
                            disabled=_reject_primary,
                        ):
                            st.session_state["_pending_human_decision"] = {
                                "action":       "approve",
                                "oid_str":      str(oid),
                                "alert_id_str": str(order.get("alert_id", "")),
                            }
                            st.rerun()
                        if btn2.button(
                            "❌ Reject",
                            key=f"reject_{oid}",
                            use_container_width=True,
                            type="primary" if _reject_primary else "secondary",
                            disabled=_approve_primary,
                        ):
                            st.session_state["_pending_human_decision"] = {
                                "action":       "reject",
                                "oid_str":      str(oid),
                                "alert_id_str": str(order.get("alert_id", "")),
                            }
                            st.rerun()


    # Confidence Calibration
    with st.expander("📊 Confidence Calibration", expanded=False):
        calibration = list(db.confidence_outcomes.aggregate([
            {"$group": {
                "_id": {
                    "confidence": "$predicted_confidence",
                    "outcome":    "$outcome",
                },
                "count": {"$sum": 1},
            }},
            {"$sort": {"_id.confidence": 1, "_id.outcome": 1}},
        ]))
        if calibration:
            summary: dict = {}
            for row in calibration:
                conf    = row["_id"]["confidence"]
                outcome = row["_id"]["outcome"]
                summary.setdefault(conf, {})[outcome] = row["count"]

            rows_html = ""
            for conf in ["high", "medium", "low"]:
                if conf not in summary:
                    continue
                counts  = summary[conf]
                total   = sum(counts.values())
                resolved = counts.get("resolved", 0)
                pct      = int(resolved / max(total, 1) * 100)
                rows_html += (
                    f"<tr><td><b>{conf}</b></td>"
                    f"<td>{total}</td>"
                    f"<td>{resolved} ({pct}%)</td>"
                    f"<td>{counts.get('escalated', 0)}</td>"
                    f"<td>{counts.get('pending', 0)}</td></tr>"
                )
            st.markdown(
                "<table style='width:100%;font-size:0.82rem'>"
                "<tr><th>Confidence</th><th>Total</th><th>Resolved</th>"
                "<th>Escalated</th><th>Pending</th></tr>"
                + rows_html + "</table>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No calibration data yet — data accumulates as orders are processed.")

    # Escalated Alerts
    if escalation_docs:
        st.markdown(
            '<div class="panel-hdr" style="margin-top:18px">🔺 Escalated Alerts</div>',
            unsafe_allow_html=True,
        )
        for esc in escalation_docs:
            reason = esc.get("escalation_reason", "unknown")
            reason_label = {
                "human_rejections":  "⛔ Rejected 3× by humans",
                "audit_max_retries": "⚠️ Audit max retries exhausted",
            }.get(reason, reason)
            escalated_at = esc.get("escalated_at")
            st.markdown(f"""
<div class="feed-card crit">
  <div class="feed-hdr">
    <div><span class="feed-sku">{esc.get('sku','?')}</span>
    <span class="feed-loc"> @ {esc.get('location','?')}</span></div>
    <span class="feed-age">{_ago(escalated_at)}</span>
  </div>
  <div style="margin:4px 0">
    <span class="badge b-crit">ESCALATED</span>
    <span class="badge b-high">{reason_label}</span>
  </div>
  <div class="feed-nums">
    <div><div class="feed-nv">{esc.get('rejection_count', 0)}</div>
         <div class="feed-nl">Rejections</div></div>
  </div>
</div>""", unsafe_allow_html=True)

            last_rec = esc.get("last_recommendation") or {}
            last_errors = esc.get("last_audit_errors") or []
            if last_rec or last_errors:
                with st.expander("Details", expanded=False):
                    if last_rec:
                        st.caption("Last recommendation:")
                        st.json(last_rec)
                    if last_errors:
                        st.caption("Audit errors:")
                        for e in last_errors:
                            st.markdown(f"- {e}")

# ---------------------------------------------------------------------------
# Auto-refresh — triggers a Streamlit rerun every REFRESH_INTERVAL seconds.
# st_autorefresh runs at the top level so it fires regardless of active tab.
# ---------------------------------------------------------------------------
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="dashboard_refresh")
