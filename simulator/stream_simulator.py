"""
Stream simulator — mimics what Atlas Stream Processing would produce.

Every 5 seconds it picks a random SKU, simulates consumption, and writes a
reorder_alert document when stock drops below the reorder point.

Run after seed.py:
    python simulator/stream_simulator.py
"""

import os
import random
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.logger import get_logger
from agent.tools import append_lifecycle_event

load_dotenv()

log = get_logger(__name__)

client = MongoClient(
    os.environ["MONGODB_URI"],
    serverSelectionTimeoutMS=30_000,
    connectTimeoutMS=10_000,
    socketTimeoutMS=30_000,
)
db = client["supply_chain_demo"]

inventory           = db["inventory"]
consumption_history = db["consumption_history"]
alerts_collection   = db["reorder_alerts"]
control_collection  = db["simulator_control"]
orders_collection   = db["proposed_orders"]

# ---------------------------------------------------------------------------
# Demo time: 1 real-world minute = 1 demo day.
# Orders marked "approved" are delivered after their expected_delivery_days
# have elapsed (in demo-minutes). 5% of orders arrive 1-3 days late.
# ---------------------------------------------------------------------------
_DEMO_SECS_PER_DAY = 60   # 1 minute = 1 day

# ---------------------------------------------------------------------------
# Simulator control — state is stored in MongoDB so the Streamlit dashboard
# can start, pause, and stop the simulator without restarting the container.
# States: "running" | "paused" | "stopped"
# "paused" and "stopped" both halt event emission; the process keeps running
# so the dashboard can resume it with ▶ Start at any time.
# ---------------------------------------------------------------------------
_CONTROL_ID = "main"

def _get_control() -> dict:
    """Return the full simulator_control document (state + speed)."""
    doc = control_collection.find_one({"_id": _CONTROL_ID})
    return doc if doc else {"state": "running", "speed": 1}

def _get_state() -> str:
    return _get_control().get("state", "running")

def _set_state(state: str) -> None:
    control_collection.update_one(
        {"_id": _CONTROL_ID}, {"$set": {"state": state}}, upsert=True
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_avg_daily_consumption(sku: str) -> float:
    """Aggregate the time series collection to get average daily consumption."""
    pipeline = [
        {"$match": {"sku": sku}},
        {
            "$group": {
                "_id": {
                    "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}
                },
                "daily_total": {"$sum": "$quantity"},
            }
        },
        {"$group": {"_id": None, "avg_daily": {"$avg": "$daily_total"}}},
    ]
    result = list(consumption_history.aggregate(pipeline))
    return round(result[0]["avg_daily"], 1) if result else 0.0


def record_consumption(sku: str, location: str, quantity: int) -> None:
    """Append a consumption event to the time series collection."""
    consumption_history.insert_one({
        "timestamp": datetime.now(timezone.utc),
        "sku": sku,
        "location": location,
        "quantity": quantity,
        "reason": "simulated_event",
    })


def check_and_alert(sku_doc: dict, consumption: int) -> None:
    """
    Subtract consumption, update inventory, and write a reorder_alert if
    on_hand + on_order has fallen below the reorder point.
    """
    sku = sku_doc["sku"]
    on_hand = max(0, sku_doc["on_hand"] - consumption)
    on_order = sku_doc["on_order"]
    reorder_point = sku_doc["reorder_point"]

    # Persist inventory change
    inventory.update_one({"sku": sku}, {"$set": {"on_hand": on_hand}})

    # Record in time series
    record_consumption(sku, sku_doc["location"], consumption)

    effective_stock = on_hand + on_order

    if effective_stock < reorder_point:
        avg_daily = get_avg_daily_consumption(sku)
        days_remaining = round(on_hand / avg_daily, 1) if avg_daily > 0 else 0.0

        # Skip if an unprocessed alert already exists for this SKU + location
        existing = alerts_collection.find_one({
            "sku": sku,
            "location": sku_doc["location"],
            "status": "pending",
        })
        if existing:
            log.info("alert already pending, skipping", extra={
                "sku": sku, "location": sku_doc["location"],
                "on_hand": on_hand, "days_remaining": days_remaining,
            })
            return

        alert = {
            "sku": sku,
            "location": sku_doc["location"],
            "on_hand": on_hand,
            "on_order": on_order,
            "reorder_point": reorder_point,
            "units_consumed_last_15min": consumption,
            "avg_daily_consumption": avg_daily,
            "days_of_stock_remaining": days_remaining,
            "status": "pending",
            "rejection_count": 0,
            "created_at": datetime.now(timezone.utc),
        }
        alerts_collection.insert_one(alert)
        append_lifecycle_event(
            str(alert["_id"]),
            "alert_created",
            {"on_hand": on_hand, "days_remaining": days_remaining,
             "avg_daily": avg_daily, "reorder_point": reorder_point},
            sku,
            sku_doc["location"],
        )
        log.warning("reorder alert created", extra={
            "sku": sku, "location": sku_doc["location"],
            "on_hand": on_hand, "days_remaining": days_remaining,
        })
    else:
        log.info("consumption recorded, stock OK", extra={
            "sku": sku, "location": sku_doc["location"],
            "consumed": consumption, "on_hand": on_hand, "reorder_point": reorder_point,
        })
        # Fire stock_recovered once per alert episode: find the most recent
        # processed alert that hasn't been marked recovered yet, then flip its
        # status so subsequent OK ticks don't re-fire the event.
        recent_alert = alerts_collection.find_one_and_update(
            {
                "sku": sku,
                "location": sku_doc["location"],
                "status": "processed",
            },
            {"$set": {"status": "recovered"}},
            sort=[("created_at", -1)],
        )
        if recent_alert:
            avg_daily = get_avg_daily_consumption(sku)
            append_lifecycle_event(
                str(recent_alert["_id"]),
                "stock_recovered",
                {"on_hand": on_hand, "avg_daily": avg_daily},
                sku,
                sku_doc["location"],
            )
            # Update confidence_outcomes with recovery data
            created_at = recent_alert.get("created_at")
            if created_at:
                now_utc = datetime.now(timezone.utc)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                days_to_recovery = round(
                    (now_utc - created_at).total_seconds() / 86400, 1
                )
                db.confidence_outcomes.update_one(
                    {"alert_id": recent_alert["_id"], "outcome": {"$ne": "resolved"}},
                    {"$set": {
                        "days_to_stock_recovery": days_to_recovery,
                        "outcome":                "resolved",
                    }},
                )
                # Graduate the live order into a completed precedent
                order_id = recent_alert.get("order_id")
                if order_id:
                    db.order_history.update_one(
                        {"proposed_order_id": order_id},
                        {"$set": {"outcome": "delivered_on_time"}},
                    )


# ---------------------------------------------------------------------------
# Delivery simulation
# ---------------------------------------------------------------------------

def deliver_pending_orders() -> None:
    """
    Check approved orders and simulate delivery using demo time (1 min = 1 day).
    95% of orders deliver on time; 5% are delayed 1–3 extra days.
    On delivery: updates proposed_orders, inventory, order_history,
    confidence_outcomes, and appends a lifecycle event.
    """
    now = datetime.now(timezone.utc)
    approved = list(orders_collection.find(
        {
            "status": "approved",
            "delivered_at": {"$exists": False},
            "quantity_recommended": {"$gt": 0},
        },
        limit=50,
    ))

    for order in approved:
        sku      = order["sku"]
        location = order["location"]
        qty      = order["quantity_recommended"]
        oid      = order["_id"]

        created_at = order.get("created_at")
        if created_at is None:
            continue
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        # Assign late-delivery penalty on first encounter (stored on doc)
        if "demo_delivery_delay_days" not in order:
            delay = random.randint(1, 3) if random.random() > 0.95 else 0
            orders_collection.update_one(
                {"_id": oid},
                {"$set": {"demo_delivery_delay_days": delay}},
            )
            order["demo_delivery_delay_days"] = delay

        delay_days = order.get("demo_delivery_delay_days", 0)
        expected_days = order.get("expected_delivery_days", 3)
        total_demo_secs = (expected_days + delay_days) * _DEMO_SECS_PER_DAY
        elapsed_secs = (now - created_at).total_seconds()

        if elapsed_secs < total_demo_secs:
            continue

        on_time = delay_days == 0
        outcome = "delivered_on_time" if on_time else "delivery_delayed"

        # Update inventory: add incoming stock, clear on_order
        inv_doc = inventory.find_one({"sku": sku, "location": location})
        if inv_doc:
            new_on_hand  = inv_doc.get("on_hand", 0) + qty
            new_on_order = max(0, inv_doc.get("on_order", 0) - qty)
            inventory.update_one(
                {"sku": sku, "location": location},
                {"$set": {"on_hand": new_on_hand, "on_order": new_on_order}},
            )
        else:
            new_on_hand = qty

        # Mark order as received
        orders_collection.update_one(
            {"_id": oid},
            {"$set": {
                "status":       "received",
                "delivered_at": now,
                "on_time":      on_time,
            }},
        )

        # Update order_history outcome
        db.order_history.update_one(
            {"proposed_order_id": oid},
            {"$set": {"outcome": outcome}},
        )

        # Update confidence_outcomes
        alert_id = order.get("alert_id")
        if alert_id:
            days_to_recovery = round(elapsed_secs / 86400, 1)
            db.confidence_outcomes.update_one(
                {"alert_id": alert_id, "outcome": {"$ne": "resolved"}},
                {"$set": {
                    "days_to_stock_recovery": days_to_recovery,
                    "outcome":                "resolved",
                }},
            )

        # Append lifecycle event
        if alert_id:
            append_lifecycle_event(
                str(alert_id),
                "order_delivered",
                {
                    "quantity":    qty,
                    "on_time":     on_time,
                    "delay_days":  delay_days,
                    "new_on_hand": new_on_hand,
                },
                sku,
                location,
            )

        log.info(
            "order delivered",
            extra={
                "sku": sku, "location": location,
                "qty": qty, "on_time": on_time,
                "delay_days": delay_days,
            },
        )


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _wait_for_inventory(timeout: int = 180) -> None:
    """
    Block until the inventory collection has data (seeder has completed),
    printing dots every 5 s so the log shows progress.
    Exits the process if inventory is still empty after `timeout` seconds.
    """
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        if inventory.count_documents({}) > 0:
            if not first:
                log.info("inventory ready")
            return
        if first:
            log.info("waiting for inventory to be seeded")
            first = False
        time.sleep(5)

    log.error("inventory still empty, exiting", extra={"timeout_s": timeout})
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    log.info("stream simulator started")
    _set_state("running")

    _wait_for_inventory()

    while True:
        try:
            state = _get_state()

            if state in ("paused", "stopped"):
                time.sleep(2)
                continue

            all_skus = list(inventory.find({}))
            if not all_skus:
                time.sleep(10)
                continue

            ctrl        = _get_control()
            speed       = max(1, int(ctrl.get("speed", 1)))
            sku_doc     = random.choice(all_skus)
            consumption = random.randint(15, 60) * speed
            check_and_alert(sku_doc, consumption)
            deliver_pending_orders()

        except KeyboardInterrupt:
            log.info("simulator stopped by interrupt")
            _set_state("stopped")
            break
        except Exception as exc:  # noqa: BLE001
            log.error("simulator loop error", extra={"error": str(exc)})

        time.sleep(10)


if __name__ == "__main__":
    run()
