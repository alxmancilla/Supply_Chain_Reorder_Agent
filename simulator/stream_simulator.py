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

load_dotenv()

client = MongoClient(
    os.environ["MONGODB_URI"],
    serverSelectionTimeoutMS=5_000,
    connectTimeoutMS=10_000,
    socketTimeoutMS=30_000,
)
db = client["supply_chain_demo"]

inventory           = db["inventory"]
consumption_history = db["consumption_history"]
alerts_collection   = db["reorder_alerts"]
control_collection  = db["simulator_control"]

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
            print(
                f"[SKIP]  {sku} — alert already pending "
                f"(on_hand={on_hand}, {days_remaining}d remaining)"
            )
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
            "created_at": datetime.now(timezone.utc),
        }
        alerts_collection.insert_one(alert)
        print(
            f"[ALERT] {sku} @ {sku_doc['location']} — "
            f"{on_hand} units remaining, {days_remaining}d of stock"
        )
    else:
        print(
            f"[OK]    {sku} @ {sku_doc['location']} — "
            f"consumed {consumption}, on_hand={on_hand} (above reorder point {reorder_point})"
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
                print(" ready.", flush=True)
            return
        if first:
            print("[WAIT] Inventory not yet seeded — waiting for seeder to complete ",
                  end="", flush=True)
            first = False
        else:
            print(".", end="", flush=True)
        time.sleep(5)

    print(
        f"\n[ERROR] Inventory still empty after {timeout}s. "
        "Ensure data/seed.py completed successfully.",
        flush=True,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    print("Stream simulator started. Setting state to 'running'.\n")
    _set_state("running")

    _wait_for_inventory()

    while True:
        try:
            state = _get_state()

            if state in ("paused", "stopped"):
                label = "PAUSED" if state == "paused" else "STOPPED"
                print(f"[{label}] Simulator halted via control panel. Waiting…", flush=True)
                time.sleep(2)
                continue

            # state == "running" — emit one event then sleep
            all_skus = list(inventory.find({}))
            if not all_skus:
                # Inventory disappeared mid-run (e.g. seeder re-ran); wait quietly.
                time.sleep(10)
                continue

            ctrl        = _get_control()
            speed       = max(1, int(ctrl.get("speed", 1)))
            sku_doc     = random.choice(all_skus)
            consumption = random.randint(15, 60) * speed
            check_and_alert(sku_doc, consumption)

        except KeyboardInterrupt:
            print("\nSimulator stopped.")
            _set_state("stopped")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {exc}")

        time.sleep(10)


if __name__ == "__main__":
    run()
