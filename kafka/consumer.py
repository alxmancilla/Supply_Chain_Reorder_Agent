"""
Kafka consumer — reads consumption events from 'wms-inventory-events' and
writes reorder alerts to MongoDB.  This is the production-mode replacement
for stream_simulator.py when running with a real (or containerised) Kafka
broker.

Expected event schema
─────────────────────
{
  "sku":       "MED-3017",
  "location":  "DC-Texas",
  "quantity":  45,
  "reason":    "pharmacy_dispensing",   # optional
  "timestamp": "2025-04-14T14:30:00Z"  # optional, defaults to now
}

Run locally (Kafka on localhost:9094):
    python kafka/consumer.py

Run via Docker Compose (kafka profile):
    docker compose --profile kafka up --build
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

from kafka import KafkaConsumer
from pymongo import MongoClient

# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------
_mongo = MongoClient(
    os.environ["MONGODB_URI"],
    serverSelectionTimeoutMS=5_000,
    connectTimeoutMS=10_000,
    socketTimeoutMS=30_000,
)
_db    = _mongo["supply_chain_demo"]

inventory           = _db["inventory"]
consumption_history = _db["consumption_history"]
alerts_collection   = _db["reorder_alerts"]

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9094")
TOPIC        = "wms-inventory-events"
GROUP_ID     = "reorder-alert-group"

# ---------------------------------------------------------------------------
# Helpers  (mirror of stream_simulator.py logic)
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


def record_consumption(sku: str, location: str, quantity: int,
                        timestamp: datetime, reason: str) -> None:
    consumption_history.insert_one({
        "timestamp": timestamp,
        "sku":       sku,
        "location":  location,
        "quantity":  quantity,
        "reason":    reason,
        "source":    "kafka",
    })


def check_and_alert(sku: str, location: str, quantity: int,
                    timestamp: datetime, reason: str) -> None:
    """Apply the consumption event and emit a reorder alert if needed."""
    inv = inventory.find_one({"sku": sku, "location": location})
    if not inv:
        print(f"  [WARN] No inventory record for {sku} @ {location} — skipping")
        return

    on_hand       = max(0, inv["on_hand"] - quantity)
    on_order      = inv.get("on_order", 0)
    reorder_point = inv["reorder_point"]

    inventory.update_one(
        {"sku": sku, "location": location},
        {"$set": {"on_hand": on_hand}},
    )
    record_consumption(sku, location, quantity, timestamp, reason)

    effective_stock = on_hand + on_order
    if effective_stock >= reorder_point:
        print(f"[OK]    {sku} @ {location} — on_hand={on_hand} (above reorder point)")
        return

    avg_daily      = get_avg_daily_consumption(sku)
    days_remaining = round(on_hand / avg_daily, 1) if avg_daily > 0 else 0.0

    existing = alerts_collection.find_one(
        {"sku": sku, "location": location, "status": "pending"}
    )
    if existing:
        print(f"[SKIP]  {sku} — alert already pending ({on_hand} on hand)")
        return

    alerts_collection.insert_one({
        "sku":                     sku,
        "location":                location,
        "on_hand":                 on_hand,
        "on_order":                on_order,
        "reorder_point":           reorder_point,
        "avg_daily_consumption":   avg_daily,
        "days_of_stock_remaining": days_remaining,
        "status":                  "pending",
        "source":                  "kafka",
        "created_at":              datetime.now(timezone.utc),
    })
    print(f"[ALERT] {sku} @ {location} — {on_hand} units, {days_remaining}d of stock (Kafka)")

# ---------------------------------------------------------------------------
# Consumer loop
# ---------------------------------------------------------------------------

def _connect(retries: int = 30, delay: int = 5) -> KafkaConsumer:
    for attempt in range(1, retries + 1):
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                group_id=GROUP_ID,
                auto_offset_reset="latest",
                enable_auto_commit=True,
            )
            print(f"Connected to Kafka at {KAFKA_BROKER}. Listening on '{TOPIC}' …\n")
            return consumer
        except Exception as exc:
            print(f"  [{attempt}/{retries}] Waiting for Kafka: {exc}")
            time.sleep(delay)
    print("ERROR: Could not connect to Kafka. Exiting.")
    sys.exit(1)


def run() -> None:
    consumer = _connect()
    for message in consumer:
        event = message.value
        try:
            raw_ts    = event.get("timestamp") or datetime.now(timezone.utc).isoformat()
            timestamp = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            check_and_alert(
                sku=event["sku"],
                location=event.get("location", "DC-Unknown"),
                quantity=int(event.get("quantity", 0)),
                timestamp=timestamp,
                reason=event.get("reason", "wms_event"),
            )
        except Exception as exc:
            print(f"[ERROR] Failed to process event {event}: {exc}")


if __name__ == "__main__":
    run()
