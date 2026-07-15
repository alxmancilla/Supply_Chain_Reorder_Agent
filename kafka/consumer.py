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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.logger import get_logger
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

_ACTIVE_ALERT_STATUSES = ["pending", "processing", "awaiting_human_approval", "human_review"]

log = get_logger(__name__)

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
        log.warning("no inventory record found", extra={"sku": sku, "location": location})
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
        log.info("stock OK", extra={"sku": sku, "location": location, "on_hand": on_hand})
        return

    avg_daily      = get_avg_daily_consumption(sku)
    days_remaining = round(on_hand / avg_daily, 1) if avg_daily > 0 else 0.0

    existing = alerts_collection.find_one(
        {"sku": sku, "location": location, "status": {"$in": _ACTIVE_ALERT_STATUSES}}
    )
    if existing:
        log.info("active alert already exists, skipping", extra={
            "sku": sku, "location": location, "on_hand": on_hand,
        })
        return

    alerts_collection.insert_one({
        "sku":                        sku,
        "location":                   location,
        "on_hand":                    on_hand,
        "on_order":                   on_order,
        "reorder_point":              reorder_point,
        "units_consumed_last_15min":  quantity,
        "avg_daily_consumption":      avg_daily,
        "days_of_stock_remaining":    days_remaining,
        "status":                     "pending",
        "source":                     "kafka",
        "created_at":                 datetime.now(timezone.utc),
    })
    log.warning("reorder alert created via Kafka", extra={
        "sku": sku, "location": location,
        "on_hand": on_hand, "days_remaining": days_remaining,
    })

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
            log.info("connected to Kafka", extra={"broker": KAFKA_BROKER, "topic": TOPIC})
            return consumer
        except Exception as exc:
            log.warning("waiting for Kafka", extra={
                "attempt": attempt, "max_retries": retries, "error": str(exc),
            })
            time.sleep(delay)
    log.error("could not connect to Kafka, exiting")
    sys.exit(1)


def run() -> None:
    from agent.schemas import validate_kafka_event, write_dead_letter

    consumer = _connect()
    for message in consumer:
        event = message.value
        try:
            # Validate schema before processing
            validated, errors = validate_kafka_event(event)
            if errors:
                log.warning("invalid Kafka event, dead-lettering", extra={"errors": errors})
                write_dead_letter(_db, "kafka_consumer", "kafka_message", event, errors)
                continue

            raw_ts    = event.get("timestamp") or datetime.now(timezone.utc).isoformat()
            timestamp = datetime.fromisoformat(raw_ts.replace("Z", "+00:00")) \
                if isinstance(raw_ts, str) else raw_ts
            check_and_alert(
                sku=validated.sku,
                location=validated.location,
                quantity=validated.quantity,
                timestamp=timestamp,
                reason=validated.reason,
            )
        except Exception as exc:
            log.error("failed to process Kafka event", extra={"error": str(exc)})


if __name__ == "__main__":
    run()
