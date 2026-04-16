"""
Kafka producer — publish a single consumption event to 'wms-inventory-events'.

Use this to manually trigger the agent pipeline when running in Kafka mode,
or to load-test the consumer without waiting for real WMS events.

Usage
─────
  # Interactive (prompts for SKU):
  python kafka/producer.py

  # Positional args:  sku  location  quantity  reason
  python kafka/producer.py MED-3017 DC-Texas 60
  python kafka/producer.py SURG-0084 DC-Ohio 30 stock_count

  # Override broker (default: localhost:9094 — the external Docker port):
  KAFKA_BROKER=1.2.3.4:9094 python kafka/producer.py MED-5502 DC-California 45

Broker address
──────────────
  localhost:9094  → Kafka running via docker-compose.kafka.yml on this machine
  <public-ip>:9094 → Kafka accessible from another host or Atlas ASP
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9094")
TOPIC        = "wms-inventory-events"

# SKUs seeded by data/seed.py  (sku → default location)
VALID_SKUS: dict[str, str] = {
    "MED-2041":  "DC-Ohio",
    "MED-3017":  "DC-Texas",
    "MED-4490":  "DC-Ohio",
    "MED-5502":  "DC-California",
    "MED-6201":  "DC-Texas",
    "SURG-0084": "DC-Ohio",
    "SURG-1122": "DC-Texas",
    "SURG-2244": "DC-California",
    "DIAG-0331": "DC-Ohio",
    "LAB-0112":  "DC-Texas",
}


def _pick_sku_interactively() -> tuple[str, str]:
    items = list(VALID_SKUS.items())
    print("\nAvailable SKUs:")
    for i, (sku, loc) in enumerate(items, 1):
        print(f"  {i:2}. {sku}  @  {loc}")
    choice = input("\nEnter number (default: 1 → MED-2041): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(items):
        return items[int(choice) - 1]
    return items[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a consumption event to wms-inventory-events"
    )
    parser.add_argument("sku",      nargs="?", default=None)
    parser.add_argument("location", nargs="?", default=None)
    parser.add_argument("quantity", nargs="?", type=int, default=50)
    parser.add_argument("reason",   nargs="?", default="manual_test")
    args = parser.parse_args()

    # Resolve SKU + location
    if not args.sku:
        args.sku, args.location = _pick_sku_interactively()
    elif not args.location:
        args.location = VALID_SKUS.get(args.sku, "DC-Ohio")

    event = {
        "sku":       args.sku,
        "location":  args.location,
        "quantity":  args.quantity,
        "reason":    args.reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\nConnecting to Kafka at {KAFKA_BROKER} …")
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=10_000,
        )
    except NoBrokersAvailable:
        print(f"ERROR: No broker reachable at {KAFKA_BROKER}.")
        print("       Is Kafka running?  docker compose -f docker-compose.kafka.yml up -d")
        sys.exit(1)

    future = producer.send(TOPIC, value=event)
    producer.flush(timeout=10)
    meta = future.get(timeout=10)
    producer.close()

    print(f"\n✅  Published to '{TOPIC}' "
          f"(partition={meta.partition}, offset={meta.offset}):")
    print(f"    {json.dumps(event, indent=4)}")


if __name__ == "__main__":
    main()
