"""
Procedural Memory Engine — derives structured preference rules from repeated patterns.

Scans proposed_orders for high-frequency supplier approval patterns and writes
candidate rules to the `procedures` collection with human_confirmed=False.
Rules must be confirmed by a human via the Streamlit admin panel before the agent
uses them.

Run manually or trigger from the dashboard "Extract Rules" button:
    python agent/procedure_extractor.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from agent.db import db_sync as _db

load_dotenv()

# A rule candidate is promoted when the same supplier is approved (by agent or human)
# at least this many times for the same category+location within the look-back window.
_MIN_APPROVALS  = 5
_LOOKBACK_DAYS  = 30


def extract_procedures() -> list[dict]:
    """Scan proposed_orders for high-frequency approval patterns.

    Returns a list of newly inserted candidate procedure dicts.
    """
    since = datetime.now(timezone.utc) - timedelta(days=_LOOKBACK_DAYS)

    # Pull all approved/received orders in the window from the permanent collection.
    # short_term_memory has a 24h TTL so a 30-day lookback there is always empty.
    approved_decisions = list(_db.proposed_orders.find(
        {
            "created_at": {"$gte": since},
            "status":     {"$in": ["approved", "received"]},
        },
        {"_id": 0, "sku": 1, "location": 1, "supplier_id": 1, "supplier_name": 1},
    ))

    if not approved_decisions:
        print("[procedure_extractor] No approved decisions in look-back window.")
        return []

    # Enrich with category from inventory so we can group by category+location.
    sku_category: dict[str, str] = {}
    for dec in approved_decisions:
        sku = dec.get("sku", "")
        if sku and sku not in sku_category:
            inv = _db.inventory.find_one({"sku": sku}, {"category": 1, "_id": 0})
            sku_category[sku] = inv.get("category", "unknown") if inv else "unknown"

    # Count approvals per (category, location, supplier_id).
    counts: dict[tuple, dict] = {}
    for dec in approved_decisions:
        sku        = dec.get("sku", "")
        location   = dec.get("location", "")
        supplier_id   = dec.get("supplier_id", "")
        supplier_name = dec.get("supplier_name", "")
        category   = sku_category.get(sku, "unknown")

        if not (location and supplier_id and category):
            continue

        key = (category, location, supplier_id)
        if key not in counts:
            counts[key] = {
                "sku_category":        category,
                "location":            location,
                "preferred_supplier_id":   supplier_id,
                "preferred_supplier_name": supplier_name,
                "evidence_count":      0,
            }
        counts[key]["evidence_count"] += 1

    # Promote patterns that meet the threshold and aren't already in procedures.
    new_rules: list[dict] = []
    for key, data in counts.items():
        if data["evidence_count"] < _MIN_APPROVALS:
            continue

        # Skip if an equivalent confirmed or pending rule already exists.
        existing = _db.procedures.find_one({
            "sku_category":          data["sku_category"],
            "location":              data["location"],
            "preferred_supplier_id": data["preferred_supplier_id"],
        })
        if existing:
            # Update evidence count if the rule already exists.
            _db.procedures.update_one(
                {"_id": existing["_id"]},
                {"$set": {
                    "evidence_count": data["evidence_count"],
                    "last_updated":   datetime.now(timezone.utc),
                }},
            )
            print(
                f"[procedure_extractor] Updated existing rule: "
                f"{data['sku_category']} @ {data['location']} → "
                f"{data['preferred_supplier_name']} "
                f"(evidence={data['evidence_count']})"
            )
            continue

        rule = {
            **data,
            "condition":      f"category={data['sku_category']} AND location={data['location']}",
            "human_confirmed": False,
            "created_at":      datetime.now(timezone.utc),
            "last_updated":    datetime.now(timezone.utc),
        }
        _db.procedures.insert_one(rule)
        new_rules.append(rule)
        print(
            f"[procedure_extractor] New candidate rule: "
            f"{data['sku_category']} @ {data['location']} → "
            f"{data['preferred_supplier_name']} "
            f"(evidence={data['evidence_count']})"
        )

    print(f"[procedure_extractor] Done — {len(new_rules)} new rule(s) created.")
    return new_rules


if __name__ == "__main__":
    rules = extract_procedures()
    if not rules:
        print("No new rules extracted.")
