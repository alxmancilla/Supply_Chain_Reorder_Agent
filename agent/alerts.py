"""Shared reorder-alert helpers used by seed, simulator, Kafka, and graph startup."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from agent.schemas import validate_alert, write_dead_letter


def build_reorder_alert(
    *,
    sku: str,
    location: str,
    on_hand: int,
    on_order: int,
    reorder_point: int,
    units_consumed_last_15min: int | None,
    avg_daily_consumption: float,
    days_of_stock_remaining: float,
    source: str | None = None,
    created_at: datetime | None = None,
    rejection_count: int = 0,
) -> dict[str, Any]:
    """Create a canonical pending reorder_alert document."""
    alert: dict[str, Any] = {
        "sku": sku,
        "location": location,
        "on_hand": int(on_hand),
        "on_order": int(on_order),
        "reorder_point": int(reorder_point),
        "units_consumed_last_15min": units_consumed_last_15min,
        "avg_daily_consumption": float(avg_daily_consumption),
        "days_of_stock_remaining": float(days_of_stock_remaining),
        "status": "pending",
        "rejection_count": int(rejection_count),
        "created_at": created_at or datetime.now(timezone.utc),
    }
    if source:
        alert["source"] = source
    return alert


def append_lifecycle_event(
    db,
    alert_id: ObjectId | str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    sku: str = "",
    location: str = "",
) -> None:
    """Append a timestamped lifecycle event without importing agent.tools."""
    try:
        oid = ObjectId(alert_id)
    except Exception:
        oid = alert_id

    event = {"type": event_type, "at": datetime.now(timezone.utc), **(payload or {})}
    db.alert_lifecycle.update_one(
        {"alert_id": oid},
        {
            "$push": {"events": event},
            "$setOnInsert": {
                "alert_id": oid,
                "sku": sku,
                "location": location,
            },
        },
        upsert=True,
    )


def validate_alert_document(db, alert: dict[str, Any], source: str) -> list[str]:
    """Validate an alert and dead-letter failures. Returns validation errors."""
    _, errors = validate_alert(alert)
    if errors:
        write_dead_letter(db, source, "reorder_alert", alert, errors)
    return errors


def insert_reorder_alert(
    db,
    alert: dict[str, Any],
    *,
    source: str,
    append_lifecycle: bool = True,
) -> ObjectId | None:
    """Validate and insert a reorder alert, returning the inserted ObjectId."""
    errors = validate_alert_document(db, alert, source)
    if errors:
        return None

    result = db.reorder_alerts.insert_one(alert)
    alert_id = result.inserted_id

    if append_lifecycle:
        append_lifecycle_event(
            db,
            alert_id,
            "alert_created",
            {
                "on_hand": alert.get("on_hand"),
                "days_remaining": alert.get("days_of_stock_remaining"),
                "avg_daily": alert.get("avg_daily_consumption"),
                "reorder_point": alert.get("reorder_point"),
                "source": alert.get("source", source),
            },
            alert.get("sku", ""),
            alert.get("location", ""),
        )

    return alert_id
