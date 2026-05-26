"""
Pydantic schemas for all system boundary inputs.

Used to validate data at entry points before it reaches the agent graph:
  • reorder_alerts arriving via Change Stream or direct insert
  • inventory events from the Kafka consumer
  • recommendation output from the LLM (via validate_recommendation in tools.py)

Validation failures are written to dead_letter_events and logged as errors.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Reorder alert — incoming from simulator, Kafka consumer, or Change Stream
# ---------------------------------------------------------------------------

class ReorderAlertSchema(BaseModel):
    sku:                      str
    location:                 str
    on_hand:                  int   = Field(ge=0)
    on_order:                 int   = Field(ge=0)
    reorder_point:            int   = Field(ge=0)
    units_consumed_last_15min: Optional[int]   = Field(default=None, ge=0)
    avg_daily_consumption:    Optional[float]  = Field(default=None, ge=0)
    days_of_stock_remaining:  float = Field(ge=0)
    status:                   Literal["pending"]

    @field_validator("sku", "location")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()


# ---------------------------------------------------------------------------
# Kafka inventory event — from the WMS event bus
# ---------------------------------------------------------------------------

class KafkaInventoryEventSchema(BaseModel):
    sku:      str
    location: str
    quantity: int  = Field(ge=0)
    reason:   str  = Field(default="wms_event")
    timestamp: Optional[datetime] = None

    @field_validator("sku", "location")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @model_validator(mode="before")
    @classmethod
    def coerce_timestamp(cls, values: dict) -> dict:
        ts = values.get("timestamp")
        if isinstance(ts, str):
            try:
                values["timestamp"] = datetime.fromisoformat(ts)
            except ValueError:
                values["timestamp"] = None
        return values


# ---------------------------------------------------------------------------
# LLM recommendation output — validated by the audit agent
# ---------------------------------------------------------------------------

class RecommendationOutputSchema(BaseModel):
    supplier_id:   str
    supplier_name: str
    quantity:      int  = Field(ge=0)
    rationale:     str  = Field(min_length=10)
    confidence:    Literal["high", "medium", "low"]

    @field_validator("supplier_id", "supplier_name")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()


# ---------------------------------------------------------------------------
# Dead-letter event — written when validation fails at any boundary
# ---------------------------------------------------------------------------

class DeadLetterEvent(BaseModel):
    source:     str            # e.g. "kafka_consumer", "change_stream", "audit_agent"
    event_type: str            # e.g. "reorder_alert", "kafka_message", "recommendation"
    raw_payload: dict
    errors:     list[str]
    occurred_at: datetime


# ---------------------------------------------------------------------------
# Validation helper — used at all ingestion boundaries
# ---------------------------------------------------------------------------

def validate_alert(raw: dict) -> tuple[ReorderAlertSchema | None, list[str]]:
    """Validate a raw alert dict. Returns (schema_obj, []) on success or (None, errors)."""
    try:
        return ReorderAlertSchema(**raw), []
    except Exception as exc:
        errors = [str(e) for e in exc.errors()] if hasattr(exc, "errors") else [str(exc)]
        return None, errors


def validate_kafka_event(raw: dict) -> tuple[KafkaInventoryEventSchema | None, list[str]]:
    """Validate a raw Kafka message dict."""
    try:
        return KafkaInventoryEventSchema(**raw), []
    except Exception as exc:
        errors = [str(e) for e in exc.errors()] if hasattr(exc, "errors") else [str(exc)]
        return None, errors


def write_dead_letter(
    db,
    source: str,
    event_type: str,
    raw_payload: dict,
    errors: list[str],
) -> None:
    """Write a validation failure to the dead_letter_events collection."""
    from datetime import timezone
    try:
        db.dead_letter_events.insert_one({
            "source":      source,
            "event_type":  event_type,
            "raw_payload": raw_payload,
            "errors":      errors,
            "occurred_at": datetime.now(timezone.utc),
        })
    except Exception as exc:
        print(f"  [ERROR] Failed to write dead letter event: {exc}")
