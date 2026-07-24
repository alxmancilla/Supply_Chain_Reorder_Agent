"""Offline tests for shared reorder-alert helpers."""

from bson import ObjectId

from agent.alerts import build_reorder_alert, insert_reorder_alert
from agent.schemas import validate_alert


class _InsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class _FakeCollection:
    def __init__(self):
        self.inserted = []
        self.updates = []

    def insert_one(self, doc):
        doc.setdefault("_id", ObjectId())
        self.inserted.append(doc)
        return _InsertResult(doc["_id"])

    def update_one(self, *args, **kwargs):
        self.updates.append((args, kwargs))


class _FakeDB:
    def __init__(self):
        self.reorder_alerts = _FakeCollection()
        self.alert_lifecycle = _FakeCollection()
        self.dead_letter_events = _FakeCollection()


def test_build_reorder_alert_matches_schema():
    alert = build_reorder_alert(
        sku="MED-3017",
        location="DC-Texas",
        on_hand=45,
        on_order=0,
        reorder_point=200,
        units_consumed_last_15min=12,
        avg_daily_consumption=25.0,
        days_of_stock_remaining=1.8,
        source="test",
    )

    parsed, errors = validate_alert(alert)

    assert errors == []
    assert parsed is not None
    assert alert["status"] == "pending"
    assert alert["rejection_count"] == 0
    assert alert["source"] == "test"


def test_insert_reorder_alert_writes_lifecycle_event():
    db = _FakeDB()
    alert = build_reorder_alert(
        sku="SURG-0084",
        location="DC-Ohio",
        on_hand=55,
        on_order=0,
        reorder_point=100,
        units_consumed_last_15min=10,
        avg_daily_consumption=18.0,
        days_of_stock_remaining=3.1,
        source="test",
    )

    inserted_id = insert_reorder_alert(db, alert, source="unit_test")

    assert inserted_id is not None
    assert len(db.reorder_alerts.inserted) == 1
    assert len(db.alert_lifecycle.updates) == 1
    update_args, update_kwargs = db.alert_lifecycle.updates[0]
    assert update_args[1]["$push"]["events"]["type"] == "alert_created"
    assert update_kwargs["upsert"] is True


def test_invalid_alert_is_dead_lettered_not_inserted():
    db = _FakeDB()
    bad_alert = {
        "sku": "",
        "location": "DC-Ohio",
        "on_hand": -1,
        "on_order": 0,
        "reorder_point": 100,
        "days_of_stock_remaining": 0,
        "status": "pending",
    }

    inserted_id = insert_reorder_alert(db, bad_alert, source="unit_test")

    assert inserted_id is None
    assert db.reorder_alerts.inserted == []
    assert len(db.dead_letter_events.inserted) == 1
