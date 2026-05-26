"""Tests for get_consumption_trend — aggregation result parsing and trend logic."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from tests.conftest import requires_db
from agent.tools import get_consumption_trend

pytestmark = requires_db


@pytest.fixture(autouse=True)
def patch_db(test_db):
    with patch("agent.tools._db", test_db):
        yield


def _insert_consumption(db, sku, location, daily_counts: dict) -> None:
    """Insert per-day consumption records. daily_counts maps 'YYYY-MM-DD' → quantity."""
    docs = []
    for date_str, qty in daily_counts.items():
        ts = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        docs.append({
            "sku":       sku,
            "location":  location,
            "timestamp": ts,
            "quantity":  qty,
        })
    if docs:
        db.consumption_history.insert_many(docs)


class TestGetConsumptionTrend:
    def test_returns_zero_when_no_history(self, test_db):
        result = get_consumption_trend.invoke(
            {"sku": "MED-3017", "location": "DC-Texas", "days": 14}
        )
        assert result["avg_daily"] == 0
        assert result["trend"] == "unknown"
        assert result["days_sampled"] == 0

    def test_computes_avg_daily_correctly(self, test_db):
        now = datetime.now(timezone.utc)
        dates = {
            (now - timedelta(days=d)).strftime("%Y-%m-%d"): 100
            for d in range(5)
        }
        _insert_consumption(test_db, "MED-3017", "DC-Texas", dates)
        result = get_consumption_trend.invoke(
            {"sku": "MED-3017", "location": "DC-Texas", "days": 14}
        )
        assert result["avg_daily"] == 100.0
        assert result["days_sampled"] == 5

    def test_detects_increasing_trend(self, test_db):
        now = datetime.now(timezone.utc)
        # First half: low, second half: high
        dates = {}
        for d in range(6, 0, -1):
            dates[(now - timedelta(days=d)).strftime("%Y-%m-%d")] = 10
        for d in range(0, 6):
            dates[(now - timedelta(days=d)).strftime("%Y-%m-%d")] = 100
        _insert_consumption(test_db, "MED-3017", "DC-Texas", dates)
        result = get_consumption_trend.invoke(
            {"sku": "MED-3017", "location": "DC-Texas", "days": 14}
        )
        assert result["trend"] == "increasing"

    def test_detects_decreasing_trend(self, test_db):
        now = datetime.now(timezone.utc)
        dates = {}
        for d in range(6, 0, -1):
            dates[(now - timedelta(days=d)).strftime("%Y-%m-%d")] = 100
        for d in range(0, 6):
            dates[(now - timedelta(days=d)).strftime("%Y-%m-%d")] = 10
        _insert_consumption(test_db, "MED-3017", "DC-Texas", dates)
        result = get_consumption_trend.invoke(
            {"sku": "MED-3017", "location": "DC-Texas", "days": 14}
        )
        assert result["trend"] == "decreasing"

    def test_filters_by_location(self, test_db):
        now = datetime.now(timezone.utc)
        test_db.consumption_history.insert_many([
            {"sku": "MED-3017", "location": "DC-Texas", "timestamp": now, "quantity": 50},
            {"sku": "MED-3017", "location": "DC-West",  "timestamp": now, "quantity": 999},
        ])
        result = get_consumption_trend.invoke(
            {"sku": "MED-3017", "location": "DC-Texas", "days": 14}
        )
        assert result["avg_daily"] == 50.0
