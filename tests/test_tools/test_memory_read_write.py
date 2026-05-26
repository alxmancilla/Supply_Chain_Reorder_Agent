"""Tests for short/long-term memory read + write round-trips (with DB fixture)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from tests.conftest import requires_db
from agent.tools import (
    get_short_term_memories,
    write_short_term_memory_sync,
)


pytestmark = requires_db


@pytest.fixture(autouse=True)
def patch_db(test_db):
    """Redirect tools._db to the test database for all tests in this module."""
    with patch("agent.tools._db", test_db):
        yield


class TestShortTermMemoryWrite:
    def test_write_inserts_document(self, test_db):
        rec = {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      200,
            "confidence":    "high",
            "rationale":     "Low stock, standard reorder.",
        }
        write_short_term_memory_sync("MED-3017", "DC-Texas", rec, 3.5, True)
        doc = test_db.short_term_memory.find_one({"sku": "MED-3017", "location": "DC-Texas"})
        assert doc is not None
        assert doc["supplier_name"] == "MedSupply Co"
        assert doc["quantity"] == 200
        assert doc["auto_approved"] is True

    def test_write_stores_rationale_truncated(self, test_db):
        long_rationale = "x" * 400
        rec = {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      100,
            "confidence":    "medium",
            "rationale":     long_rationale,
        }
        write_short_term_memory_sync("MED-3017", "DC-Texas", rec, 2.0, False)
        doc = test_db.short_term_memory.find_one({"sku": "MED-3017"})
        assert len(doc["rationale_summary"]) <= 300


class TestShortTermMemoryRead:
    def test_read_returns_recent_entries(self, test_db):
        test_db.short_term_memory.insert_many([
            {
                "sku": "MED-3017", "location": "DC-Texas",
                "decided_at": datetime.now(timezone.utc),
                "supplier_name": "MedSupply Co", "quantity": 100,
            },
            {
                "sku": "MED-3017", "location": "DC-Texas",
                "decided_at": datetime.now(timezone.utc) - timedelta(hours=1),
                "supplier_name": "QuickMed LLC", "quantity": 50,
            },
        ])
        results = get_short_term_memories("MED-3017", "DC-Texas", hours=24)
        assert len(results) == 2
        assert results[0]["supplier_name"] == "MedSupply Co"

    def test_read_excludes_old_entries(self, test_db):
        test_db.short_term_memory.insert_one({
            "sku": "MED-3017", "location": "DC-Texas",
            "decided_at": datetime.now(timezone.utc) - timedelta(hours=48),
            "supplier_name": "OldSupplier", "quantity": 200,
        })
        results = get_short_term_memories("MED-3017", "DC-Texas", hours=24)
        assert results == []

    def test_read_returns_empty_for_unknown_sku(self, test_db):
        results = get_short_term_memories("UNKNOWN-SKU", "DC-Texas", hours=24)
        assert results == []

    def test_read_filters_by_location(self, test_db):
        test_db.short_term_memory.insert_many([
            {
                "sku": "MED-3017", "location": "DC-Texas",
                "decided_at": datetime.now(timezone.utc), "supplier_name": "A",
            },
            {
                "sku": "MED-3017", "location": "DC-West",
                "decided_at": datetime.now(timezone.utc), "supplier_name": "B",
            },
        ])
        results = get_short_term_memories("MED-3017", "DC-Texas", hours=24)
        assert all(r["location"] == "DC-Texas" for r in results)


class TestFailedMemoryWriteRecorded:
    def test_write_records_failure_on_db_error(self, test_db):
        """If the insert fails, a record should appear in failed_memory_writes."""
        from unittest.mock import MagicMock, patch

        bad_collection = MagicMock()
        bad_collection.insert_one.side_effect = RuntimeError("simulated DB failure")

        with patch.object(test_db, "short_term_memory", bad_collection):
            rec = {
                "supplier_id":   "SUP-001",
                "supplier_name": "MedSupply Co",
                "quantity":      100,
                "confidence":    "high",
                "rationale":     "Test rationale for error path.",
            }
            write_short_term_memory_sync("MED-3017", "DC-Texas", rec, 3.0, False)

        fail_doc = test_db.failed_memory_writes.find_one({"sku": "MED-3017"})
        assert fail_doc is not None
        assert fail_doc["memory_type"] == "short_term"
        assert fail_doc["resolved"] is False
        assert "simulated DB failure" in fail_doc["error"]
