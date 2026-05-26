"""Tests for agent.sku_lock — distributed SKU-level processing lock.

Three layers:
  1. Pure unit tests  — mock the async DB, no network required
  2. DB-backed tests  — real MongoDB (skipped when Atlas is unreachable)
  3. Concurrency test — proves exclusive locking under asyncio.gather
"""

import asyncio
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.errors import DuplicateKeyError

from agent import sku_lock
from tests.conftest import requires_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine from a sync test."""
    return asyncio.run(coro)


def _mock_db():
    """Return (db_mock, collection_mock) with an AsyncMock collection."""
    coll = MagicMock()
    db   = MagicMock()
    db.sku_processing_locks = coll
    return db, coll


# ---------------------------------------------------------------------------
# 1. Unit tests — mocked async DB
# ---------------------------------------------------------------------------

class TestAcquire:
    def test_returns_true_on_first_insert(self):
        db, coll = _mock_db()
        coll.insert_one = AsyncMock(return_value=MagicMock())

        result = _run(sku_lock.acquire(db, "MED-3017", "DC-Texas", "alert-A"))

        assert result is True
        coll.insert_one.assert_awaited_once()

    def test_inserts_required_fields(self):
        db, coll = _mock_db()
        coll.insert_one = AsyncMock(return_value=MagicMock())

        _run(sku_lock.acquire(db, "MED-3017", "DC-Texas", "alert-A"))

        doc = coll.insert_one.call_args[0][0]
        assert doc["sku"]      == "MED-3017"
        assert doc["location"] == "DC-Texas"
        assert doc["held_by"]  == "alert-A"
        assert "locked_at"  in doc
        assert "expires_at" in doc

    def test_expires_at_is_in_the_future(self):
        db, coll = _mock_db()
        coll.insert_one = AsyncMock(return_value=MagicMock())

        before = datetime.now(timezone.utc)
        _run(sku_lock.acquire(db, "MED-3017", "DC-Texas", "alert-A"))
        after  = datetime.now(timezone.utc)

        doc = coll.insert_one.call_args[0][0]
        assert doc["expires_at"] > before
        assert doc["expires_at"] > after  # must be in the future

    def test_returns_false_when_live_lock_held(self):
        """DuplicateKeyError + no expired doc → return False without raising."""
        db, coll = _mock_db()
        coll.insert_one = AsyncMock(side_effect=DuplicateKeyError("", 11000))
        coll.delete_one = AsyncMock(return_value=MagicMock(deleted_count=0))

        result = _run(sku_lock.acquire(db, "MED-3017", "DC-Texas", "alert-B"))

        assert result is False

    def test_removes_expired_lock_and_returns_true(self):
        """DuplicateKeyError + expired doc deleted → retry insert succeeds."""
        db, coll = _mock_db()
        coll.insert_one = AsyncMock(
            side_effect=[DuplicateKeyError("", 11000), MagicMock()]
        )
        coll.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))

        result = _run(sku_lock.acquire(db, "MED-3017", "DC-Texas", "alert-C"))

        assert result is True
        assert coll.insert_one.call_count == 2

    def test_returns_false_if_retry_insert_also_races(self):
        """Two DuplicateKeyErrors in a row → False (another worker beat both)."""
        db, coll = _mock_db()
        coll.insert_one = AsyncMock(
            side_effect=[
                DuplicateKeyError("", 11000),
                DuplicateKeyError("", 11000),
            ]
        )
        coll.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))

        result = _run(sku_lock.acquire(db, "MED-3017", "DC-Texas", "alert-D"))

        assert result is False

    def test_expired_lock_filter_uses_lt_now(self):
        """delete_one must only target documents whose expires_at < now."""
        db, coll = _mock_db()
        coll.insert_one = AsyncMock(side_effect=DuplicateKeyError("", 11000))
        coll.delete_one = AsyncMock(return_value=MagicMock(deleted_count=0))

        before = datetime.now(timezone.utc)
        _run(sku_lock.acquire(db, "MED-3017", "DC-Texas", "alert-E"))
        after  = datetime.now(timezone.utc)

        filter_doc = coll.delete_one.call_args[0][0]
        assert filter_doc["sku"]      == "MED-3017"
        assert filter_doc["location"] == "DC-Texas"
        # The $lt value must be between before and after — i.e., "now"
        lt_val = filter_doc["expires_at"]["$lt"]
        assert before <= lt_val <= after


class TestRelease:
    def test_deletes_document_matching_held_by(self):
        db, coll = _mock_db()
        coll.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))

        _run(sku_lock.release(db, "MED-3017", "DC-Texas", "alert-A"))

        coll.delete_one.assert_awaited_once_with({
            "sku":      "MED-3017",
            "location": "DC-Texas",
            "held_by":  "alert-A",
        })

    def test_does_not_raise_when_nothing_deleted(self):
        """Releasing a lock not held by this alert is a silent no-op."""
        db, coll = _mock_db()
        coll.delete_one = AsyncMock(return_value=MagicMock(deleted_count=0))

        # Must not raise
        _run(sku_lock.release(db, "MED-3017", "DC-Texas", "wrong-alert"))

    def test_held_by_is_included_in_filter(self):
        """release() must scope the delete to the specific alert — never evict
        another holder's lock by matching only sku + location."""
        db, coll = _mock_db()
        coll.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))

        _run(sku_lock.release(db, "MED-3017", "DC-Texas", "alert-X"))

        filter_doc = coll.delete_one.call_args[0][0]
        assert "held_by" in filter_doc, "held_by must be in the delete filter"
        assert filter_doc["held_by"] == "alert-X"


# ---------------------------------------------------------------------------
# 2. DB-backed tests — real MongoDB via motor (skipped when unreachable)
# ---------------------------------------------------------------------------

@requires_db
class TestAcquireRealDB:
    """Verifies the unique-index enforcement that mocks can't prove."""

    @pytest.fixture(autouse=True)
    def motor_db(self, test_db):
        from motor.motor_asyncio import AsyncIOMotorClient
        client    = AsyncIOMotorClient(
            os.environ["MONGODB_URI"],
            serverSelectionTimeoutMS=5_000,
        )
        # Re-use the same isolated DB name that test_db already created
        self._async_db = client[test_db.name]
        yield
        client.close()

    def test_ensure_indexes_is_idempotent(self):
        """Calling ensure_indexes twice must not raise."""
        async def _run_indexes():
            await sku_lock.ensure_indexes(self._async_db)
            await sku_lock.ensure_indexes(self._async_db)  # second call — no-op

        _run(_run_indexes())  # no exception → pass

    def test_first_acquire_succeeds(self):
        async def _go():
            await sku_lock.ensure_indexes(self._async_db)
            return await sku_lock.acquire(
                self._async_db, "MED-3017", "DC-Texas", "alert-real-1"
            )

        assert _run(_go()) is True

    def test_second_acquire_for_same_sku_fails(self):
        async def _go():
            await sku_lock.ensure_indexes(self._async_db)
            await sku_lock.acquire(
                self._async_db, "MED-3017", "DC-Texas", "alert-real-1"
            )
            return await sku_lock.acquire(
                self._async_db, "MED-3017", "DC-Texas", "alert-real-2"
            )

        assert _run(_go()) is False

    def test_different_sku_acquires_independently(self):
        async def _go():
            await sku_lock.ensure_indexes(self._async_db)
            r1 = await sku_lock.acquire(
                self._async_db, "MED-3017", "DC-Texas", "alert-real-1"
            )
            r2 = await sku_lock.acquire(
                self._async_db, "MED-9999", "DC-Texas", "alert-real-2"  # different SKU
            )
            return r1, r2

        r1, r2 = _run(_go())
        assert r1 is True
        assert r2 is True  # different SKU — no contention

    def test_release_then_reacquire_succeeds(self):
        async def _go():
            await sku_lock.ensure_indexes(self._async_db)
            await sku_lock.acquire(
                self._async_db, "MED-3017", "DC-Texas", "alert-real-1"
            )
            await sku_lock.release(
                self._async_db, "MED-3017", "DC-Texas", "alert-real-1"
            )
            return await sku_lock.acquire(
                self._async_db, "MED-3017", "DC-Texas", "alert-real-2"
            )

        assert _run(_go()) is True

    def test_release_wrong_held_by_leaves_lock_intact(self):
        """A late release from a crashed worker must not evict the new holder."""
        async def _go():
            await sku_lock.ensure_indexes(self._async_db)
            # alert-1 acquires, alert-2 releases with wrong held_by
            await sku_lock.acquire(
                self._async_db, "MED-3017", "DC-Texas", "alert-real-1"
            )
            await sku_lock.release(
                self._async_db, "MED-3017", "DC-Texas", "alert-real-WRONG"
            )
            # Lock should still be held by alert-1 → alert-3 cannot acquire
            return await sku_lock.acquire(
                self._async_db, "MED-3017", "DC-Texas", "alert-real-3"
            )

        assert _run(_go()) is False


# ---------------------------------------------------------------------------
# 3. Concurrency proof — asyncio.gather with real MongoDB
# ---------------------------------------------------------------------------

@requires_db
class TestConcurrentAcquire:
    """Proves mutual exclusion under real concurrent asyncio tasks.

    asyncio.gather schedules both coroutines on the same event loop,
    interleaving them at every await point. MongoDB's unique index is the
    arbiter — exactly one insert can win for a given (sku, location) pair.
    """

    @pytest.fixture(autouse=True)
    def motor_db(self, test_db):
        from motor.motor_asyncio import AsyncIOMotorClient
        client         = AsyncIOMotorClient(
            os.environ["MONGODB_URI"],
            serverSelectionTimeoutMS=5_000,
        )
        self._async_db = client[test_db.name]
        self._sync_db  = test_db
        yield
        client.close()

    def test_exactly_one_winner_among_two_concurrent_acquires(self):
        async def _go():
            await sku_lock.ensure_indexes(self._async_db)
            results = await asyncio.gather(
                sku_lock.acquire(self._async_db, "MED-3017", "DC-Texas", "alert-A"),
                sku_lock.acquire(self._async_db, "MED-3017", "DC-Texas", "alert-B"),
                return_exceptions=True,
            )
            return results

        results = _run(_go())
        assert not any(isinstance(r, Exception) for r in results), \
            f"Unexpected exception in concurrent acquire: {results}"
        winners = [r for r in results if r is True]
        assert len(winners) == 1, (
            f"Expected exactly 1 winner, got {winners} from results {results}"
        )

    def test_exactly_one_winner_among_five_concurrent_acquires(self):
        """Stress case: 5 tasks race for the same lock — exactly 1 wins."""
        alert_ids = [f"alert-{i}" for i in range(5)]

        async def _go():
            await sku_lock.ensure_indexes(self._async_db)
            results = await asyncio.gather(
                *[
                    sku_lock.acquire(self._async_db, "MED-3017", "DC-Texas", aid)
                    for aid in alert_ids
                ],
                return_exceptions=True,
            )
            return results

        results = _run(_go())
        assert not any(isinstance(r, Exception) for r in results)
        winners = [r for r in results if r is True]
        assert len(winners) == 1, (
            f"Expected exactly 1 winner across 5 concurrent acquires, got {winners}"
        )

    def test_lock_in_db_matches_winner(self):
        """The lock document in MongoDB must record the alert that won."""
        async def _go():
            await sku_lock.ensure_indexes(self._async_db)
            results = await asyncio.gather(
                sku_lock.acquire(self._async_db, "MED-3017", "DC-Texas", "alert-A"),
                sku_lock.acquire(self._async_db, "MED-3017", "DC-Texas", "alert-B"),
                return_exceptions=True,
            )
            winner_id = "alert-A" if results[0] is True else "alert-B"
            return winner_id

        winner_id = _run(_go())

        lock_doc = self._sync_db.sku_processing_locks.find_one({
            "sku":      "MED-3017",
            "location": "DC-Texas",
        })
        assert lock_doc is not None, "Lock document must exist in DB"
        assert lock_doc["held_by"] == winner_id, (
            f"DB shows held_by={lock_doc['held_by']} but winner was {winner_id}"
        )
