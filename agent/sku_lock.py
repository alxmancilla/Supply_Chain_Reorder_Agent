"""Distributed SKU-level processing lock backed by MongoDB.

Each (sku, location) pair can be held by exactly one alert at a time.
A unique compound index enforces exclusivity; a TTL index auto-expires
stale locks so a crashed worker never blocks the queue permanently.
"""
from datetime import datetime, timezone, timedelta

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError

LOCK_TTL_SECONDS = 300  # 5-minute safety net for crashed workers


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create required indexes. Idempotent — safe to call on every startup."""
    await db.sku_processing_locks.create_index(
        [("sku", ASCENDING), ("location", ASCENDING)],
        unique=True,
        name="sku_location_unique",
    )
    # MongoDB TTL monitor checks every 60 s; this is only the crash-recovery
    # safety net — normal release happens via release() below.
    await db.sku_processing_locks.create_index(
        "expires_at",
        expireAfterSeconds=0,
        name="sku_lock_ttl",
    )


async def acquire(
    db: AsyncIOMotorDatabase,
    sku: str,
    location: str,
    held_by: str,
    ttl_seconds: int = LOCK_TTL_SECONDS,
) -> bool:
    """Try to acquire the lock for (sku, location).

    Returns True if acquired. Returns False if another alert holds the lock.
    Automatically removes expired (stale) locks before giving up.
    """
    now        = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=ttl_seconds)
    doc = {
        "sku":        sku,
        "location":   location,
        "held_by":    held_by,
        "locked_at":  now,
        "expires_at": expires_at,
    }
    try:
        await db.sku_processing_locks.insert_one(doc)
        return True
    except DuplicateKeyError:
        pass

    # Remove an expired lock and retry once — handles crashed workers.
    removed = await db.sku_processing_locks.delete_one({
        "sku":        sku,
        "location":   location,
        "expires_at": {"$lt": now},
    })
    if removed.deleted_count == 0:
        return False  # live lock held by another alert

    try:
        await db.sku_processing_locks.insert_one(doc)
        return True
    except DuplicateKeyError:
        return False


async def release(
    db: AsyncIOMotorDatabase,
    sku: str,
    location: str,
    held_by: str,
) -> None:
    """Release the lock. Only removes the document if held_by matches,
    so a late release never evicts a lock acquired by a different alert."""
    await db.sku_processing_locks.delete_one({
        "sku":      sku,
        "location": location,
        "held_by":  held_by,
    })
