"""
Memory retry worker — re-attempts failed memory writes from failed_memory_writes.

Polls the collection for unresolved entries (resolved=False) and retries each up to
MAX_RETRIES times with a fixed RETRY_DELAY_SECONDS gap between runs.  On success the
document is marked resolved=True.  After MAX_RETRIES failed attempts the document is
left as-is with retry_count=MAX_RETRIES so operators can inspect and act on it.

Run as a background process alongside the agent:
    python agent/memory_retry_worker.py

Or import and call run_once() from a scheduled job or admin panel.
"""

import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

_client = MongoClient(
    os.environ["MONGODB_URI"],
    serverSelectionTimeoutMS=5_000,
    connectTimeoutMS=10_000,
    socketTimeoutMS=30_000,
)
_db = _client["supply_chain_demo"]

MAX_RETRIES         = 3
RETRY_DELAY_SECONDS = 30   # seconds between each full poll cycle
POLL_INTERVAL       = 30   # seconds between polls when queue is empty


def _retry_short_term(payload: dict) -> None:
    """Retry a failed short_term_memory write."""
    from datetime import datetime as _dt
    doc = {k: v for k, v in payload.items() if k != "decided_at"}
    decided_at_raw = payload.get("decided_at")
    if isinstance(decided_at_raw, str):
        try:
            doc["decided_at"] = _dt.fromisoformat(decided_at_raw)
        except ValueError:
            doc["decided_at"] = _dt.now(timezone.utc)
    else:
        doc["decided_at"] = _dt.now(timezone.utc)
    _db.short_term_memory.insert_one(doc)


def _retry_long_term(payload: dict) -> None:
    """Retry a failed long_term_memory (agent_memory) write — re-embeds content."""
    import voyageai
    from datetime import datetime as _dt

    voyage  = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    content = payload.get("content", "")
    if not content:
        raise ValueError("payload has no 'content' field — cannot embed")

    result    = voyage.embed([content], model="voyage-4-large", input_type="document")
    embedding = result.embeddings[0]

    created_at_raw = payload.get("created_at")
    if isinstance(created_at_raw, str):
        try:
            created_at = _dt.fromisoformat(created_at_raw)
        except ValueError:
            created_at = _dt.now(timezone.utc)
    else:
        created_at = _dt.now(timezone.utc)

    doc = {k: v for k, v in payload.items() if k != "created_at"}
    doc["embedding"]  = embedding
    doc["created_at"] = created_at
    _db.agent_memory.insert_one(doc)


def run_once() -> dict:
    """Process all unresolved failed memory writes. Returns {"retried": n, "resolved": n}."""
    pending = list(_db.failed_memory_writes.find(
        {"resolved": False, "retry_count": {"$lt": MAX_RETRIES}},
    ))

    retried  = 0
    resolved = 0

    for doc in pending:
        doc_id      = doc["_id"]
        memory_type = doc.get("memory_type", "")
        sku         = doc.get("sku", "?")
        location    = doc.get("location", "?")
        payload     = doc.get("payload", {})
        retry_count = doc.get("retry_count", 0)

        try:
            if memory_type == "short_term":
                _retry_short_term(payload)
            elif memory_type == "long_term":
                _retry_long_term(payload)
            else:
                raise ValueError(f"Unknown memory_type: {memory_type!r}")

            _db.failed_memory_writes.update_one(
                {"_id": doc_id},
                {"$set": {
                    "resolved":     True,
                    "last_attempt": datetime.now(timezone.utc),
                    "retry_count":  retry_count + 1,
                }},
            )
            print(f"  [retry_worker] Resolved {memory_type} memory for {sku} @ {location}")
            resolved += 1

        except Exception as exc:
            new_count = retry_count + 1
            _db.failed_memory_writes.update_one(
                {"_id": doc_id},
                {"$set": {
                    "retry_count":  new_count,
                    "last_attempt": datetime.now(timezone.utc),
                    "error":        str(exc),
                }},
            )
            print(
                f"  [retry_worker] Retry {new_count}/{MAX_RETRIES} failed for "
                f"{memory_type} memory ({sku} @ {location}): {exc}"
            )

        retried += 1

    return {"retried": retried, "resolved": resolved}


def run() -> None:
    """Poll failed_memory_writes in a loop until interrupted."""
    print("Memory retry worker started. Polling every", RETRY_DELAY_SECONDS, "s …")
    while True:
        try:
            stats = run_once()
            if stats["retried"] > 0:
                print(
                    f"[retry_worker] Cycle complete — "
                    f"retried={stats['retried']}, resolved={stats['resolved']}"
                )
        except Exception as exc:
            print(f"[retry_worker] Error during poll cycle: {exc}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
