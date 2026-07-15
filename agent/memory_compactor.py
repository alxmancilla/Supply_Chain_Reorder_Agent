"""
Memory Compactor — prevents agent_memory from growing unboundedly.

Two operations:
  1. Deduplication: entries for the same (sku, location) with cosine similarity
     > 0.95 are merged into a single entry. Human-decision entries are always kept.
  2. Summarisation: entries older than AGE_THRESHOLD_DAYS are condensed into one
     composite memory per (sku, location) using a simple text concatenation
     (LLM summarisation can be added later without changing the schema).

Run manually or trigger from the Streamlit admin panel:
    python agent/memory_compactor.py
"""

import math
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from agent.db import db_sync as _db

load_dotenv()

AGE_THRESHOLD_DAYS  = 30   # entries older than this are eligible for compaction
DEDUP_SIMILARITY    = 0.95  # cosine similarity threshold for near-duplicate merging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length float vectors."""
    dot  = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Step 1: Deduplication
# ---------------------------------------------------------------------------

def deduplicate_memories() -> int:
    """Remove near-duplicate entries within each (sku, location) group.

    Keeps the most recent entry among any cluster with cosine similarity > DEDUP_SIMILARITY.
    Human-decision entries (decided_by='human') are always retained.

    Returns the number of entries removed.
    """
    removed = 0

    # Get all distinct (sku, location) pairs
    groups = _db.agent_memory.distinct("sku")
    for sku in groups:
        locations = _db.agent_memory.distinct("location", {"sku": sku})
        for location in locations:
            entries = list(_db.agent_memory.find(
                {"sku": sku, "location": location},
                {"_id": 1, "embedding": 1, "content": 1, "decided_by": 1, "created_at": 1},
                sort=[("created_at", -1)],
            ))

            if len(entries) < 2:
                continue

            to_delete: set = set()
            for i, entry_a in enumerate(entries):
                if str(entry_a["_id"]) in to_delete:
                    continue
                emb_a = entry_a.get("embedding", [])
                if not emb_a:
                    continue
                for entry_b in entries[i + 1:]:
                    if str(entry_b["_id"]) in to_delete:
                        continue
                    # Never auto-remove human-decision entries
                    if entry_b.get("decided_by") == "human":
                        continue
                    emb_b = entry_b.get("embedding", [])
                    if not emb_b or len(emb_a) != len(emb_b):
                        continue
                    sim = _cosine_similarity(emb_a, emb_b)
                    if sim >= DEDUP_SIMILARITY:
                        # Keep entry_a (more recent), discard entry_b (older)
                        to_delete.add(str(entry_b["_id"]))

            if to_delete:
                from bson import ObjectId
                _db.agent_memory.delete_many(
                    {"_id": {"$in": [ObjectId(oid) for oid in to_delete]}}
                )
                removed += len(to_delete)
                print(
                    f"  [compactor] Removed {len(to_delete)} near-duplicate(s) "
                    f"for {sku} @ {location}"
                )

    print(f"[compactor] Deduplication complete — {removed} entries removed.")
    return removed


# ---------------------------------------------------------------------------
# Step 2: Age-based summarisation
# ---------------------------------------------------------------------------

def compact_old_memories() -> int:
    """Condense entries older than AGE_THRESHOLD_DAYS into one composite entry per group.

    The composite entry concatenates the content of all collapsed entries.
    Individual human-decision entries older than the threshold are still collapsed
    into the composite so their signal is preserved in summary form.

    Returns the number of original entries replaced by composites.
    """
    cutoff  = datetime.now(timezone.utc) - timedelta(days=AGE_THRESHOLD_DAYS)
    compacted = 0

    groups = _db.agent_memory.distinct("sku")
    for sku in groups:
        locations = _db.agent_memory.distinct("location", {"sku": sku})
        for location in locations:
            old_entries = list(_db.agent_memory.find(
                {"sku": sku, "location": location, "created_at": {"$lt": cutoff}},
                {"_id": 1, "content": 1, "confidence": 1, "decided_by": 1},
                sort=[("created_at", 1)],
            ))

            if len(old_entries) < 3:
                # Not worth compacting fewer than 3 entries
                continue

            composite_content = (
                f"[Compacted memory for {sku} @ {location} — "
                f"{len(old_entries)} entries consolidated on "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}]\n"
                + "\n".join(
                    f"• {e.get('content', '')[:300]}"
                    for e in old_entries
                )
            )

            # Re-embed the composite so it remains vector-searchable.
            # Import here to avoid circular imports at module level.
            from agent.tools import _get_document_embedding  # type: ignore[attr-defined]
            try:
                embedding = _get_document_embedding(composite_content[:2000])
            except Exception as exc:
                print(
                    f"  [compactor] Embedding failed for {sku} @ {location}: {exc} — skipping"
                )
                continue

            from bson import ObjectId
            old_ids = [e["_id"] for e in old_entries]

            _db.agent_memory.insert_one({
                "sku":          sku,
                "location":     location,
                "content":      composite_content,
                "embedding":    embedding,
                "confidence":   "medium",
                "auto_approved": False,
                "decided_by":   "compactor",
                "created_at":   datetime.now(timezone.utc),
            })
            _db.agent_memory.delete_many({"_id": {"$in": old_ids}})
            compacted += len(old_entries)
            print(
                f"  [compactor] Compacted {len(old_entries)} old entries "
                f"for {sku} @ {location} into 1 composite"
            )

    print(f"[compactor] Age compaction complete — {compacted} entries replaced.")
    return compacted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_compaction() -> dict:
    """Run full compaction pipeline: deduplication then age-based summarisation."""
    print("[compactor] Starting memory compaction …")
    deduped   = deduplicate_memories()
    compacted = compact_old_memories()
    result = {"deduped": deduped, "compacted": compacted}
    print(f"[compactor] Done — {result}")
    return result


if __name__ == "__main__":
    run_compaction()
