"""Pure unit tests for memory payload shape."""

from unittest.mock import MagicMock, patch

from agent.tools import write_short_term_memory_sync


def test_short_term_memory_records_human_rejection_fields():
    db = MagicMock()
    rec = {
        "supplier_id": "SUP-001",
        "supplier_name": "MedSupply Co",
        "quantity": 100,
        "confidence": "medium",
        "rationale": "Human rejected this proposed supplier and quantity.",
    }

    with patch("agent.tools._db", db):
        write_short_term_memory_sync(
            "MED-3017", "DC-Texas", rec, 1.5, False, "human", "rejected"
        )

    inserted = db.short_term_memory.insert_one.call_args.args[0]
    assert inserted["decided_by"] == "human"
    assert inserted["human_decision"] == "rejected"
    assert inserted["auto_approved"] is False
