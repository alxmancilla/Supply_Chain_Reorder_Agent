"""Tests for current inline recommendation validation."""

from agent.tools import validate_recommendation


class TestValidateRecommendation:
    def _state(self, gap=155, category="surgical", fda_registered=True):
        return {
            "coverage_gap": gap,
            "inventory": {"category": category},
            "suppliers": [
                {
                    "supplier_id": "SUP-001",
                    "supplier_name": "MedSupply Co",
                    "fda_registered": fda_registered,
                },
            ],
        }

    def test_valid_recommendation_returns_no_errors(self):
        rec = {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      500,
            "rationale":     "Stock critically low; ordering 500 units.",
            "confidence":    "high",
        }
        errors = validate_recommendation(rec, self._state())
        assert errors == []

    def test_missing_required_field_returns_error(self):
        rec = {
            "supplier_name": "MedSupply Co",
            "quantity":      500,
            "rationale":     "Valid rationale text.",
            "confidence":    "high",
        }
        errors = validate_recommendation(rec, self._state())
        assert any("supplier_id" in e for e in errors)

    def test_invalid_confidence_returns_error(self):
        rec = {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      100,
            "rationale":     "Valid rationale here.",
            "confidence":    "super-high",
        }
        errors = validate_recommendation(rec, self._state())
        assert any("confidence" in e for e in errors)

    def test_short_rationale_returns_error(self):
        rec = {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      100,
            "rationale":     "Too short",
            "confidence":    "medium",
        }
        errors = validate_recommendation(rec, self._state())
        assert any("rationale" in e for e in errors)

    def test_zero_gap_with_nonzero_quantity_returns_error(self):
        rec = {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      100,
            "rationale":     "Ordering more stock just in case the gap reappears.",
            "confidence":    "medium",
        }
        errors = validate_recommendation(rec, self._state(gap=0))
        assert any("quantity" in e or "coverage_gap" in e for e in errors)

    def test_zero_quantity_for_zero_gap_is_valid(self):
        rec = {
            "supplier_id":   None,
            "supplier_name": "N/A",
            "quantity":      0,
            "rationale":     "Existing orders already cover the reorder point requirement.",
            "confidence":    "high",
        }
        errors = validate_recommendation(rec, self._state(gap=0))
        assert errors == []

    def test_negative_quantity_returns_error(self):
        rec = {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      -5,
            "rationale":     "This should not have a negative quantity value.",
            "confidence":    "low",
        }
        errors = validate_recommendation(rec, self._state())
        assert any("quantity" in e for e in errors)

    def test_unknown_supplier_returns_error(self):
        rec = {
            "supplier_id":   "SUP-999",
            "supplier_name": "Hallucinated Supplier",
            "quantity":      100,
            "rationale":     "This supplier was not returned by the approved supplier tool.",
            "confidence":    "high",
        }
        errors = validate_recommendation(rec, self._state())
        assert any("Unknown supplier_id" in e for e in errors)

    def test_non_fda_supplier_rejected_for_pharmaceutical(self):
        rec = {
            "supplier_id":   "SUP-001",
            "supplier_name": "MedSupply Co",
            "quantity":      100,
            "rationale":     "Ordering from the only listed supplier for this medication.",
            "confidence":    "medium",
        }
        errors = validate_recommendation(
            rec, self._state(category="pharmaceutical", fda_registered=False)
        )
        assert any("FDA-registered" in e for e in errors)
