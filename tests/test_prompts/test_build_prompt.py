"""Tests for prompt builder output — structure and section presence."""

import pytest

from agent.prompts import build_analysis_prompt, build_recommendation_prompt


class TestBuildAnalysisPrompt:
    def test_contains_sku(self, sample_state):
        prompt = build_analysis_prompt(sample_state)
        assert "MED-3017" in prompt

    def test_contains_location(self, sample_state):
        prompt = build_analysis_prompt(sample_state)
        assert "DC-Texas" in prompt

    def test_contains_supplier_section(self, sample_state):
        prompt = build_analysis_prompt(sample_state)
        assert "MedSupply Co" in prompt or "SUPPLIER" in prompt.upper()

    def test_contains_consumption_data(self, sample_state):
        prompt = build_analysis_prompt(sample_state)
        assert "25.0" in prompt or "increasing" in prompt

    def test_returns_non_empty_string(self, sample_state):
        prompt = build_analysis_prompt(sample_state)
        assert isinstance(prompt, str)
        assert len(prompt) > 100


class TestBuildRecommendationPrompt:
    def test_contains_analyst_assessment(self, sample_state):
        prompt = build_recommendation_prompt(sample_state)
        # Should contain the analysis section
        assert "ANALYST" in prompt.upper() or "MedSupply Co" in prompt

    def test_contains_coverage_gap(self, sample_state):
        prompt = build_recommendation_prompt(sample_state)
        assert "155" in prompt  # coverage_gap from sample_state

    def test_contains_supplier_options(self, sample_state):
        prompt = build_recommendation_prompt(sample_state)
        assert "MedSupply Co" in prompt

    def test_returns_string_within_token_limit(self, sample_state):
        """Prompt should not be enormous after trim_to_budget is applied."""
        from agent.context_budget import count_tokens, TOKEN_BUDGET_TOTAL
        prompt = build_recommendation_prompt(sample_state)
        tokens = count_tokens(prompt)
        # Allow some headroom above TOKEN_BUDGET_TOTAL (system + user combined)
        assert tokens < TOKEN_BUDGET_TOTAL * 3

    def test_urgent_stock_level_reflected_in_prompt(self, sample_state):
        """When days_of_stock_remaining < 2 the low stock value should appear in the prompt."""
        sample_state["alert"]["days_of_stock_remaining"] = 1.8
        prompt = build_recommendation_prompt(sample_state)
        assert "1.8" in prompt
