"""Tests for the pluggable reranking layer in agent/tools.py."""

from unittest.mock import MagicMock, patch

import pytest

from agent.tools import find_similar_past_orders, get_long_term_memories
from tests.conftest import requires_db

pytestmark = requires_db


@pytest.fixture
def mock_results():
    return [
        {"rationale": "Order A", "content": "Memory A", "similarity_score": 0.9},
        {"rationale": "Order B", "content": "Memory B", "similarity_score": 0.85},
        {"rationale": "Order C", "content": "Memory C", "similarity_score": 0.8},
    ]


@pytest.fixture
def mock_reranker():
    m = MagicMock()
    # Mock rerank to reverse the order to show it's doing something
    m.rerank.side_effect = lambda query, documents, text_key, top_k: list(reversed(documents))[:top_k]
    return m


def test_find_similar_past_orders_calls_reranker(mock_results, mock_reranker):
    """Verify that find_similar_past_orders calls the reranker when results are found."""
    mock_db = MagicMock()
    mock_db.order_history.aggregate.return_value = mock_results

    with patch("agent.tools._db", mock_db), \
         patch("agent.tools._get_query_embedding", return_value=[0.0] * 1024), \
         patch("agent.tools._reranker", mock_reranker):

        # Access the underlying function if it's a StructuredTool
        func = getattr(find_similar_past_orders, "func", find_similar_past_orders)
        results = func("some situation", limit=2)

        assert mock_reranker.rerank.called
        # Our mock reverses, so we expect C then B
        assert results[0]["rationale"] == "Order C"
        assert results[1]["rationale"] == "Order B"
        assert len(results) == 2


def test_get_long_term_memories_calls_reranker(mock_results, mock_reranker):
    """Verify that get_long_term_memories calls the reranker when results are found."""
    mock_db = MagicMock()
    mock_db.agent_memory.aggregate.return_value = mock_results

    with patch("agent.tools._db", mock_db), \
         patch("agent.tools._get_query_embedding", return_value=[0.0] * 1024), \
         patch("agent.tools._reranker", mock_reranker):

        results = get_long_term_memories("some situation", limit=2)

        assert mock_reranker.rerank.called
        # Our mock reverses, so we expect C then B (content field)
        assert results[0]["content"] == "Memory C"
        assert results[1]["content"] == "Memory B"
        assert len(results) == 2
