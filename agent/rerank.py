"""
Reranking provider abstraction for the Supply Chain Reorder Agent.

Allows for pluggable reranking (e.g., Voyage Rerank, Cohere, or an LLM) to
refine vector search results. Enabled via RERANKER_ENABLED=1 in .env.
"""

import os
import time
from typing import List, Protocol, runtime_checkable

import voyageai
from dotenv import load_dotenv

from agent.logger import get_logger

load_dotenv()
log = get_logger(__name__)

# -- Configuration --
RERANKER_ENABLED = os.environ.get("RERANKER_ENABLED", "0") == "1"
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "rerank-2")


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        documents: List[dict],
        text_key: str,
        top_k: int = 3,
    ) -> List[dict]:
        """
        Rerank a list of documents based on a query string.

        Args:
            query: The original user situation/query.
            documents: List of candidate documents from initial retrieval.
            text_key: The dictionary key containing the text to be reranked.
            top_k: How many results to return after reranking.
        """
        ...


class VoyageReranker(Reranker):
    """Voyage AI implementation of the Reranker protocol."""

    def __init__(self, model: str = RERANKER_MODEL, max_retries: int = 3):
        self._client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
        self._model = model
        self._max_retries = max_retries

    def rerank(
        self,
        query: str,
        documents: List[dict],
        text_key: str,
        top_k: int = 3,
    ) -> List[dict]:
        if not documents:
            return []

        # Extract texts for reranking from the specified field
        texts = [doc.get(text_key, "") for doc in documents]

        for attempt in range(self._max_retries):
            try:
                result = self._client.rerank(
                    query=query,
                    documents=texts,
                    model=self._model,
                    top_k=top_k,
                )

                # result.results is a list of objects with index and relevance_score
                reranked_docs = []
                for r in result.results:
                    doc = documents[r.index].copy()
                    doc["rerank_score"] = r.relevance_score
                    reranked_docs.append(doc)
                return reranked_docs

            except Exception as exc:
                if attempt < self._max_retries - 1:
                    wait = 2**attempt
                    log.warning(
                        "reranking attempt failed, retrying",
                        extra={
                            "attempt": attempt + 1,
                            "wait_s": wait,
                            "error": str(exc),
                        },
                    )
                    time.sleep(wait)
                else:
                    log.error(
                        "reranking failed after all retries; falling back to original order",
                        extra={"error": str(exc)},
                    )
                    return documents[:top_k]

        return documents[:top_k]


class NullReranker(Reranker):
    """Pass-through reranker that respects top_k but does no scoring."""

    def rerank(
        self,
        query: str,
        documents: List[dict],
        text_key: str,
        top_k: int = 3,
    ) -> List[dict]:
        return documents[:top_k]


# Module-level singleton instance
reranker: Reranker = VoyageReranker() if RERANKER_ENABLED else NullReranker()
