"""
Embedding provider abstraction for the Supply Chain Reorder Agent.

Every embedding call in the project (data/seed.py, agent/tools.py,
agent/memory_compactor.py) goes through this module instead of talking to
the Voyage AI SDK directly. To switch embedding providers or models:

  1. Set EMBEDDING_MODEL / EMBEDDING_DIMS in .env — the seeder and the agent
     tools both read from here, so they can never drift out of sync.
  2. If switching SDKs (not just the Voyage model name), implement a new
     `Embeddings` subclass below with embed_query()/embed_documents() and
     point the `embeddings` singleton at it.
  3. Re-run `python data/seed.py`. Embeddings from a different model or
     dimensionality are not interchangeable with what's already stored —
     the seeder drops and recreates the Atlas Vector Search indexes with
     the new EMBEDDING_DIMS and regenerates every stored vector.

Exposes a LangChain-compatible `Embeddings` singleton (`embeddings`) so
callers never touch the underlying SDK directly.
"""

import os
import time

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
import voyageai

from agent.logger import get_logger

load_dotenv()

log = get_logger(__name__)

# ── Configuration — override in .env to change model/dimensions ───────────
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "voyage-4-large")
EMBEDDING_DIMS = int(os.environ.get("EMBEDDING_DIMS", "1024"))


class VoyageEmbeddings(Embeddings):
    """LangChain-compatible wrapper around the Voyage AI embedding API.

    Voyage supports asymmetric "query" vs "document" embedding modes for
    better retrieval quality — embed_query() uses query mode, and
    embed_documents() uses document mode, matching LangChain's Embeddings
    contract (https://python.langchain.com/docs/concepts/embedding_models/).
    """

    def __init__(self, model: str = EMBEDDING_MODEL, max_retries: int = 3):
        self._client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self._model = model
        self._max_retries = max_retries

    def _embed_with_retry(self, text: str, input_type: str) -> list:
        """Embed one string, retrying with exponential back-off (1s → 2s → 4s)
        so transient Voyage AI errors don't cause vector search tools to fail
        silently.
        """
        for attempt in range(self._max_retries):
            try:
                result = self._client.embed([text], model=self._model, input_type=input_type)
                return result.embeddings[0]
            except Exception as exc:
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    log.warning("embedding attempt failed, retrying", extra={
                        "attempt": attempt + 1, "wait_s": wait, "error": str(exc),
                    })
                    time.sleep(wait)
                else:
                    raise

    def embed_query(self, text: str) -> list:
        """Embed a retrieval query using Voyage's query-side embedding mode."""
        return self._embed_with_retry(text, input_type="query")

    def embed_documents(self, texts: list) -> list:
        """Embed persisted documents (orders, memories) using Voyage's document-side mode."""
        return [self._embed_with_retry(t, input_type="document") for t in texts]


# Module-level singleton, reused across the process (mirrors agent/db.py's
# client-singleton pattern). Swap this line to change embedding providers.
embeddings: Embeddings = VoyageEmbeddings()
