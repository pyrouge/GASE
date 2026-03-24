"""Vector retrieval wrapper for GASE."""

from typing import Dict

from src.gase.config import QdrantConfig
from src.gase.indexing.qdrant_indexer import QdrantIndexer


class VectorRetriever:
    """Thin wrapper around QdrantIndexer to return per-chunk similarity scores."""

    def __init__(self, indexer: QdrantIndexer, qdrant_config: QdrantConfig):
        self.indexer = indexer
        self.qdrant_config = qdrant_config

    def _collection_name(self, doc_name: str) -> str:
        return f"{self.qdrant_config.collection_name_prefix}_{doc_name.replace('.', '_')}"

    def retrieve(self, query: str, doc_name: str, top_k: int = 20) -> Dict[str, float]:
        """Return vector similarity scores keyed by chunk_id."""
        results = self.indexer.search(
            query=query,
            collection_name=self._collection_name(doc_name),
            top_k=top_k,
        )
        # Qdrant cosine score can be slightly >1 due to floating-point artifacts.
        return {
            chunk_id: max(0.0, min(float(score), 1.0))
            for chunk_id, score, _ in results
            if chunk_id is not None
        }
