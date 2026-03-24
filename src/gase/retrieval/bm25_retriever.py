"""BM25 retrieval wrapper for GASE."""

from typing import Dict

from src.gase.indexing.bm25_indexer import BM25Indexer


class BM25Retriever:
    """Thin wrapper around BM25Indexer to return normalized per-chunk scores."""

    def __init__(self, indexer: BM25Indexer):
        self.indexer = indexer

    def retrieve(self, query: str, doc_name: str, top_k: int = 20) -> Dict[str, float]:
        """Return BM25 scores keyed by chunk_id."""
        results = self.indexer.search(query=query, doc_name=doc_name, top_k=top_k)
        return {chunk_id: float(score) for chunk_id, score in results}
