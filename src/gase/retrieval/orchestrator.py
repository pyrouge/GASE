"""Retrieval orchestrator for GASE Phase 3."""

from typing import Dict, Optional

import networkx as nx

from src.gase.config import Config, FusionConfig, get_config
from src.gase.indexing.indexer import GASE_Indexer
from src.gase.models import ChunkNode, QueryContext, RankedResult
from src.gase.retrieval.bm25_retriever import BM25Retriever
from src.gase.retrieval.fusion import fuse_candidates
from src.gase.retrieval.graph_traversal import GraphTraversal
from src.gase.retrieval.vector_retriever import VectorRetriever


class GASE_Retriever:
    """Run BM25 + Vector retrieval, structural expansion, and fusion ranking."""

    def __init__(
        self,
        config: Optional[Config] = None,
        indexer: Optional[GASE_Indexer] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        graph_traversal: Optional[GraphTraversal] = None,
    ):
        self.config = config or get_config()
        self.indexer = indexer or GASE_Indexer(self.config)

        self.bm25_retriever = bm25_retriever or BM25Retriever(self.indexer.bm25_indexer)
        self.vector_retriever = vector_retriever or VectorRetriever(
            self.indexer.qdrant_indexer,
            self.config.qdrant,
        )
        self.graph_traversal = graph_traversal or GraphTraversal(self.indexer.graph_indexer)

    @staticmethod
    def _chunks_from_graph(graph: nx.DiGraph, doc_name: str) -> Dict[str, ChunkNode]:
        """Build minimal ChunkNode map from graph node payload when tree data is unavailable."""
        chunks: Dict[str, ChunkNode] = {}
        for chunk_id, attrs in graph.nodes(data=True):
            chunks[str(chunk_id)] = ChunkNode(
                id=str(chunk_id),
                text=str(attrs.get("text", "")),
                chunk_type=str(attrs.get("chunk_type", "text")),
                depth=int(attrs.get("depth", 0)),
                breadcrumb_path=str(attrs.get("breadcrumb", "")),
                document_name=doc_name,
                authority_score=float(attrs.get("authority", 1.0)),
            )
        return chunks

    def _fusion_config_from_query_context(self, query_context: Optional[QueryContext]) -> FusionConfig:
        """Return effective fusion config, optionally overridden per query."""
        if query_context is None:
            return self.config.fusion

        return FusionConfig(
            alpha=(
                query_context.fusion_alpha
                if query_context.fusion_alpha is not None
                else self.config.fusion.alpha
            ),
            beta=(
                query_context.fusion_beta
                if query_context.fusion_beta is not None
                else self.config.fusion.beta
            ),
            gamma=(
                query_context.fusion_gamma
                if query_context.fusion_gamma is not None
                else self.config.fusion.gamma
            ),
        )

    def retrieve(
        self,
        doc_name: str,
        query: Optional[str] = None,
        top_k: int = 5,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        graph: Optional[nx.DiGraph] = None,
        chunks_by_id: Optional[Dict[str, ChunkNode]] = None,
        query_context: Optional[QueryContext] = None,
    ) -> list[RankedResult]:
        """Execute multi-signal retrieval and return ranked results."""
        effective_query = query_context.query if query_context is not None else query
        if not effective_query:
            raise ValueError("query is required (or pass query_context with query)")

        effective_top_k = query_context.top_k if query_context is not None else top_k
        effective_fusion_config = self._fusion_config_from_query_context(query_context)

        vector_scores = self.vector_retriever.retrieve(effective_query, doc_name=doc_name, top_k=vector_top_k)
        bm25_scores = self.bm25_retriever.retrieve(effective_query, doc_name=doc_name, top_k=bm25_top_k)

        graph_obj = graph or self.indexer.load_graph(doc_name)
        if graph_obj is None:
            expanded_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
            authority_scores = {chunk_id: 1.0 for chunk_id in expanded_ids}
        else:
            seed_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
            expanded_ids = self.graph_traversal.expand_candidate_ids(graph_obj, seed_ids, max_depth=1)
            authority_scores = self.graph_traversal.authority_map(graph_obj, expanded_ids)

        if chunks_by_id is None:
            if graph_obj is None:
                raise ValueError("chunks_by_id is required when graph is unavailable")
            chunks_by_id = self._chunks_from_graph(graph_obj, doc_name)

        return fuse_candidates(
            chunks_by_id=chunks_by_id,
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            authority_scores=authority_scores,
            expanded_ids=expanded_ids,
            config=effective_fusion_config,
            top_k=effective_top_k,
        )
