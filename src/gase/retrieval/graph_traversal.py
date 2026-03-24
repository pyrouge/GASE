"""Graph expansion utilities for retrieval candidate enrichment."""

from typing import Dict, Iterable, Set

import networkx as nx

from src.gase.indexing.graph_indexer import GraphIndexer


class GraphTraversal:
    """Structural expansion over the document graph."""

    def __init__(self, graph_indexer: GraphIndexer):
        self.graph_indexer = graph_indexer

    def expand_candidate_ids(
        self,
        graph: nx.DiGraph,
        seed_chunk_ids: Iterable[str],
        max_depth: int = 1,
    ) -> Set[str]:
        """Expand seed chunk IDs with structural neighbors (parents + siblings)."""
        seed_ids = [chunk_id for chunk_id in seed_chunk_ids if chunk_id in graph]
        if not seed_ids:
            return set()
        expanded = self.graph_indexer.expand_candidates(graph, seed_ids, max_depth=max_depth)
        return set(expanded)

    @staticmethod
    def authority_map(graph: nx.DiGraph, chunk_ids: Iterable[str]) -> Dict[str, float]:
        """Return authority values for the requested chunk IDs."""
        values: Dict[str, float] = {}
        for chunk_id in chunk_ids:
            if chunk_id in graph:
                values[chunk_id] = float(graph.nodes[chunk_id].get("authority", 1.0))
        return values
