"""Unit tests for retrieval orchestrator behavior."""

import networkx as nx

from src.gase.config import Config
from src.gase.models import ChunkNode, ChunkType, QueryContext
from src.gase.retrieval.orchestrator import GASE_Retriever


class _FakeIndexer:
    def __init__(self, graph: nx.DiGraph):
        self._graph = graph

    def load_graph(self, doc_name: str):
        return self._graph


class _FakeVectorRetriever:
    def retrieve(self, query: str, doc_name: str, top_k: int = 20):
        return {"c1": 0.9, "c2": 0.4}


class _FakeBM25Retriever:
    def retrieve(self, query: str, doc_name: str, top_k: int = 20):
        return {"c3": 0.85, "c1": 0.6}


class _FakeGraphTraversal:
    def expand_candidate_ids(self, graph, seed_chunk_ids, max_depth: int = 1):
        return set(seed_chunk_ids) | {"c4"}

    def authority_map(self, graph, chunk_ids):
        return {chunk_id: float(graph.nodes[chunk_id].get("authority", 1.0)) for chunk_id in chunk_ids}


def _chunk(chunk_id: str, text: str, authority: float) -> ChunkNode:
    return ChunkNode(
        id=chunk_id,
        text=text,
        chunk_type=ChunkType.TEXT,
        depth=0,
        breadcrumb_path=f"Root > {chunk_id}",
        document_name="sample.pdf",
        authority_score=authority,
    )


def test_orchestrator_retrieve_fuses_and_sorts_results():
    graph = nx.DiGraph()
    graph.add_node("c1", authority=1.6, text="net income q3", breadcrumb="Root > Financials", depth=0, chunk_type="text")
    graph.add_node("c2", authority=1.1, text="cost details", breadcrumb="Root > Costs", depth=0, chunk_type="text")
    graph.add_node("c3", authority=1.2, text="revenue details", breadcrumb="Root > Revenue", depth=0, chunk_type="text")
    graph.add_node("c4", authority=1.5, text="executive summary", breadcrumb="Root > Summary", depth=0, chunk_type="text")

    chunks = {
        "c1": _chunk("c1", "net income q3", 1.6),
        "c2": _chunk("c2", "cost details", 1.1),
        "c3": _chunk("c3", "revenue details", 1.2),
        "c4": _chunk("c4", "executive summary", 1.5),
    }

    retriever = GASE_Retriever(
        config=Config(),
        indexer=_FakeIndexer(graph),
        bm25_retriever=_FakeBM25Retriever(),
        vector_retriever=_FakeVectorRetriever(),
        graph_traversal=_FakeGraphTraversal(),
    )

    results = retriever.retrieve(
        query="net income q3 2023",
        doc_name="sample.pdf",
        top_k=3,
        chunks_by_id=chunks,
    )

    assert len(results) == 3
    assert results[0].chunk.id == "c1"
    assert results[0].rank_score >= results[1].rank_score
    assert any("BM25" in r.methods_used or "Vector" in r.methods_used for r in results)


def test_orchestrator_query_context_overrides_fusion_weights():
    graph = nx.DiGraph()
    graph.add_node("c1", authority=1.6, text="net income q3", breadcrumb="Root > Financials", depth=0, chunk_type="text")
    graph.add_node("c2", authority=1.1, text="cost details", breadcrumb="Root > Costs", depth=0, chunk_type="text")
    graph.add_node("c3", authority=1.2, text="revenue details", breadcrumb="Root > Revenue", depth=0, chunk_type="text")
    graph.add_node("c4", authority=1.5, text="executive summary", breadcrumb="Root > Summary", depth=0, chunk_type="text")

    chunks = {
        "c1": _chunk("c1", "net income q3", 1.6),
        "c2": _chunk("c2", "cost details", 1.1),
        "c3": _chunk("c3", "revenue details", 1.2),
        "c4": _chunk("c4", "executive summary", 1.5),
    }

    retriever = GASE_Retriever(
        config=Config(),
        indexer=_FakeIndexer(graph),
        bm25_retriever=_FakeBM25Retriever(),
        vector_retriever=_FakeVectorRetriever(),
        graph_traversal=_FakeGraphTraversal(),
    )

    results = retriever.retrieve(
        doc_name="sample.pdf",
        query_context=QueryContext(
            query="focus on bm25",
            top_k=3,
            fusion_alpha=0.0,
            fusion_beta=1.0,
            fusion_gamma=0.0,
        ),
        chunks_by_id=chunks,
    )

    assert len(results) == 3
    assert results[0].chunk.id == "c3"
