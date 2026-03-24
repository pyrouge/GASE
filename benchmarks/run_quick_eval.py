"""Quick retrieval benchmark: GASE vs Vector-only vs BM25-only.

This is a lightweight, retrieval-only evaluation on sample PDFs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gase.config import BM25Config, Config, EmbeddingConfig, GraphConfig, QdrantConfig
from src.gase.indexing.indexer import GASE_Indexer
from src.gase.models import ChunkNode, QueryContext
from src.gase.retrieval.bm25_retriever import BM25Retriever
from src.gase.retrieval.orchestrator import GASE_Retriever
from src.gase.retrieval.vector_retriever import VectorRetriever


@dataclass
class EvalQuery:
    doc_name: str
    query: str
    expected_terms: List[str]


def text_hit(text: str, terms: List[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def rank_of_first_hit(items: List[Tuple[str, str]], expected_terms: List[str]) -> int | None:
    for i, (_, chunk_text) in enumerate(items, start=1):
        if text_hit(chunk_text, expected_terms):
            return i
    return None


def mrr_from_ranks(ranks: List[int | None]) -> float:
    vals = [(1.0 / r) for r in ranks if r is not None]
    if not ranks:
        return 0.0
    return sum(vals) / len(ranks)


def mean_hit_at_k(ranks: List[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


def topk_texts_by_score(score_map: Dict[str, float], chunks: Dict[str, ChunkNode], k: int) -> List[Tuple[str, str]]:
    ordered = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:k]
    items: List[Tuple[str, str]] = []
    for chunk_id, _ in ordered:
        if chunk_id in chunks:
            items.append((chunk_id, chunks[chunk_id].text))
    return items


def main() -> None:
    sample_dir = Path("data") / "sample_documents"
    if not sample_dir.exists():
        raise SystemExit("Missing data/sample_documents")

    queries = [
        EvalQuery("financial_report_2023.pdf", "What was net income in 2023?", ["net income", "$450", "450m"]),
        EvalQuery("financial_report_2023.pdf", "What happened in Q3 2023 revenue?", ["q3 2023", "$520", "revenue"]),
        EvalQuery("financial_report_2023.pdf", "What are key risk factors?", ["risk factors", "competition", "currency"]),
        EvalQuery("research_paper_ai.pdf", "What is the fusion algorithm formula?", ["r(n)", "alpha", "beta", "gamma", "fusion"]),
        EvalQuery("research_paper_ai.pdf", "What datasets were used in evaluation?", ["financebench", "cuad", "experimental results"]),
        EvalQuery("research_paper_ai.pdf", "What is the paper conclusion?", ["conclusion", "future work"]),
        EvalQuery("contract_terms.pdf", "What are the payment terms?", ["payment", "$150,000", "30 days", "fees"]),
        EvalQuery("contract_terms.pdf", "How can the agreement be terminated?", ["termination", "30 days", "10 days cure period"]),
        EvalQuery("contract_terms.pdf", "Who are the parties in this agreement?", ["techcorp", "clientco", "parties"]),
    ]

    config = Config(
        embedding=EmbeddingConfig(device="cpu", batch_size=16),
        qdrant=QdrantConfig(mode="memory", collection_name_prefix="gase_eval"),
        bm25=BM25Config(cache_dir="data/bm25_eval_cache"),
        graph=GraphConfig(cache_dir="data/graph_eval_cache"),
    )

    indexer = GASE_Indexer(config=config)
    retriever = GASE_Retriever(config=config, indexer=indexer)
    vector_retriever = VectorRetriever(indexer.qdrant_indexer, config.qdrant)
    bm25_retriever = BM25Retriever(indexer.bm25_indexer)

    # Index each document once.
    trees: Dict[str, Dict[str, ChunkNode]] = {}
    for doc_name in sorted({q.doc_name for q in queries}):
        doc_path = sample_dir / doc_name
        tree = indexer.index_document(str(doc_path), force_reindex=True)
        trees[doc_name] = tree.all_chunks

    gase_ranks: List[int | None] = []
    vector_ranks: List[int | None] = []
    bm25_ranks: List[int | None] = []

    for q in queries:
        chunks = trees[q.doc_name]

        vector_scores = vector_retriever.retrieve(q.query, q.doc_name, top_k=5)
        bm25_scores = bm25_retriever.retrieve(q.query, q.doc_name, top_k=5)

        vector_items = topk_texts_by_score(vector_scores, chunks, k=5)
        bm25_items = topk_texts_by_score(bm25_scores, chunks, k=5)

        gase_results = retriever.retrieve(
            doc_name=q.doc_name,
            query_context=QueryContext(query=q.query, top_k=5),
            chunks_by_id=chunks,
        )
        gase_items = [(r.chunk.id, r.chunk.text) for r in gase_results]

        vector_ranks.append(rank_of_first_hit(vector_items, q.expected_terms))
        bm25_ranks.append(rank_of_first_hit(bm25_items, q.expected_terms))
        gase_ranks.append(rank_of_first_hit(gase_items, q.expected_terms))

    print("\n=== QUICK RETRIEVAL EVALUATION (9 queries) ===")
    print("Metric: Hit@5 and MRR based on expected-term match in retrieved chunk text")
    print()

    def row(name: str, ranks: List[int | None]) -> str:
        return (
            f"{name:<14} "
            f"Hit@1={mean_hit_at_k(ranks, 1):.3f} "
            f"Hit@3={mean_hit_at_k(ranks, 3):.3f} "
            f"Hit@5={mean_hit_at_k(ranks, 5):.3f} "
            f"MRR={mrr_from_ranks(ranks):.3f}"
        )

    print(row("GASE", gase_ranks))
    print(row("Vector-only", vector_ranks))
    print(row("BM25-only", bm25_ranks))


if __name__ == "__main__":
    main()
