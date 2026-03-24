"""Benchmark GASE vs Vector-only vs BM25-only on 25 queries.

Outputs:
- benchmarks/reports/eval_25_per_query.csv
- benchmarks/reports/eval_25_summary.md
"""

from __future__ import annotations

import csv
import json
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
    qid: int
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
    if not ranks:
        return 0.0
    return sum((1.0 / r) for r in ranks if r is not None) / len(ranks)


def mean_hit_at_k(ranks: List[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def topk_texts_by_score(score_map: Dict[str, float], chunks: Dict[str, ChunkNode], k: int) -> List[Tuple[str, str]]:
    ordered = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(chunk_id, chunks[chunk_id].text) for chunk_id, _ in ordered if chunk_id in chunks]


def load_queries(dataset_path: Path) -> List[EvalQuery]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [
        EvalQuery(
            qid=int(item["id"]),
            doc_name=item["doc_name"],
            query=item["query"],
            expected_terms=list(item["expected_terms"]),
        )
        for item in data
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / "benchmarks" / "datasets" / "sample_25_queries.json"
    sample_dir = repo_root / "data" / "sample_documents"
    reports_dir = repo_root / "benchmarks" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(dataset_path)

    config = Config(
        embedding=EmbeddingConfig(device="cpu", batch_size=16),
        qdrant=QdrantConfig(mode="memory", collection_name_prefix="gase_eval25"),
        bm25=BM25Config(cache_dir="data/bm25_eval25_cache"),
        graph=GraphConfig(cache_dir="data/graph_eval25_cache"),
    )

    indexer = GASE_Indexer(config=config)
    retriever = GASE_Retriever(config=config, indexer=indexer)
    vector_retriever = VectorRetriever(indexer.qdrant_indexer, config.qdrant)
    bm25_retriever = BM25Retriever(indexer.bm25_indexer)

    docs_needed = sorted({q.doc_name for q in queries})
    trees: Dict[str, Dict[str, ChunkNode]] = {}

    for doc_name in docs_needed:
        doc_path = sample_dir / doc_name
        if not doc_path.exists():
            raise FileNotFoundError(f"Missing sample document: {doc_path}")
        tree = indexer.index_document(str(doc_path), force_reindex=True)
        trees[doc_name] = tree.all_chunks

    gase_ranks: List[int | None] = []
    vector_ranks: List[int | None] = []
    bm25_ranks: List[int | None] = []

    rows: List[Dict[str, object]] = []

    for q in queries:
        chunks = trees[q.doc_name]

        vector_scores = vector_retriever.retrieve(q.query, q.doc_name, top_k=5)
        bm25_scores = bm25_retriever.retrieve(q.query, q.doc_name, top_k=5)
        gase_results = retriever.retrieve(
            doc_name=q.doc_name,
            query_context=QueryContext(query=q.query, top_k=5),
            chunks_by_id=chunks,
        )

        vector_items = topk_texts_by_score(vector_scores, chunks, k=5)
        bm25_items = topk_texts_by_score(bm25_scores, chunks, k=5)
        gase_items = [(r.chunk.id, r.chunk.text) for r in gase_results]

        r_vec = rank_of_first_hit(vector_items, q.expected_terms)
        r_bm25 = rank_of_first_hit(bm25_items, q.expected_terms)
        r_gase = rank_of_first_hit(gase_items, q.expected_terms)

        vector_ranks.append(r_vec)
        bm25_ranks.append(r_bm25)
        gase_ranks.append(r_gase)

        rows.append(
            {
                "id": q.qid,
                "doc_name": q.doc_name,
                "query": q.query,
                "expected_terms": " | ".join(q.expected_terms),
                "gase_rank": r_gase,
                "vector_rank": r_vec,
                "bm25_rank": r_bm25,
                "gase_hit_at_1": int(r_gase is not None and r_gase <= 1),
                "vector_hit_at_1": int(r_vec is not None and r_vec <= 1),
                "bm25_hit_at_1": int(r_bm25 is not None and r_bm25 <= 1),
            }
        )

    csv_path = reports_dir / "eval_25_per_query.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "doc_name",
                "query",
                "expected_terms",
                "gase_rank",
                "vector_rank",
                "bm25_rank",
                "gase_hit_at_1",
                "vector_hit_at_1",
                "bm25_hit_at_1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "GASE": {
            "hit1": mean_hit_at_k(gase_ranks, 1),
            "hit3": mean_hit_at_k(gase_ranks, 3),
            "hit5": mean_hit_at_k(gase_ranks, 5),
            "mrr": mrr_from_ranks(gase_ranks),
        },
        "Vector-only": {
            "hit1": mean_hit_at_k(vector_ranks, 1),
            "hit3": mean_hit_at_k(vector_ranks, 3),
            "hit5": mean_hit_at_k(vector_ranks, 5),
            "mrr": mrr_from_ranks(vector_ranks),
        },
        "BM25-only": {
            "hit1": mean_hit_at_k(bm25_ranks, 1),
            "hit3": mean_hit_at_k(bm25_ranks, 3),
            "hit5": mean_hit_at_k(bm25_ranks, 5),
            "mrr": mrr_from_ranks(bm25_ranks),
        },
    }

    md_path = reports_dir / "eval_25_summary.md"
    md = []
    md.append("# GASE 25-Query Retrieval Evaluation")
    md.append("")
    md.append("Dataset: `benchmarks/datasets/sample_25_queries.json` (25 queries)")
    md.append("")
    md.append("| Method | Hit@1 | Hit@3 | Hit@5 | MRR |")
    md.append("|---|---:|---:|---:|---:|")
    for name, metrics in summary.items():
        md.append(
            f"| {name} | {metrics['hit1']:.3f} | {metrics['hit3']:.3f} | {metrics['hit5']:.3f} | {metrics['mrr']:.3f} |"
        )

    md.append("")
    md.append(f"Per-query details: `{csv_path.as_posix()}`")

    md_path.write_text("\n".join(md), encoding="utf-8")

    print("\n=== 25-QUERY RETRIEVAL EVALUATION ===")
    for name, metrics in summary.items():
        print(
            f"{name:<14} Hit@1={metrics['hit1']:.3f} Hit@3={metrics['hit3']:.3f} "
            f"Hit@5={metrics['hit5']:.3f} MRR={metrics['mrr']:.3f}"
        )
    print(f"\nWrote: {csv_path.as_posix()}")
    print(f"Wrote: {md_path.as_posix()}")


if __name__ == "__main__":
    main()
