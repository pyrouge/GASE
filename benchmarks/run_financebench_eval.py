"""Community-ready retrieval benchmark on FinanceBench.

Compares:
- GASE
- Vector-only
- BM25-only

Outputs:
- benchmarks/reports/financebench_per_query.csv
- benchmarks/reports/financebench_summary.md
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gase.config import BM25Config, Config, EmbeddingConfig, GraphConfig, QdrantConfig
from src.gase.indexing.indexer import GASE_Indexer
from src.gase.models import ChunkNode, QueryContext
from src.gase.retrieval.bm25_retriever import BM25Retriever
from src.gase.retrieval.orchestrator import GASE_Retriever
from src.gase.retrieval.vector_retriever import VectorRetriever


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_numeric_terms(text: str) -> List[str]:
    return re.findall(r"\$?\d+(?:[\.,]\d+)?%?", text)


def extract_keyword_terms(text: str, max_terms: int = 8) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", text.lower())
    seen = []
    for w in words:
        if w not in seen:
            seen.append(w)
        if len(seen) >= max_terms:
            break
    return seen


def expected_terms_from_row(row: dict) -> List[str]:
    terms: List[str] = []

    answer = str(row.get("answer", "")).strip()
    if answer:
        if len(answer) <= 80:
            terms.append(answer)
        terms.extend(extract_numeric_terms(answer))
        terms.extend(extract_keyword_terms(answer, max_terms=6))

    evidence = row.get("evidence", []) or []
    if isinstance(evidence, list) and evidence:
        ev0 = evidence[0]
        ev_text = str(ev0.get("evidence_text", "")) if isinstance(ev0, dict) else str(ev0)
        terms.extend(extract_numeric_terms(ev_text)[:5])
        terms.extend(extract_keyword_terms(ev_text, max_terms=6))

    # Deduplicate and keep non-empty terms.
    dedup: List[str] = []
    for t in terms:
        t = t.strip()
        if t and t not in dedup:
            dedup.append(t)
    return dedup


def text_hit(text: str, expected_terms: List[str]) -> bool:
    norm = normalize_text(text)
    for term in expected_terms:
        t = normalize_text(term)
        if not t:
            continue
        if t in norm:
            return True
    return False


def rank_of_first_hit(items: List[Tuple[str, str]], expected_terms: List[str]) -> Optional[int]:
    for idx, (_, chunk_text) in enumerate(items, start=1):
        if text_hit(chunk_text, expected_terms):
            return idx
    return None


def mean_hit_at_k(ranks: List[Optional[int]], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: List[Optional[int]]) -> float:
    if not ranks:
        return 0.0
    return sum((1.0 / r) for r in ranks if r is not None) / len(ranks)


def topk_texts_by_score(score_map: Dict[str, float], chunks: Dict[str, ChunkNode], k: int) -> List[Tuple[str, str]]:
    ordered = sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:k]
    return [(chunk_id, chunks[chunk_id].text) for chunk_id, _ in ordered if chunk_id in chunks]


def infer_extension(url: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in {".pdf", ".html", ".htm", ".txt", ".md"}:
        return suffix
    return ".pdf"


def download_document(doc_link: str, out_path: Path, timeout: int = 60) -> bool:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            return True

        with requests.get(doc_link, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def select_queries(dataset_rows: list, max_queries: int, max_docs: int) -> List[dict]:
    selected: List[dict] = []
    docs = []

    for row in dataset_rows:
        doc_name = row.get("doc_name")
        if doc_name not in docs and len(docs) >= max_docs:
            continue

        if doc_name not in docs:
            docs.append(doc_name)

        selected.append(row)
        if len(selected) >= max_queries:
            break

    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=25)
    parser.add_argument("--max-docs", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    reports_dir = repo_root / "benchmarks" / "reports"
    docs_dir = repo_root / "data" / "financebench_docs"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FinanceBench dataset...")
    ds = load_dataset("PatronusAI/financebench", split="train")
    rows = [ds[i] for i in range(len(ds))]
    selected = select_queries(rows, max_queries=args.max_queries, max_docs=args.max_docs)

    print(f"Selected {len(selected)} queries across up to {args.max_docs} documents")

    doc_to_local_path: Dict[str, Path] = {}
    query_rows: List[dict] = []

    for row in selected:
        doc_name = row.get("doc_name", "unknown_doc")
        doc_link = row.get("doc_link", "")
        if not doc_link:
            continue

        ext = infer_extension(doc_link)
        local = docs_dir / f"{doc_name}{ext}"
        ok = download_document(doc_link, local)
        if not ok:
            continue

        doc_to_local_path[doc_name] = local
        query_rows.append(row)

    if not doc_to_local_path:
        raise RuntimeError("No documents downloaded successfully from FinanceBench links")

    print(f"Downloaded/cached {len(doc_to_local_path)} documents")
    print(f"Evaluating {len(query_rows)} queries")

    config = Config(
        embedding=EmbeddingConfig(device="cpu", batch_size=16),
        qdrant=QdrantConfig(mode="memory", collection_name_prefix="gase_finbench"),
        bm25=BM25Config(cache_dir="data/bm25_financebench_cache"),
        graph=GraphConfig(cache_dir="data/graph_financebench_cache"),
    )

    # FinanceBench filings can be very large; use lightweight local parsing for benchmark stability.
    os.environ["GASE_DISABLE_DOCLING"] = "1"

    indexer = GASE_Indexer(config=config)
    retriever = GASE_Retriever(config=config, indexer=indexer)
    vector_retriever = VectorRetriever(indexer.qdrant_indexer, config.qdrant)
    bm25_retriever = BM25Retriever(indexer.bm25_indexer)

    trees: Dict[str, Dict[str, ChunkNode]] = {}
    indexed_name_by_doc: Dict[str, str] = {}
    for doc_name, doc_path in doc_to_local_path.items():
        tree = indexer.index_document(str(doc_path), force_reindex=True)
        trees[doc_name] = tree.all_chunks
        indexed_name_by_doc[doc_name] = tree.document.name

    per_doc_counter = defaultdict(int)
    gase_ranks: List[Optional[int]] = []
    vec_ranks: List[Optional[int]] = []
    bm25_ranks: List[Optional[int]] = []

    per_query_records: List[dict] = []

    for row in query_rows:
        doc_name = row["doc_name"]
        if doc_name not in trees:
            continue

        question = str(row.get("question", ""))
        expected_terms = expected_terms_from_row(row)
        if not expected_terms:
            continue

        chunks = trees[doc_name]

        indexed_doc_name = indexed_name_by_doc[doc_name]

        vector_scores = vector_retriever.retrieve(question, indexed_doc_name, top_k=args.top_k)
        bm25_scores = bm25_retriever.retrieve(question, indexed_doc_name, top_k=args.top_k)
        gase_results = retriever.retrieve(
            doc_name=indexed_doc_name,
            query_context=QueryContext(query=question, top_k=args.top_k),
            chunks_by_id=chunks,
        )

        vec_items = topk_texts_by_score(vector_scores, chunks, k=args.top_k)
        bm_items = topk_texts_by_score(bm25_scores, chunks, k=args.top_k)
        ga_items = [(r.chunk.id, r.chunk.text) for r in gase_results]

        r_gase = rank_of_first_hit(ga_items, expected_terms)
        r_vec = rank_of_first_hit(vec_items, expected_terms)
        r_bm = rank_of_first_hit(bm_items, expected_terms)

        gase_ranks.append(r_gase)
        vec_ranks.append(r_vec)
        bm25_ranks.append(r_bm)

        per_doc_counter[doc_name] += 1

        per_query_records.append(
            {
                "financebench_id": row.get("financebench_id"),
                "doc_name": doc_name,
                "question": question,
                "answer": str(row.get("answer", ""))[:200],
                "gase_rank": r_gase,
                "vector_rank": r_vec,
                "bm25_rank": r_bm,
                "gase_hit_at_1": int(r_gase is not None and r_gase <= 1),
                "vector_hit_at_1": int(r_vec is not None and r_vec <= 1),
                "bm25_hit_at_1": int(r_bm is not None and r_bm <= 1),
                "expected_terms_preview": " | ".join(expected_terms[:8]),
            }
        )

    if not per_query_records:
        raise RuntimeError("No queries evaluated successfully")

    summary = {
        "GASE": {
            "hit1": mean_hit_at_k(gase_ranks, 1),
            "hit3": mean_hit_at_k(gase_ranks, 3),
            "hit5": mean_hit_at_k(gase_ranks, 5),
            "mrr": mrr(gase_ranks),
        },
        "Vector-only": {
            "hit1": mean_hit_at_k(vec_ranks, 1),
            "hit3": mean_hit_at_k(vec_ranks, 3),
            "hit5": mean_hit_at_k(vec_ranks, 5),
            "mrr": mrr(vec_ranks),
        },
        "BM25-only": {
            "hit1": mean_hit_at_k(bm25_ranks, 1),
            "hit3": mean_hit_at_k(bm25_ranks, 3),
            "hit5": mean_hit_at_k(bm25_ranks, 5),
            "mrr": mrr(bm25_ranks),
        },
    }

    csv_path = reports_dir / "financebench_per_query.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query_records[0].keys()))
        writer.writeheader()
        writer.writerows(per_query_records)

    md_path = reports_dir / "financebench_summary.md"
    lines = []
    lines.append("# FinanceBench Retrieval Evaluation")
    lines.append("")
    lines.append(f"Queries evaluated: {len(per_query_records)}")
    lines.append(f"Documents indexed: {len(doc_to_local_path)}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Method | Hit@1 | Hit@3 | Hit@5 | MRR |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, m in summary.items():
        lines.append(f"| {name} | {m['hit1']:.3f} | {m['hit3']:.3f} | {m['hit5']:.3f} | {m['mrr']:.3f} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This run uses lexical expected-term matching derived from answer and evidence fields.")
    lines.append("- Use this as retrieval-grounding evaluation; pair with answer-level metrics for final publication.")
    lines.append("- Per-query details are available in CSV for auditability.")
    lines.append("")
    lines.append(f"Per-query CSV: {csv_path.as_posix()}")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== FINANCEBENCH RETRIEVAL EVAL ===")
    for name, m in summary.items():
        print(
            f"{name:<12} Hit@1={m['hit1']:.3f} Hit@3={m['hit3']:.3f} "
            f"Hit@5={m['hit5']:.3f} MRR={m['mrr']:.3f}"
        )
    print(f"\nWrote: {csv_path.as_posix()}")
    print(f"Wrote: {md_path.as_posix()}")


if __name__ == "__main__":
    main()
