"""End-to-end retrieval integration test on sample PDFs."""

from pathlib import Path

import pytest

from src.gase.config import BM25Config, Config, EmbeddingConfig, GraphConfig, QdrantConfig
from src.gase.indexing.indexer import GASE_Indexer
from src.gase.models import QueryContext
from src.gase.retrieval.orchestrator import GASE_Retriever


@pytest.mark.integration
@pytest.mark.slow
def test_e2e_index_and_retrieve_on_sample_pdfs(tmp_path: Path):
    sample_dir = Path("data") / "sample_documents"
    sample_queries = {
        "financial_report_2023.pdf": "net income q3 2023",
        "research_paper_ai.pdf": "fusion algorithm",
        "contract_terms.pdf": "payment and fees",
    }

    missing = [name for name in sample_queries if not (sample_dir / name).exists()]
    if missing:
        pytest.skip(f"Missing sample documents: {missing}")

    config = Config(
        embedding=EmbeddingConfig(device="cpu", batch_size=16),
        qdrant=QdrantConfig(mode="memory", collection_name_prefix="gase_it"),
        bm25=BM25Config(cache_dir=str(tmp_path / "bm25_cache")),
        graph=GraphConfig(cache_dir=str(tmp_path / "graph_cache")),
    )

    indexer = GASE_Indexer(config=config)
    retriever = GASE_Retriever(config=config, indexer=indexer)

    for doc_name, query in sample_queries.items():
        doc_path = sample_dir / doc_name
        doc_tree = indexer.index_document(str(doc_path), force_reindex=True)

        results = retriever.retrieve(
            doc_name=doc_name,
            query_context=QueryContext(query=query, top_k=5),
            chunks_by_id=doc_tree.all_chunks,
        )

        assert results, f"Expected non-empty retrieval results for {doc_name}"
        assert all(0.0 <= item.rank_score <= 1.0 for item in results)
        assert any(item.methods_used for item in results)

        if doc_name == "financial_report_2023.pdf":
            top_texts = " ".join(item.chunk.text.lower() for item in results[:5])
            assert (
                "net income" in top_texts
                or "q3" in top_texts
                or "revenue" in top_texts
            )
