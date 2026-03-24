"""Unit tests for fusion scoring and provenance behavior."""

from src.gase.config import FusionConfig
from src.gase.models import ChunkNode, ChunkType
from src.gase.retrieval.fusion import compute_fusion_score, fuse_candidates


def _chunk(chunk_id: str, text: str, authority: float = 1.0) -> ChunkNode:
    return ChunkNode(
        id=chunk_id,
        text=text,
        chunk_type=ChunkType.TEXT,
        depth=0,
        breadcrumb_path=f"Root > {chunk_id}",
        document_name="sample.pdf",
        authority_score=authority,
    )


def test_fusion_score_vector_only_mode():
    cfg = FusionConfig(alpha=1.0, beta=0.0, gamma=0.0)
    score = compute_fusion_score(vector_score=0.83, bm25_score=0.25, authority_score=1.8, config=cfg)
    assert score == 0.83


def test_fuse_candidates_ranks_expected_top_chunk():
    cfg = FusionConfig(alpha=0.4, beta=0.4, gamma=0.2)

    chunks = {
        "a": _chunk("a", "net income increased", authority=1.6),
        "b": _chunk("b", "operating costs remained stable", authority=1.1),
        "c": _chunk("c", "risk factor details", authority=1.0),
    }

    vector_scores = {"a": 0.9, "b": 0.7}
    bm25_scores = {"a": 0.8, "c": 0.9}
    authority_scores = {"a": 1.6, "b": 1.1, "c": 1.0}
    expanded_ids = {"a", "b", "c"}

    ranked = fuse_candidates(
        chunks_by_id=chunks,
        vector_scores=vector_scores,
        bm25_scores=bm25_scores,
        authority_scores=authority_scores,
        expanded_ids=expanded_ids,
        config=cfg,
        top_k=3,
    )

    assert len(ranked) == 3
    assert ranked[0].chunk.id == "a"
    assert ranked[0].rank_score >= ranked[1].rank_score
    assert "Vector" in ranked[0].methods_used
    assert "BM25" in ranked[0].methods_used
