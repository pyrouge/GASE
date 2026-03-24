"""Fusion scoring for multi-signal retrieval."""

from typing import Dict, Iterable, List, Set

from src.gase.config import FusionConfig
from src.gase.models import ChunkNode, RankedResult
from src.gase.retrieval.provenance import build_why_authority, methods_used_for_chunk


def normalize_authority(authority_score: float) -> float:
    """Normalize authority score from [0, 2] to [0, 1]."""
    return max(0.0, min(authority_score / 2.0, 1.0))


def compute_fusion_score(
    vector_score: float,
    bm25_score: float,
    authority_score: float,
    config: FusionConfig,
) -> float:
    """Compute final fused score and clamp into [0, 1]."""
    fused = (
        config.alpha * vector_score
        + config.beta * bm25_score
        + config.gamma * normalize_authority(authority_score)
    )
    return max(0.0, min(float(fused), 1.0))


def fuse_candidates(
    chunks_by_id: Dict[str, ChunkNode],
    vector_scores: Dict[str, float],
    bm25_scores: Dict[str, float],
    authority_scores: Dict[str, float],
    expanded_ids: Iterable[str],
    config: FusionConfig,
    top_k: int,
) -> List[RankedResult]:
    """Fuse per-signal scores into a ranked list with provenance."""
    vector_ids: Set[str] = set(vector_scores.keys())
    bm25_ids: Set[str] = set(bm25_scores.keys())
    expanded_set: Set[str] = set(expanded_ids)
    expanded_only_ids = expanded_set - vector_ids - bm25_ids

    candidate_ids = (vector_ids | bm25_ids | expanded_set) & set(chunks_by_id.keys())

    ranked: List[RankedResult] = []
    for chunk_id in candidate_ids:
        chunk = chunks_by_id[chunk_id]

        v_score = max(0.0, min(float(vector_scores.get(chunk_id, 0.0)), 1.0))
        b_score = max(0.0, min(float(bm25_scores.get(chunk_id, 0.0)), 1.0))
        a_score = float(authority_scores.get(chunk_id, chunk.authority_score))

        rank_score = compute_fusion_score(v_score, b_score, a_score, config)
        methods = methods_used_for_chunk(chunk_id, vector_ids, bm25_ids, expanded_only_ids)

        ranked.append(
            RankedResult(
                chunk=chunk,
                rank_score=rank_score,
                vector_score=v_score,
                bm25_score=b_score,
                authority_score=a_score,
                methods_used=methods,
                breadcrumb_path=chunk.breadcrumb_path,
                why_authority=build_why_authority(chunk.breadcrumb_path, a_score),
            )
        )

    ranked.sort(key=lambda item: item.rank_score, reverse=True)
    return ranked[:top_k]
