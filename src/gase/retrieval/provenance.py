"""Provenance helpers for retrieval result transparency."""

from typing import List


def build_why_authority(breadcrumb_path: str, authority_score: float) -> str:
    """Create a concise human-readable explanation for authority influence."""
    if authority_score >= 1.4:
        return f"High-authority section from '{breadcrumb_path}' (score={authority_score:.2f})."
    if authority_score >= 1.1:
        return f"Moderately authoritative section from '{breadcrumb_path}' (score={authority_score:.2f})."
    return f"Standard section authority from '{breadcrumb_path}' (score={authority_score:.2f})."


def methods_used_for_chunk(
    chunk_id: str,
    vector_ids: set,
    bm25_ids: set,
    expanded_only_ids: set,
) -> List[str]:
    """Generate method provenance list for a chunk."""
    methods: List[str] = []
    if chunk_id in vector_ids:
        methods.append("Vector")
    if chunk_id in bm25_ids:
        methods.append("BM25")
    if chunk_id in expanded_only_ids:
        methods.append("Expansion")
    return methods
