"""
GASE: Graph-Augmented Structural Ensemble
A structure-first RAG system combining semantic, lexical, and structural retrieval signals.
"""

__version__ = "0.1.0"
__author__ = "GASE Authors"

from src.gase.models import (
    Document,
    Chunk,
    ChunkNode,
    DocumentTree,
    RankedResult,
    QueryContext,
)
from src.gase.config import Config, get_config
from src.gase.indexing.indexer import GASE_Indexer
from src.gase.retrieval.orchestrator import GASE_Retriever

__all__ = [
    "Document",
    "Chunk",
    "ChunkNode",
    "DocumentTree",
    "RankedResult",
    "QueryContext",
    "Config",
    "get_config",
    "GASE_Indexer",
    "GASE_Retriever",
]
