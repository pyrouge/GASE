"""
BM25s-based keyword indexer - Fast lexical search with term frequency matching.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from src.gase.models import DocumentTree, ChunkNode
from src.gase.config import BM25Config

logger = logging.getLogger(__name__)


class BM25Indexer:
    """
    Keyword indexer using BM25s for fast lexical matching.
    
    BM25s advantages:
    - Pure Python, no dependencies on search engines
    - 573 QPS (40x faster than older BM25 implementations)
    - Token-level control for language-specific stemming
    """
    
    def __init__(self, config: BM25Config):
        """
        Initialize BM25 indexer.
        
        Args:
            config: BM25 configuration
        """
        self.config = config
        self._load_bm25()
        
        # Cache directory
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_bm25(self) -> None:
        """Load BM25s library."""
        try:
            import bm25s
            from tokenizers import Tokenizer
            self.bm25s = bm25s
            self.Tokenizer = Tokenizer
            logger.info("✓ BM25s loaded successfully")
        except ImportError:
            raise ImportError(
                "BM25s not installed. Install with: "
                "pip install bm25s tokenizers"
            )
    
    def _get_tokenizer(self) -> Any:
        """Get language-appropriate tokenizer."""
        # For now, use basic whitespace tokenizer
        # Could be extended with language-specific stemmers
        return lambda text: text.lower().split()
    
    def index_document(self, doc_tree: DocumentTree) -> None:
        """
        Build BM25 index from document tree.
        
        Args:
            doc_tree: DocumentTree with chunks to index
        """
        doc_name = doc_tree.document.name
        cache_path = Path(self.config.cache_dir) / f"{doc_name}.bm25"
        
        # Prepare documents
        chunks = list(doc_tree.all_chunks.values())
        documents = [
            {
                "chunk_id": c.id,
                "text": c.text,
                "breadcrumb": c.breadcrumb_path,
            }
            for c in chunks
        ]
        
        # Tokenize
        tokenizer = self._get_tokenizer()
        corpus = [doc["text"] for doc in documents]
        corpus_tokens = [tokenizer(text) for text in corpus]
        
        # Build BM25 index (bm25s 0.3+ expects params in constructor, then index())
        retriever = self.bm25s.BM25(k1=self.config.k1, b=self.config.b)
        retriever.index(corpus_tokens, show_progress=False)
        
        # Store metadata
        metadata = {
            "chunk_ids": [doc["chunk_id"] for doc in documents],
            "breadcrumbs": [doc["breadcrumb"] for doc in documents],
            "document_name": doc_name,
        }
        
        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump((retriever, metadata), f)
            logger.info(f"✓ Indexed {len(documents)} chunks via BM25 -> {cache_path}")
        except Exception as e:
            logger.error(f"Error saving BM25 index: {e}")
            raise
    
    def search(
        self,
        query: str,
        doc_name: str,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Search for chunks matching query using BM25.
        
        Args:
            query: Query text
            doc_name: Document name (must be indexed first)
            top_k: Number of results to return
        
        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by score descending
        """
        cache_path = Path(self.config.cache_dir) / f"{doc_name}.bm25"
        
        if not cache_path.exists():
            logger.warning(f"BM25 index not found for {doc_name}")
            return []
        
        # Load index
        try:
            with open(cache_path, "rb") as f:
                retriever, metadata = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
            return []
        
        # Tokenize query
        tokenizer = self._get_tokenizer()
        query_tokens = tokenizer(query)
        
        # Search
        retrieved = retriever.retrieve([query_tokens], k=top_k, show_progress=False)

        # bm25s 0.3+ returns a Results object with `documents` and `scores`.
        if hasattr(retrieved, "documents") and hasattr(retrieved, "scores"):
            indices = retrieved.documents
            scores = retrieved.scores
        else:
            # Fallback for tuple-like return shape.
            indices, scores = retrieved
        
        # Format results
        results = []
        for idx, score in zip(indices[0], scores[0]):  # First query (we have just one)
            chunk_id = metadata["chunk_ids"][int(idx)]
            # Normalize score to 0-1 range (BM25 scores can be > 1)
            normalized_score = min(float(score), 1.0)
            results.append((chunk_id, normalized_score))
        
        return results
    
    def search_all_loaded(
        self,
        query: str,
        doc_names: List[str],
        top_k: int = 20,
    ) -> List[Tuple[str, float, str]]:
        """
        Search across multiple indexed documents.
        
        Args:
            query: Query text
            doc_names: List of document names to search
            top_k: Results per document
        
        Returns:
            List of (chunk_id, score, document) tuples
        """
        all_results = []
        
        for doc_name in doc_names:
            try:
                results = self.search(query, doc_name, top_k=top_k)
                for chunk_id, score in results:
                    all_results.append((chunk_id, score, doc_name))
            except Exception as e:
                logger.warning(f"Error searching {doc_name}: {e}")
        
        # Sort by score
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k * len(doc_names)]


# Export
__all__ = ["BM25Indexer"]
