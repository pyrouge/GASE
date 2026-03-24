"""
Main indexer orchestrator - Coordinates all indexing modules (Parser, Qdrant, BM25, Graph).
"""

import logging
import time
from pathlib import Path
from typing import Optional
import networkx as nx

from src.gase.models import DocumentTree
from src.gase.config import Config, get_config
from src.gase.parser.docling_parser import DoclingParser
from src.gase.indexing.qdrant_indexer import QdrantIndexer
from src.gase.indexing.bm25_indexer import BM25Indexer
from src.gase.indexing.graph_indexer import GraphIndexer
from src.gase.logging import setup_logging, get_logger

logger = logging.getLogger(__name__)


class GASE_Indexer:
    """
    Main indexer orchestrator for GASE.
    
    Workflow:
    1. Parse document (Docling) -> DocumentTree with hierarchy
    2. Index in parallel:
       - Qdrant: Vector embeddings + metadata
       - BM25: Keyword search index
       - NetworkX: Document structure graph
    3. Enable multi-signal retrieval
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize GASE indexer.
        
        Args:
            config: GASE configuration (uses defaults if not provided)
        """
        self.config = config or get_config()
        
        # Setup logging
        setup_logging(level=self.config.log_level)
        
        # Initialize components
        self.parser = DoclingParser(self.config.parsing)
        self.qdrant_indexer = QdrantIndexer(self.config.qdrant, self.config.embedding)
        self.bm25_indexer = BM25Indexer(self.config.bm25)
        self.graph_indexer = GraphIndexer(self.config.graph)
        
        # Track indexed documents
        self.indexed_documents = []
        self.loaded_graphs = {}  # Cache for loaded graphs
        
        logger.info("✓ GASE Indexer initialized")
    
    def index_document(self, file_path: str, force_reindex: bool = False) -> DocumentTree:
        """
        Index a single document end-to-end.
        
        Workflow:
        1. Parse PDF/Doc -> DocumentTree (with hierarchy)
        2. Index in 3 backends simultaneously:
           - Qdrant (vector + metadata)
           - BM25 (keyword)
           - NetworkX (graph + authority)
        
        Args:
            file_path: Path to document (PDF, DOCX, Markdown, HTML)
            force_reindex: If True, delete and rebuild indices
        
        Returns:
            Parsed DocumentTree
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        doc_name = file_path.name
        logger.info(f"Starting indexing pipeline for: {doc_name}")
        start_time = time.time()
        
        # STEP 1: Parse
        logger.info("Step 1/4: Parsing document...")
        try:
            doc_tree = self.parser.parse(str(file_path))
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            raise
        
        # STEP 2: Qdrant indexing
        logger.info("Step 2/4: Indexing in Qdrant...")
        try:
            if force_reindex:
                collection_name = f"{self.config.qdrant.collection_name_prefix}_{doc_name.replace('.', '_')}"
                self.qdrant_indexer.delete_collection(collection_name)
            self.qdrant_indexer.index_document(doc_tree, vectorize=True)
        except Exception as e:
            logger.error(f"Qdrant indexing failed: {e}")
            # Don't raise - allow graceful degradation
        
        # STEP 3: BM25 indexing
        logger.info("Step 3/4: Indexing with BM25...")
        try:
            self.bm25_indexer.index_document(doc_tree)
        except Exception as e:
            logger.error(f"BM25 indexing failed: {e}")
            # Don't raise
        
        # STEP 4: Graph indexing
        logger.info("Step 4/4: Building document graph...")
        try:
            G = self.graph_indexer.index_document(doc_tree)
            self.loaded_graphs[doc_name] = G
        except Exception as e:
            logger.error(f"Graph indexing failed: {e}")
            # Don't raise
        
        # Track
        self.indexed_documents.append(doc_name)
        elapsed = time.time() - start_time
        
        logger.info(f"✓ Indexing complete in {elapsed:.2f}s")
        logger.info(f"  - Chunks: {len(doc_tree.all_chunks)}")
        logger.info(f"  - Indexed documents: {len(self.indexed_documents)}")
        
        return doc_tree
    
    def index_batch(self, file_paths: list, force_reindex: bool = False) -> list:
        """
        Index multiple documents.
        
        Args:
            file_paths: List of document paths
            force_reindex: If True, rebuild all indices
        
        Returns:
            List of DocumentTrees
        """
        results = []
        for file_path in file_paths:
            try:
                doc_tree = self.index_document(file_path, force_reindex=force_reindex)
                results.append(doc_tree)
            except Exception as e:
                logger.error(f"Failed to index {file_path}: {e}")
        
        logger.info(f"✓ Batch indexing complete: {len(results)}/{len(file_paths)} successful")
        return results
    
    def get_indexed_documents(self) -> list:
        """Get list of indexed document names."""
        return self.indexed_documents
    
    def load_graph(self, doc_name: str) -> Optional[nx.DiGraph]:
        """
        Load document graph (from cache or return loaded instance).
        
        Args:
            doc_name: Document name
        
        Returns:
            NetworkX DiGraph or None
        """
        # Check if already loaded
        if doc_name in self.loaded_graphs:
            return self.loaded_graphs[doc_name]
        
        # Load from disk
        G = self.graph_indexer.load_graph(doc_name)
        if G:
            self.loaded_graphs[doc_name] = G
        return G
    
    def get_all_graphs(self) -> dict:
        """Get all loaded graphs."""
        return self.loaded_graphs
    
    def summary(self) -> dict:
        """Get indexing status summary."""
        return {
            "indexed_documents": self.indexed_documents,
            "num_documents": len(self.indexed_documents),
            "loaded_graphs": list(self.loaded_graphs.keys()),
            "config": {
                "embedding_model": self.config.embedding.model_name,
                "qdrant_mode": self.config.qdrant.mode,
                "bm25_k1": self.config.bm25.k1,
                "fusion_weights": {
                    "alpha": self.config.fusion.alpha,
                    "beta": self.config.fusion.beta,
                    "gamma": self.config.fusion.gamma,
                },
            }
        }


# Export
__all__ = ["GASE_Indexer"]
