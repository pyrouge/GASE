"""
Qdrant-based vector indexer - Semantic search with structure-aware metadata filtering.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from src.gase.models import ChunkNode, DocumentTree
from src.gase.config import QdrantConfig, EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Handles embedding generation (supports multiple backends)."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._load_embeddings()
    
    def _load_embeddings(self) -> None:
        """Load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
            logger.info(f"✓ Loaded embeddings: {self.config.model_name} on {self.config.device}")
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
        )
        return embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0].tolist()


class QdrantIndexer:
    """
    Vector indexer using Qdrant for semantic search.
    
    Stores chunks with metadata for structure-aware filtering:
    - breadcrumb_path: Document hierarchy path
    - depth: Hierarchy depth
    - chunk_type: Type of chunk (header, text, table, etc.)
    - authority_score: Structural authority multiplier
    """
    
    def __init__(self, config: QdrantConfig, embedding_config: EmbeddingConfig):
        """
        Initialize Qdrant indexer.
        
        Args:
            config: Qdrant configuration
            embedding_config: Embedding model configuration
        """
        self.config = config
        self.embedding_config = embedding_config
        self.embedding_provider = EmbeddingProvider(embedding_config)
        self._init_qdrant()
    
    def _init_qdrant(self) -> None:
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
            
            self.QdrantClient = QdrantClient
            self.Distance = Distance
            self.VectorParams = VectorParams
            
            if self.config.mode == "memory":
                self.client = QdrantClient(":memory:")
                logger.info("✓ Qdrant initialized in memory mode")
            else:
                self.client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                )
                logger.info(f"✓ Qdrant connected to {self.config.host}:{self.config.port}")
        
        except ImportError:
            raise ImportError("Install qdrant-client: pip install qdrant-client")
    
    def create_collection(self, collection_name: str, vector_size: Optional[int] = None) -> None:
        """
        Create a Qdrant collection for a document.
        
        Args:
            collection_name: Name of collection
            vector_size: Dimensionality of vectors (default: from config)
        """
        vector_size = vector_size or self.config.vector_size
        
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=self.VectorParams(
                    size=vector_size,
                    distance=self.Distance.COSINE,  # Cosine similarity for normalized embeddings
                ),
            )
            logger.info(f"✓ Created collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Collection may already exist: {e}")
    
    def index_document(self, doc_tree: DocumentTree, vectorize: bool = True) -> None:
        """
        Index all chunks from a document tree.
        
        Args:
            doc_tree: DocumentTree with chunks to index
            vectorize: Whether to compute embeddings (True) or use pre-computed (False)
        """
        collection_name = f"{self.config.collection_name_prefix}_{doc_tree.document.name.replace('.', '_')}"
        
        # Create collection
        self.create_collection(collection_name)
        
        # Prepare chunks
        chunks_to_index = list(doc_tree.all_chunks.values())
        chunk_ids = [c.id for c in chunks_to_index]
        chunk_texts = [c.text[:1000] for c in chunks_to_index]  # Truncate for embedding
        
        # Compute embeddings
        if vectorize:
            logger.info(f"Embedding {len(chunks_to_index)} chunks...")
            embeddings = self.embedding_provider.embed(chunk_texts)
        else:
            embeddings = None
        
        # Batch insert (Qdrant is fast with batch operations)
        batch_size = 100
        for i in range(0, len(chunks_to_index), batch_size):
            batch_chunks = chunks_to_index[i:i+batch_size]
            batch_ids = chunk_ids[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size] if embeddings is not None else None
            
            points = []
            for chunk_id, chunk, embedding in zip(batch_ids, batch_chunks, batch_embeddings):
                payload = {
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "breadcrumb_path": chunk.breadcrumb_path,
                    "depth": chunk.depth,
                    "chunk_type": chunk.chunk_type.value,
                    "authority_score": chunk.authority_score,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                }
                
                from qdrant_client.http.models import PointStruct
                point = PointStruct(
                    id=hash(chunk_id) % (2**31),  # Convert string ID to int
                    vector=embedding.tolist() if embedding is not None else [0.0] * self.config.vector_size,
                    payload=payload,
                )
                points.append(point)
            
            try:
                self.client.upsert(collection_name=collection_name, points=points)
            except Exception as e:
                logger.error(f"Error indexing batch: {e}")
        
        logger.info(f"✓ Indexed {len(chunks_to_index)} chunks in {collection_name}")
    
    def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query: Query text
            collection_name: Collection to search in
            top_k: Number of results to return
            filters: Optional Qdrant filters for metadata
        
        Returns:
            List of (chunk_id, similarity_score, payload) tuples
        """
        # Embed query
        query_embedding = self.embedding_provider.embed_single(query)
        
        # Search (qdrant-client>=1.17 uses query_points)
        if hasattr(self.client, "query_points"):
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=filters,
            )
            result_points = results.points
        else:
            # Backward compatibility with older qdrant-client versions.
            result_points = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=filters,
            )
        
        # Format results
        formatted = []
        for result in result_points:
            formatted.append((
                result.payload.get("chunk_id"),
                result.score,
                result.payload,
            ))
        
        return formatted
    
    def get_chunk(self, collection_name: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk by ID."""
        try:
            point_id = hash(chunk_id) % (2**31)
            point = self.client.retrieve(collection_name=collection_name, ids=[point_id])
            if point:
                return point[0].payload
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
        return None
    
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection (useful for cleanup/reindexing)."""
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"✓ Deleted collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Error deleting collection: {e}")


# Export
__all__ = ["QdrantIndexer", "EmbeddingProvider"]
