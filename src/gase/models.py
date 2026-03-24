"""
Data models for GASE - Pydantic dataclasses for type safety and validation.
"""

from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ChunkType(str, Enum):
    """Types of content chunks extracted from documents."""
    HEADER = "header"
    TEXT = "text"
    TABLE = "table"
    LIST = "list"
    IMAGE = "image"


class Document(BaseModel):
    """Represents a parsed document."""
    name: str = Field(..., description="Document name (e.g., 'financial_report_2023.pdf')")
    path: str = Field(..., description="File path to the original document")
    source: str = Field(default="unknown", description="Document source (e.g., 'SEC', 'website')")
    parsed_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")

    class Config:
        use_enum_values = False


class Chunk(BaseModel):
    """Represents a single chunk of text extracted from a document."""
    id: str = Field(..., description="Unique chunk identifier (e.g., 'doc_123_chunk_45')")
    text: str = Field(..., description="Chunk text content")
    chunk_type: ChunkType = Field(default=ChunkType.TEXT)
    depth: int = Field(..., ge=0, description="Hierarchy depth (0=root/top-level)")
    breadcrumb_path: str = Field(..., description="Path in document hierarchy (e.g., 'Section > Subsection > Para')")
    document_name: str = Field(..., description="Name of source document")
    page_number: Optional[int] = Field(default=None)
    character_offset: int = Field(default=0, description="Starting character offset in document")
    
    class Config:
        use_enum_values = False


class ChunkNode(Chunk):
    """Extended chunk with structural relationships in the document graph."""
    parent_id: Optional[str] = Field(default=None, description="ID of parent chunk in hierarchy")
    sibling_ids: List[str] = Field(default_factory=list, description="IDs of sibling chunks")
    child_ids: List[str] = Field(default_factory=list, description="IDs of child chunks")
    authority_score: float = Field(default=1.0, ge=0.0, le=2.0, description="Structural authority multiplier")


class DocumentTree(BaseModel):
    """Complete representation of a document as a hierarchical tree."""
    document: Document
    root_chunks: List[ChunkNode] = Field(..., description="Top-level chunks (depth=0)")
    all_chunks: Dict[str, ChunkNode] = Field(..., description="Map of chunk_id -> ChunkNode")
    chunk_count: int = Field(default=0)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.chunk_count = len(self.all_chunks)

    class Config:
        arbitrary_types_allowed = True


class QueryContext(BaseModel):
    """Context for a retrieval query."""
    query: str = Field(..., description="User's query text")
    query_embedding: Optional[List[float]] = Field(default=None, description="Query embedding vector")
    query_keywords: List[str] = Field(default_factory=list, description="Extracted query keywords")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to retrieve")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filters")
    fusion_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fusion_beta: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fusion_gamma: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    class Config:
        arbitrary_types_allowed = True


class RankedResult(BaseModel):
    """A single retrieval result with ranking scores and provenance."""
    chunk: ChunkNode
    rank_score: float = Field(..., ge=0.0, le=1.0, description="Final fusion rank score")
    
    # Component scores
    vector_score: float = Field(default=0.0, ge=0.0, le=1.0)
    bm25_score: float = Field(default=0.0, ge=0.0, le=1.0)
    authority_score: float = Field(default=0.0, ge=0.0, le=2.0)
    
    # Provenance
    methods_used: List[str] = Field(default_factory=list, description="Which retrievers contributed")
    breadcrumb_path: str = Field(..., description="Full hierarchy path to chunk")
    why_authority: Optional[str] = Field(default=None, description="Human-readable explanation of authority score")
    
    # Metadata
    distance_to_query: Optional[float] = Field(default=None, description="Vector distance to query")
    retrieved_at: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


class RetrievalPipeline(BaseModel):
    """State and configuration for a retrieval operation."""
    query_context: QueryContext
    candidates: List[RankedResult] = Field(default_factory=list)
    retrieved_count: int = Field(default=0)
    retrieval_time_ms: float = Field(default=0.0)
    
    class Config:
        arbitrary_types_allowed = True


@field_validator('rank_score')
def validate_rank_score(cls, v):
    if not (0.0 <= v <= 1.0):
        raise ValueError('rank_score must be between 0.0 and 1.0')
    return v
