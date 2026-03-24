"""
Configuration management for GASE - supports YAML files, environment variables, and code defaults.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import yaml


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model."""
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    device: str = Field(default="cpu", description="'cpu' or 'cuda'")
    batch_size: int = Field(default=32, ge=1)


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector store."""
    mode: str = Field(default="memory", description="'memory' or 'server'")
    host: Optional[str] = Field(default="localhost")
    port: Optional[int] = Field(default=6333)
    collection_name_prefix: str = Field(default="gase")
    # Vector dimensions for all-MiniLM-L6-v2
    vector_size: int = Field(default=384)


class BM25Config(BaseModel):
    """Configuration for BM25 keyword search."""
    language: str = Field(default="english", description="Language for stemming")
    k1: float = Field(default=1.5, description="BM25 k1 parameter")
    b: float = Field(default=0.75, description="BM25 b parameter")
    cache_dir: str = Field(default="data/bm25_cache")


class GraphConfig(BaseModel):
    """Configuration for NetworkX graph indexing."""
    format: str = Field(default="graphml", description="Graph serialization format")
    cache_dir: str = Field(default="data/graph_cache")
    # Authority scoring
    authority_multipliers: Dict[str, float] = Field(
        default_factory=lambda: {
            "Executive_Summary": 1.5,
            "Summary": 1.3,
            "Conclusion": 1.2,
            "Results": 1.2,
            "Methodology": 1.1,
        }
    )
    neighbor_density_boost: float = Field(default=0.2)


class FusionConfig(BaseModel):
    """Configuration for retrieval fusion algorithm."""
    alpha: float = Field(default=0.4, ge=0.0, le=1.0, description="Vector signal weight")
    beta: float = Field(default=0.4, ge=0.0, le=1.0, description="BM25 signal weight")
    gamma: float = Field(default=0.2, ge=0.0, le=1.0, description="Authority signal weight")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Warn if weights don't sum to <= 1.0 (will be clamped to 1.0)
        total = self.alpha + self.beta + self.gamma
        if total > 1.0:
            print(f"⚠ Warning: Fusion weights sum to {total} > 1.0 (will clamp results)")


class ParsingConfig(BaseModel):
    """Configuration for document parsing."""
    chunk_size: int = Field(default=512, ge=100, description="Target chunk size in characters")
    overlap: int = Field(default=100, ge=0, description="Overlap between chunks")
    extract_tables: bool = Field(default=True)
    extract_images: bool = Field(default=False)
    ocr_enabled: bool = Field(default=False, description="Enable OCR for scanned PDFs")


class Config(BaseModel):
    """
    Master configuration for GASE combining all components.
    Loaded from: environment variables → YAML file → defaults
    """
    # Core
    project_name: str = Field(default="GASE")
    data_dir: str = Field(default="data")
    log_level: str = Field(default="INFO")
    
    # Sub-configs
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def load_env() -> None:
    """Load environment variables from .env file."""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)


def load_config_from_yaml(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(path):
        return {}
    
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}
    
    return config_dict


def get_config(config_path: Optional[str] = None, env_override: bool = True) -> Config:
    """
    Get configuration by merging sources: defaults → YAML → environment variables.
    
    Args:
        config_path: Path to YAML config file (optional)
        env_override: If True, environment variables override file config
    
    Returns:
        Config object with merged settings
    """
    load_env()
    
    # Start with defaults
    config_dict = {}
    
    # Merge YAML file if provided
    if config_path:
        yaml_config = load_config_from_yaml(config_path)
        config_dict.update(yaml_config)
    
    # Merge environment variables if enabled
    if env_override:
        for key in [
            "embedding_model_name", "embedding_device",
            "qdrant_mode", "qdrant_host", "qdrant_port",
            "bm25_language",
            "fusion_alpha", "fusion_beta", "fusion_gamma",
            "parsing_chunk_size", "parsing_ocr_enabled",
            "log_level", "data_dir"
        ]:
            env_val = os.getenv(key.upper())
            if env_val:
                # Parse as appropriate type
                if "_" in key:
                    parts = key.split("_")
                    if len(parts) >= 2:
                        # Reconstruct nested config
                        category = parts[0]
                        if category not in config_dict:
                            config_dict[category] = {}
                        if isinstance(config_dict[category], dict):
                            field_name = "_".join(parts[1:])
                            config_dict[category][field_name] = env_val
    
    return Config(**config_dict)


# Export
__all__ = ["Config", "get_config", "load_env", "load_config_from_yaml"]
