"""
Structured logging setup for GASE - provides visibility into retrieval pipeline operations.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
import structlog


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_structlog: bool = True
) -> None:
    """
    Configure structured logging for GASE.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        use_structlog: Use structlog for JSON-formatted logs
    """
    
    # Convert level string to logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure structlog if enabled
    if use_structlog:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with structured logging."""
    return logging.getLogger(name)


class RetrievalLogger:
    """Helper class for logging retrieval pipeline operations with context."""
    
    def __init__(self, query: str):
        self.logger = logging.getLogger("gase.retrieval")
        self.query = query
        self.context = {"query": query[:100]}
    
    def log_bm25_retrieval(self, count: int, time_ms: float) -> None:
        """Log BM25 retrieval completion."""
        self.logger.debug(
            f"BM25 retrieval: {count} candidates in {time_ms:.2f}ms",
            extra={**self.context, "method": "bm25", "count": count, "time_ms": time_ms}
        )
    
    def log_vector_retrieval(self, count: int, time_ms: float) -> None:
        """Log vector retrieval completion."""
        self.logger.debug(
            f"Vector retrieval: {count} candidates in {time_ms:.2f}ms",
            extra={**self.context, "method": "vector", "count": count, "time_ms": time_ms}
        )
    
    def log_graph_expansion(self, added_count: int) -> None:
        """Log graph expansion results."""
        self.logger.debug(
            f"Graph expansion: added {added_count} structural neighbors",
            extra={**self.context, "added": added_count}
        )
    
    def log_fusion_ranking(self, total_candidates: int, top_k: int, time_ms: float) -> None:
        """Log fusion ranking completion."""
        self.logger.info(
            f"Fusion ranking: {total_candidates} candidates → top {top_k} in {time_ms:.2f}ms",
            extra={**self.context, "candidates": total_candidates, "top_k": top_k, "time_ms": time_ms}
        )
    
    def log_component_scores(self, chunk_id: str, vector: float, bm25: float, authority: float) -> None:
        """Log component scores for a chunk."""
        self.logger.debug(
            f"Chunk {chunk_id}: vector={vector:.3f}, bm25={bm25:.3f}, authority={authority:.3f}",
            extra={**self.context, "chunk_id": chunk_id, "vector": vector, "bm25": bm25, "authority": authority}
        )


# Export
__all__ = ["setup_logging", "get_logger", "RetrievalLogger"]
