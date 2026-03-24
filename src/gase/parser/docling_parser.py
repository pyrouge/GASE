"""
Docling-based document parser - Converts PDFs/Documents into hierarchical DocumentTree structures.
"""

import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from src.gase.models import Document, ChunkNode, DocumentTree, ChunkType
from src.gase.config import ParsingConfig
from datetime import datetime

logger = logging.getLogger(__name__)


class DoclingParser:
    """
    Parses documents using IBM Docling library to extract hierarchical structure.
    
    Docling advantages:
    - Preserves document hierarchy (H1→H4 headers)
    - Detects tables, lists, images
    - Better than Unstructured for structure preservation
    - Supports PDF, DOCX, HTML, Markdown
    """
    
    def __init__(self, config: Optional[ParsingConfig] = None):
        """
        Initialize parser.
        
        Args:
            config: Parsing configuration
        """
        self.config = config or ParsingConfig()
        self.use_docling = os.getenv("GASE_DISABLE_DOCLING", "0") not in {"1", "true", "TRUE"}
        if self.use_docling:
            self._load_docling()
    
    def _load_docling(self) -> None:
        """Lazy-load Docling to avoid import errors if not installed."""
        try:
            from docling.document_converter import DocumentConverter
            self.DocumentConverter = DocumentConverter
            logger.info("✓ Docling loaded successfully")
        except ImportError:
            raise ImportError(
                "Docling not installed. Install with: pip install docling"
            )
    
    def parse(self, file_path: str) -> DocumentTree:
        """
        Parse a document file into hierarchical DocumentTree.
        
        Args:
            file_path: Path to document (PDF, DOCX, Markdown, HTML, etc.)
        
        Returns:
            DocumentTree with hierarchy and breadcrumbs
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        logger.info(f"Parsing document: {path.name}")

        if not self.use_docling:
            return self._parse_with_local_fallback(path)
        
        # Create Document metadata
        doc = Document(
            name=path.name,
            path=str(path.absolute()),
            source="file",
            parsed_at=datetime.now(),
        )
        
        # Convert using Docling
        raw_text = ""
        try:
            converter = self.DocumentConverter()
            docling_result = converter.convert(str(path))
            logger.debug(f"Docling conversion complete for {path.name}")

            doc_obj = getattr(docling_result, "document", docling_result)
            if hasattr(doc_obj, "export_to_markdown"):
                raw_text = doc_obj.export_to_markdown()
            elif hasattr(doc_obj, "to_markdown"):
                raw_text = doc_obj.to_markdown()
            elif hasattr(doc_obj, "text"):
                raw_text = str(doc_obj.text)
            elif hasattr(docling_result, "text"):
                raw_text = str(docling_result.text)
        except Exception as e:
            logger.warning(f"Docling parsing failed ({e}); using local PDF/text fallback")
            raw_text = self._extract_text_fallback(path)
        
        # Extract hierarchy
        all_chunks: Dict[str, ChunkNode] = {}
        root_chunks: List[ChunkNode] = []
        chunk_counter = 0

        try:
            # Build hierarchy from markdown/text.
            if not raw_text.strip():
                raw_text = self._extract_text_fallback(path)

            breadcrumb_stack: List[str] = []
            current_depth = 0

            for line in raw_text.splitlines():
                text = line.strip()
                if not text:
                    continue

                chunk_counter += 1
                chunk_id = f"{doc.name}_{chunk_counter}"

                if text.startswith("#"):
                    level = len(text) - len(text.lstrip("#"))
                    heading_text = text[level:].strip()
                    depth = max(level - 1, 0)
                    chunk_type = ChunkType.HEADER

                    if len(breadcrumb_stack) <= depth:
                        breadcrumb_stack += [""] * (depth - len(breadcrumb_stack) + 1)
                    breadcrumb_stack = breadcrumb_stack[: depth + 1]
                    breadcrumb_stack[depth] = heading_text
                    breadcrumb = " > ".join([seg for seg in breadcrumb_stack if seg]) or "Root"
                    current_depth = depth
                else:
                    depth = current_depth + 1
                    chunk_type = ChunkType.TEXT
                    breadcrumb = " > ".join([seg for seg in breadcrumb_stack if seg]) or "Root"

                # Create chunk
                chunk = ChunkNode(
                    id=chunk_id,
                    text=text,
                    chunk_type=chunk_type,
                    depth=depth,
                    breadcrumb_path=breadcrumb,
                    document_name=doc.name,
                    page_number=None,
                    character_offset=0,
                    parent_id=None,
                    sibling_ids=[],
                    child_ids=[],
                    authority_score=self._calculate_authority(breadcrumb, depth),
                )

                all_chunks[chunk_id] = chunk

                if depth == 0:
                    root_chunks.append(chunk)

            if not all_chunks:
                raise ValueError("No chunks extracted from parsed document")

        except Exception as e:
            logger.warning(f"Error during chunk extraction: {e}. Falling back to simple text split.")
            all_chunks, root_chunks = self._fallback_split(doc)
        
        # Link parent-child relationships
        self._link_hierarchy(all_chunks, root_chunks)
        
        logger.info(f"✓ Parsed {len(all_chunks)} chunks from {doc.name}")
        
        return DocumentTree(
            document=doc,
            root_chunks=root_chunks,
            all_chunks=all_chunks,
        )

    def _parse_with_local_fallback(self, path: Path) -> DocumentTree:
        """Parse using lightweight local text extraction only."""
        doc = Document(
            name=path.name,
            path=str(path.absolute()),
            source="file",
            parsed_at=datetime.now(),
        )

        raw_text = self._extract_text_fallback(path)
        all_chunks: Dict[str, ChunkNode] = {}
        root_chunks: List[ChunkNode] = []
        chunk_counter = 0

        breadcrumb = "Root"
        for line in raw_text.splitlines():
            text = line.strip()
            if not text:
                continue

            chunk_counter += 1
            chunk_id = f"{doc.name}_{chunk_counter}"

            chunk = ChunkNode(
                id=chunk_id,
                text=text,
                chunk_type=ChunkType.TEXT,
                depth=0,
                breadcrumb_path=breadcrumb,
                document_name=doc.name,
                page_number=None,
                character_offset=0,
                parent_id=None,
                sibling_ids=[],
                child_ids=[],
                authority_score=1.0,
            )

            all_chunks[chunk_id] = chunk
            root_chunks.append(chunk)

        if not all_chunks:
            all_chunks, root_chunks = self._fallback_split(doc)

        logger.info(f"✓ Parsed {len(all_chunks)} chunks from {doc.name} (light mode)")
        return DocumentTree(document=doc, root_chunks=root_chunks, all_chunks=all_chunks)
    
    def _classify_element(self, element_type: str, element: Any) -> tuple[ChunkType, int]:
        """Classify element type and determine depth."""
        if "heading" in element_type.lower():
            # H1, H2, H3, H4 mapping to depth
            level = getattr(element, 'level', 1)
            return ChunkType.HEADER, level - 1
        elif "table" in element_type.lower():
            return ChunkType.TABLE, 2
        elif "list" in element_type.lower():
            return ChunkType.LIST, 2
        elif "image" in element_type.lower():
            return ChunkType.IMAGE, 2
        else:
            return ChunkType.TEXT, 1
    
    def _extract_text(self, element: Any) -> str:
        """Extract text content from element."""
        if hasattr(element, 'text'):
            return element.text.strip()
        elif hasattr(element, 'content'):
            return str(element.content).strip()
        else:
            return str(element)[:500].strip()
    
    def _build_breadcrumb(self, element: Any) -> str:
        """Build breadcrumb path from element context."""
        # Ideally get this from document context
        # Fallback to simple type-based breadcrumb
        element_type = type(element).__name__
        
        # This would be enhanced with actual document hierarchy
        breadcrumb_segments = [element_type.replace("_", " ")]
        return " > ".join(breadcrumb_segments)
    
    def _calculate_authority(self, breadcrumb: str, depth: int) -> float:
        """
        Calculate authority score for chunk.
        
        Args:
            breadcrumb: Breadcrumb path
            depth: Hierarchy depth
        
        Returns:
            Authority score (1.0 base, up to 2.0 with multipliers)
        """
        authority = 1.0
        
        # Boost certain sections
        boosted_sections = {
            "Executive Summary": 1.5,
            "Summary": 1.3,
            "Conclusion": 1.2,
            "Results": 1.2,
            "Introduction": 1.1,
        }
        
        for section, multiplier in boosted_sections.items():
            if section.lower() in breadcrumb.lower():
                authority = multiplier
                break
        
        # Headers get small boost
        if "Header" in breadcrumb or "Heading" in breadcrumb:
            authority *= 1.1
        
        return min(authority, 2.0)
    
    def _link_hierarchy(self, all_chunks: Dict[str, ChunkNode], root_chunks: List[ChunkNode]) -> None:
        """Link parent-child relationships in chunk graph."""
        # Group by breadcrumb for hierarchy detection
        breadcrumb_to_chunks = {}
        for chunk_id, chunk in all_chunks.items():
            if chunk.breadcrumb_path not in breadcrumb_to_chunks:
                breadcrumb_to_chunks[chunk.breadcrumb_path] = []
            breadcrumb_to_chunks[chunk.breadcrumb_path].append(chunk_id)
        
        # Simple linking: chunks at depth N link to depth N-1
        depth_groups = {}
        for chunk_id, chunk in all_chunks.items():
            if chunk.depth not in depth_groups:
                depth_groups[chunk.depth] = []
            depth_groups[chunk.depth].append(chunk_id)
        
        # Link parent-child
        for depth in sorted(depth_groups.keys()):
            if depth > 0 and depth - 1 in depth_groups:
                # Simplified: each chunk at depth N links to previous chunk at depth N-1 as parent
                prev_depth_chunks = depth_groups[depth - 1]
                if prev_depth_chunks:
                    parent_id = prev_depth_chunks[-1]  # Last chunk of previous depth
                    for chunk_id in depth_groups[depth]:
                        if parent_id in all_chunks:
                            all_chunks[chunk_id].parent_id = parent_id
                            if chunk_id not in all_chunks[parent_id].child_ids:
                                all_chunks[parent_id].child_ids.append(chunk_id)
        
        # Link siblings (chunks at same depth under same parent)
        for chunk_id, chunk in all_chunks.items():
            if chunk.parent_id and chunk.parent_id in all_chunks:
                parent = all_chunks[chunk.parent_id]
                for sibling_id in parent.child_ids:
                    if sibling_id != chunk_id and sibling_id not in chunk.sibling_ids:
                        chunk.sibling_ids.append(sibling_id)
    
    def _fallback_split(self, doc: Document) -> tuple[Dict[str, ChunkNode], List[ChunkNode]]:
        """
        Fallback: simple text chunking if Docling extraction fails.
        """
        logger.warning("Using fallback text splitting (no hierarchy)")
        
        all_chunks = {}
        root_chunks = []
        chunk_counter = 0
        
        # Create a single root chunk
        chunk_id = f"{doc.name}_chunk_1"
        chunk = ChunkNode(
            id=chunk_id,
            text="[Document parsing failed - content unavailable]",
            chunk_type=ChunkType.TEXT,
            depth=0,
            breadcrumb_path="Root",
            document_name=doc.name,
            parent_id=None,
            sibling_ids=[],
            child_ids=[],
            authority_score=1.0,
        )
        all_chunks[chunk_id] = chunk
        root_chunks.append(chunk)
        
        return all_chunks, root_chunks

    def _extract_text_fallback(self, path: Path) -> str:
        """Extract text locally when Docling conversion is unavailable."""
        if path.suffix.lower() == ".pdf":
            try:
                import pypdfium2 as pdfium

                pdf = pdfium.PdfDocument(str(path))
                pages_text: List[str] = []
                for i in range(len(pdf)):
                    page = pdf[i]
                    text_page = page.get_textpage()
                    pages_text.append(text_page.get_text_range())
                return "\n".join(pages_text)
            except Exception as e:
                logger.warning(f"PDF text fallback failed for {path.name}: {e}")

        # Last-resort fallback for non-PDFs or extraction failures.
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""


# Export
__all__ = ["DoclingParser"]
