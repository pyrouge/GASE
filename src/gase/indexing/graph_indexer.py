"""
NetworkX-based graph indexer - Document structure hierarchy with authority scoring.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
import networkx as nx
from src.gase.models import DocumentTree, ChunkNode
from src.gase.config import GraphConfig

logger = logging.getLogger(__name__)


class GraphIndexer:
    """
    Graph-based indexer for document structure using NetworkX.
    
    Features:
    - Maintains parent-child-sibling relationships
    - Pre-computes authority scores (high-value sections)
    - Supports graph traversal for structural expansion
    - Persistent storage (GraphML format)
    """
    
    def __init__(self, config: GraphConfig):
        """
        Initialize graph indexer.
        
        Args:
            config: Graph configuration
        """
        self.config = config
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def index_document(self, doc_tree: DocumentTree) -> nx.DiGraph:
        """
        Build document structure graph from DocumentTree.
        
        Args:
            doc_tree: DocumentTree with hierarchical chunks
        
        Returns:
            NetworkX directed graph representing document structure
        """
        doc_name = doc_tree.document.name
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add all chunks as nodes
        for chunk_id, chunk in doc_tree.all_chunks.items():
            # Recalculate authority with density boost
            authority = self._compute_authority(chunk, doc_tree.all_chunks)
            
            # Only include non-None values for GraphML compatibility
            node_attrs = {
                "text": chunk.text[:500],  # Store preview
                "breadcrumb": chunk.breadcrumb_path,
                "depth": chunk.depth,
                "chunk_type": chunk.chunk_type.value,
                "authority": authority,
            }
            if chunk.parent_id:
                node_attrs["parent_id"] = chunk.parent_id
            
            G.add_node(chunk_id, **node_attrs)
        
        # Add edges (structural relationships)
        for chunk_id, chunk in doc_tree.all_chunks.items():
            # Parent edge
            if chunk.parent_id and chunk.parent_id in G:
                G.add_edge(chunk.parent_id, chunk_id, relation="parent-child")
            
            # Sibling edges (bidirectional)
            for sibling_id in chunk.sibling_ids:
                if sibling_id in G:
                    G.add_edge(chunk_id, sibling_id, relation="sibling")
        
        # Save graph
        cache_path = Path(self.config.cache_dir) / f"{doc_name}.graphml"
        try:
            nx.write_graphml(G, cache_path)
            logger.info(f"✓ Saved graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) -> {cache_path}")
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
        
        return G
    
    def load_graph(self, doc_name: str) -> Optional[nx.DiGraph]:
        """Load a previously saved graph."""
        cache_path = Path(self.config.cache_dir) / f"{doc_name}.graphml"
        
        if not cache_path.exists():
            logger.warning(f"Graph not found: {cache_path}")
            return None
        
        try:
            G = nx.read_graphml(cache_path)
            logger.info(f"✓ Loaded graph ({G.number_of_nodes()} nodes)")
            return G
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            return None
    
    def expand_candidates(
        self,
        G: nx.DiGraph,
        chunk_ids: List[str],
        max_depth: int = 1,
    ) -> List[str]:
        """
        Expand candidate chunks with structural neighbors (parents + siblings).
        
        Args:
            G: Document structure graph
            chunk_ids: Initial candidate chunk IDs
            max_depth: How far to traverse (1 = direct neighbors)
        
        Returns:
            Expanded list of chunk IDs
        """
        expanded = set(chunk_ids)
        
        for chunk_id in chunk_ids:
            if chunk_id not in G:
                continue
            
            # Add parent
            predecessors = list(G.predecessors(chunk_id))
            expanded.update(predecessors)
            
            # Add siblings (via parent)
            for parent_id in predecessors:
                siblings = list(G.successors(parent_id))
                expanded.update(siblings)
        
        return list(expanded)
    
    def get_breadcrumb_path(self, G: nx.DiGraph, chunk_id: str) -> str:
        """Get the full breadcrumb path from root to chunk."""
        if chunk_id not in G:
            return ""
        
        # Walk backwards to root via parents
        path = [chunk_id]
        current = chunk_id
        
        while current in G:
            parents = list(G.predecessors(current))
            if not parents:
                break
            current = parents[0]  # Take first parent
            path.insert(0, current)
        
        # Get breadcrumbs and join
        breadcrumbs = [G.nodes[node_id].get('breadcrumb', 'Unknown') for node_id in path]
        return " > ".join(breadcrumbs)
    
    def get_chunk_info(self, G: nx.DiGraph, chunk_id: str) -> Optional[Dict]:
        """Get node information from graph."""
        if chunk_id not in G:
            return None
        
        node = G.nodes[chunk_id]
        return {
            "chunk_id": chunk_id,
            "text": node.get("text", ""),
            "breadcrumb": node.get("breadcrumb", ""),
            "depth": node.get("depth", 0),
            "chunk_type": node.get("chunk_type", "text"),
            "authority": node.get("authority", 1.0),
        }
    
    def _compute_authority(self, chunk: ChunkNode, all_chunks: Dict[str, ChunkNode]) -> float:
        """
        Compute authority score for a chunk.
        
        Authority = base (1.0) × section_multiplier × density_boost
        
        Args:
            chunk: ChunkNode to score
            all_chunks: All chunks for density calculation
        
        Returns:
            Authority score (1.0-2.0)
        """
        authority = 1.0
        
        # Section multiplier (strategic sections get boosted)
        for section_name, multiplier in self.config.authority_multipliers.items():
            if section_name.lower() in chunk.breadcrumb_path.lower():
                authority = multiplier
                break
        
        # Density boost (chunks with many children = information hubs)
        child_count = len(chunk.child_ids)
        density_boost = 1.0 + (self.config.neighbor_density_boost * min(child_count, 5) / 5.0)
        authority *= density_boost
        
        return min(authority, 2.0)
    
    def compute_pagerank(self, G: nx.DiGraph) -> Dict[str, float]:
        """
        Compute PageRank scores for all chunks.
        
        (Could be used for additional ranking in v1.1)
        """
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
        return pagerank
    
    def get_related_chunks(
        self,
        G: nx.DiGraph,
        chunk_id: str,
        relation_type: str = "all",
    ) -> List[str]:
        """
        Get related chunks (parents/children/siblings).
        
        Args:
            G: Document structure graph
            chunk_id: Source chunk ID
            relation_type: 'parent', 'children', 'siblings', or 'all'
        
        Returns:
            List of related chunk IDs
        """
        if chunk_id not in G:
            return []
        
        related = []
        
        if relation_type in ["parent", "all"]:
            related.extend(list(G.predecessors(chunk_id)))
        
        if relation_type in ["children", "all"]:
            related.extend(list(G.successors(chunk_id)))
        
        if relation_type in ["siblings", "all"]:
            parents = list(G.predecessors(chunk_id))
            for parent in parents:
                related.extend([s for s in G.successors(parent) if s != chunk_id])
        
        return list(set(related))


# Export
__all__ = ["GraphIndexer"]
