#!/usr/bin/env python
"""
Quick unit tests for Phase 1 & 2 components (Models, Config, Qdrant, BM25, Graph)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_models():
    """Test Pydantic models."""
    print("\n" + "="*60)
    print("TEST 1: Pydantic Models")
    print("="*60)
    
    from gase.models import Document, Chunk, ChunkNode, ChunkType
    
    # Create document
    doc = Document(name="test.pdf", path="/tmp/test.pdf")
    print(f"✓ Document: {doc.name}")
    
    # Create chunk
    chunk = Chunk(
        id="chunk_1",
        text="This is a test",
        chunk_type=ChunkType.TEXT,
        depth=0,
        breadcrumb_path="Root",
        document_name="test.pdf"
    )
    print(f"✓ Chunk: {chunk.id} (type={chunk.chunk_type})")
    
    # Create chunk node
    node = ChunkNode(
        id="node_1",
        text="This is a node",
        chunk_type=ChunkType.HEADER,
        depth=0,
        breadcrumb_path="Root > Header",
        document_name="test.pdf",
        authority_score=1.5
    )
    print(f"✓ ChunkNode: {node.id} (authority={node.authority_score})")
    
    return True


def test_config():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("TEST 2: Configuration")
    print("="*60)
    
    from gase.config import Config, get_config
    
    # Test defaults
    config = Config()
    print(f"✓ Config defaults:")
    print(f"  - Embedding model: {config.embedding.model_name}")
    print(f"  - Qdrant mode: {config.qdrant.mode}")
    print(f"  - Fusion weights: α={config.fusion.alpha}, β={config.fusion.beta}, γ={config.fusion.gamma}")
    
    # Test validation
    try:
        config.fusion.alpha = 0.5
        print(f"✓ Weight update: alpha={config.fusion.alpha}")
    except Exception as e:
        print(f"✗ Weight update failed: {e}")
    
    return True


def test_embeddings():
    """Test embedding generation."""
    print("\n" + "="*60)
    print("TEST 3: Embeddings (Sentence Transformers)")
    print("="*60)
    
    from gase.indexing.qdrant_indexer import EmbeddingProvider
    from gase.config import EmbeddingConfig
    
    config = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    
    try:
        provider = EmbeddingProvider(config)
        
        texts = ["Hello world", "Test sentence"]
        embeddings = provider.embed(texts)
        
        print(f"✓ Embedded {len(texts)} texts")
        print(f"  - Embedding shape: {embeddings.shape}")
        print(f"  - Dimension: {embeddings.shape[1]}")
        
        # Test single embedding
        single = provider.embed_single("Test")
        print(f"✓ Single embedding: len={len(single)}")
        
        return True
    except Exception as e:
        print(f"✗ Embedding failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qdrant():
    """Test Qdrant indexing."""
    print("\n" + "="*60)
    print("TEST 4: Qdrant Vector Store")
    print("="*60)
    
    from gase.indexing.qdrant_indexer import QdrantIndexer
    from gase.config import QdrantConfig, EmbeddingConfig
    from gase.models import DocumentTree, Document, ChunkNode, ChunkType
    
    # Create mock document tree
    doc = Document(name="test_qdrant.pdf", path="/tmp/test.pdf")
    
    chunks_dict = {}
    for i in range(5):
        chunk = ChunkNode(
            id=f"test_chunk_{i}",
            text=f"This is test chunk number {i} with some content",
            chunk_type=ChunkType.TEXT,
            depth=0,
            breadcrumb_path=f"Root > Section > Chunk{i}",
            document_name="test_qdrant.pdf",
            authority_score=1.0
        )
        chunks_dict[chunk.id] = chunk
    
    doc_tree = DocumentTree(
        document=doc,
        root_chunks=list(chunks_dict.values()),
        all_chunks=chunks_dict
    )
    
    try:
        # Create indexer
        qdrant_config = QdrantConfig(mode="memory")
        embed_config = EmbeddingConfig(device="cpu")
        indexer = QdrantIndexer(qdrant_config, embed_config)
        
        print("✓ Qdrant indexer created")
        
        # Index document
        indexer.index_document(doc_tree, vectorize=True)
        print(f"✓ Indexed {len(doc_tree.all_chunks)} chunks")
        
        # Search
        collection_name = f"{qdrant_config.collection_name_prefix}_test_qdrant_pdf"
        results = indexer.search(
            query="test chunk content",
            collection_name=collection_name,
            top_k=3
        )
        
        print(f"✓ Search returned {len(results)} results")
        for chunk_id, score, payload in results[:2]:
            print(f"  - {chunk_id}: score={score:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Qdrant test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bm25():
    """Test BM25 indexing."""
    print("\n" + "="*60)
    print("TEST 5: BM25 Keyword Search")
    print("="*60)
    
    from gase.indexing.bm25_indexer import BM25Indexer
    from gase.config import BM25Config
    from gase.models import DocumentTree, Document, ChunkNode, ChunkType
    
    # Create mock document tree
    doc = Document(name="test_bm25.pdf", path="/tmp/test.pdf")
    
    chunks_dict = {}
    test_texts = [
        "Financial results for 2023",
        "Revenue increased by 23 percent",
        "Net income reached 450 million",
        "Operating margin remained stable",
        "Risk factors include market competition",
    ]
    
    for i, text in enumerate(test_texts):
        chunk = ChunkNode(
            id=f"bm25_chunk_{i}",
            text=text,
            chunk_type=ChunkType.TEXT,
            depth=0,
            breadcrumb_path=f"Root > Finance",
            document_name="test_bm25.pdf",
            authority_score=1.0
        )
        chunks_dict[chunk.id] = chunk
    
    doc_tree = DocumentTree(
        document=doc,
        root_chunks=list(chunks_dict.values()),
        all_chunks=chunks_dict
    )
    
    try:
        # Create indexer
        bm25_config = BM25Config()
        indexer = BM25Indexer(bm25_config)
        
        print("✓ BM25 indexer created")
        
        # Index document
        indexer.index_document(doc_tree)
        print(f"✓ Indexed {len(doc_tree.all_chunks)} chunks")
        
        # Search
        results = indexer.search(
            query="financial revenue income",
            doc_name="test_bm25.pdf",
            top_k=3
        )
        
        print(f"✓ Search returned {len(results)} results")
        for chunk_id, score in results[:3]:
            print(f"  - {chunk_id}: score={score:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ BM25 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph():
    """Test NetworkX graph indexing."""
    print("\n" + "="*60)
    print("TEST 6: NetworkX Graph Indexer")
    print("="*60)
    
    from gase.indexing.graph_indexer import GraphIndexer
    from gase.config import GraphConfig
    from gase.models import DocumentTree, Document, ChunkNode, ChunkType
    import networkx as nx
    
    # Create hierarchical document tree
    doc = Document(name="test_graph.pdf", path="/tmp/test.pdf")
    
    chunks_dict = {}
    
    # Root/Header
    root = ChunkNode(
        id="graph_chunk_0",
        text="Executive Summary",
        chunk_type=ChunkType.HEADER,
        depth=0,
        breadcrumb_path="Root",
        document_name="test_graph.pdf",
        authority_score=1.5  # High authority
    )
    chunks_dict[root.id] = root
    
    # Child chunks
    for i in range(1, 4):
        chunk = ChunkNode(
            id=f"graph_chunk_{i}",
            text=f"Section {i} content",
            chunk_type=ChunkType.TEXT,
            depth=1,
            breadcrumb_path=f"Root > Section{i}",
            document_name="test_graph.pdf",
            parent_id=root.id,
            authority_score=1.0
        )
        chunks_dict[chunk.id] = chunk
        root.child_ids.append(chunk.id)
    
    doc_tree = DocumentTree(
        document=doc,
        root_chunks=[root],
        all_chunks=chunks_dict
    )
    
    try:
        # Create indexer
        graph_config = GraphConfig()
        indexer = GraphIndexer(graph_config)
        
        print("✓ Graph indexer created")
        
        # Index document
        G = indexer.index_document(doc_tree)
        print(f"✓ Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Test expansion
        expanded = indexer.expand_candidates(G, ["graph_chunk_1"], max_depth=1)
        print(f"✓ Expanded candidates: {len(expanded)} chunks")
        
        # Test authority scores
        for node_id in G.nodes():
            authority = G.nodes[node_id].get("authority", 1.0)
            print(f"  - {node_id}: authority={authority:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Graph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("GASE PHASE 1 & 2 - UNIT TESTS")
    print("="*60)
    
    tests = [
        ("Models", test_models),
        ("Config", test_config),
        ("Embeddings", test_embeddings),
        ("Qdrant", test_qdrant),
        ("BM25", test_bm25),
        ("Graph", test_graph),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = "✓" if passed else "✗"
        except Exception as e:
            print(f"✗ {test_name} error: {e}")
            results[test_name] = "✗"
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        print(f"  {result} {test_name}")
    
    passed = sum(1 for r in results.values() if r == "✓")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} passed")
    print("="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
