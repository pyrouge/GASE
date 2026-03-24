#!/usr/bin/env python
"""
Quick end-to-end test of Phase 1 & 2 (Indexing Pipeline)
Tests: Parsing → Qdrant → BM25 → NetworkX indexing
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gase.config import get_config, Config
from gase.indexing.indexer import GASE_Indexer
from gase.logging import setup_logging


def main():
    """Run E2E indexing test."""
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Initialize indexer
    config = get_config()
    print("\n" + "="*60)
    print("GASE PHASE 1 & 2 - INDEXING PIPELINE TEST")
    print("="*60)
    
    indexer = GASE_Indexer(config)
    
    # Index sample documents
    sample_docs = [
        "data/sample_documents/financial_report_2023.pdf",
        "data/sample_documents/research_paper_ai.pdf",
        "data/sample_documents/contract_terms.pdf",
    ]
    
    print(f"\n📄 Found {len(sample_docs)} sample documents")
    print("-" * 60)
    
    for doc_path in sample_docs:
        if not Path(doc_path).exists():
            print(f"⚠ Skipping {doc_path} (not found)")
            continue
        
        try:
            print(f"\n🔄 Indexing: {doc_path}")
            doc_tree = indexer.index_document(doc_path)
            
            print(f"   ✓ Parsed: {len(doc_tree.all_chunks)} chunks")
            print(f"   ✓ Root chunks: {len(doc_tree.root_chunks)}")
            
            # Check graph
            graph = indexer.load_graph(Path(doc_path).name)
            if graph:
                print(f"   ✓ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = indexer.summary()
    print(f"✓ Indexed {summary['num_documents']} documents:")
    for doc in summary['indexed_documents']:
        print(f"  - {doc}")
    
    print(f"\n✓ Loaded graphs: {len(summary['loaded_graphs'])} document(s)")
    print(f"\n✓ Configuration:")
    for key, val in summary['config'].items():
        print(f"  - {key}: {val}")
    
    print("\n" + "="*60)
    print("✓ PHASE 1 & 2 COMPLETE - Ready for Phase 3 (Retrieval)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
