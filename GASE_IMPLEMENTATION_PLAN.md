# GASE: Graph-Augmented Structural Ensemble - Complete Implementation Plan

## Executive Summary
GASE is a structure-first RAG system that treats documents as hierarchical knowledge graphs. It combines three retrieval signals (semantic + keyword + structural) via a custom fusion algorithm to outperform entity-only systems on structure-dependent queries.

**Key insight**: Current systems (GraphRAG, LightRAG) focus on entity graphs. GASE prioritizes *document structure hierarchy* (sections, subsections, headers) as the primary retrieval signal.

---

## Architecture Overview

```
User Query → Parallel Retrieval (BM25 + Vector + Graph) → Fusion Ranking → Ranked Results with Provenance
```

**Tech Stack** (all production-ready, actively maintained):
- **Parsing**: Docling (v2.81+) — superior hierarchy preservation
- **Vector Store**: Qdrant (v1.17+) — metadata filtering for structure queries
- **Keyword Search**: BM25s (0.3+) — 573 QPS, pure Python
- **Graph Library**: NetworkX (3.6+) — battle-tested, 1000+ algorithms
- **Evaluation**: Ragas (v0.4+) — industry standard RAG metrics

---

## 5-Phase Implementation (12 weeks)

### Phase 1: Core Infrastructure (Weeks 1–2, 1 person)
**Goal**: Foundation that all other phases depend on

**Deliverables**:
- Pydantic dataclasses: `Document`, `Chunk`, `ChunkNode`, `DocumentTree`, `RankedResult`
- Config system: YAML + env vars for hyperparameters (α=0.4, β=0.4, γ=0.2)
- Structured logging for retrieval pipeline transparency

**Files to create**:
```
src/gase/config.py          # Hyperparameters, env vars
src/gase/models.py          # Pydantic dataclasses
src/gase/logging.py         # Structured logging
tests/unit/test_models.py   # Type validation tests
tests/unit/test_config.py   # Config loading tests
```

**Acceptance criteria**: Dataclasses instantiate correctly, type hints enforced, config loads from YAML + env

---

### Phase 2: Multi-Signal Indexing (Weeks 3–5, 5 people parallel)

#### P2A: Document Parser (Docling)
**Responsibility**: Convert PDFs/Docs into structured JSON tree
- Extract hierarchy (H1→H4 headers)
- Assign "breadcrumb_path" (e.g., "10_K > Risk_Factors > Market > Volatility")
- Assign "depth_score": deeper = lower authority by default
- Preserve table/list structure
- Fallback: OCR for scanned PDFs (pytesseract)

**Files**:
```
src/gase/parser/docling_parser.py
tests/unit/test_docling_parser.py  # Verify breadcrumbs on real 10-Ks
```

#### P2B: Vector Store Indexer (Qdrant)
**Responsibility**: Semantic search + metadata filtering
- Embed chunks via SentenceTransformer/OpenAI/Ollama (configurable)
- Store in Qdrant with payloads: {text, breadcrumb_path, depth, chunk_type}
- Create scalar indices on depth/chunk_type for filtering
- Batch upsert (100–1000 chunks/batch)

**Files**:
```
src/gase/indexing/qdrant_indexer.py
tests/unit/test_qdrant_indexer.py
```

#### P2C: Keyword Index (BM25s)
**Responsibility**: Fast exact keyword matching
- Tokenize chunks with language-specific stemming
- Build inverted index, persist to disk (pickle)
- Support incremental updates

**Files**:
```
src/gase/indexing/bm25_indexer.py
tests/unit/test_bm25_indexer.py
```

#### P2D: Graph Store (NetworkX)
**Responsibility**: Structural relationships + authority scoring
- Build DiGraph: Nodes=Chunks, Edges=(parent→child, sibling→sibling)
- Authority scoring:
  - Base: 1.0
  - Section multipliers: "Executive_Summary"→1.5x, "Methodology"→1.2x
  - Neighbor density boost: +0.2 per neighbor (local hub authority)
- Persist to GraphML for reproducibility

**Files**:
```
src/gase/indexing/graph_indexer.py
tests/unit/test_graph_indexer.py
```

#### P2E: Indexer Orchestrator
**Responsibility**: Coordinate all indexers
- Launch Docling → (Qdrant + BM25s + NetworkX) in parallel
- Transactional: all-or-nothing (rollback if any fails)
- Support incremental updates (re-index changed sections only)

**Files**:
```
src/gase/indexing/indexer.py  # GASE_Indexer class
tests/integration/test_indexing.py
```

**Acceptance criteria**: 
- Parse real 10-K, verify breadcrumb structure
- 100+ chunks indexed to all three stores
- Qdrant collection visible in UI, BM25s searchable, graph visualizable

---

### Phase 3: Hybrid Retrieval & Fusion (Weeks 6–8, 2 people) ⭐ **CRITICAL PATH**

#### Retrieval Orchestrator
**Responsibility**: Parallel multi-signal retrieval

**Logic**:
1. Embed query via same embedding model
2. Launch 3 retrievals in parallel (async):
   - `bm25_retrieve(query, top_k=20)` → ranked list
   - `vector_retrieve(query, top_k=20)` → ranked list via Qdrant
   - `graph_expand(bm25_results ∪ vector_results)` → pull parents + siblings
3. Deduplicate union → candidate_pool (~30–50 chunks)

**Files**:
```
src/gase/retrieval/bm25_retriever.py       # BM25s wrapper
src/gase/retrieval/vector_retriever.py     # Qdrant wrapper
src/gase/retrieval/graph_traversal.py      # Parent + sibling expansion
src/gase/retrieval/orchestrator.py         # GASE_Retriever class
```

#### Fusion Ranking Algorithm ⭐ THE SECRET SAUCE
**Formula**:
```
For each chunk n in candidate_pool:
  
  vector_score = (1.0 - distance_to_query)  [normalized 0–1]
  bm25_score = normalized BM25 score        [0–1]
  authority_score = base_authority(n) × depth_multiplier(n)
  
  R(n) = min(1.0, 
            α · vector_score 
            + β · bm25_score 
            + γ · authority_score)
  
  Final ranking: Sort by R(n) descending
```

**Default weights**: α=0.4, β=0.4, γ=0.2 (equal signal importance)
**Tunable via**: config.py (manual) or QueryContext (per-query override)

**Why this works**:
- Vector: captures semantic relevance
- BM25: catches exact keywords queries miss
- Authority: amplifies high-value sections (Executive Summary, Final Ruling)
- γ smaller (0.2): prevents structure from dominating; it's a tie-breaker

**Files**:
```
src/gase/retrieval/fusion.py               # Fusion algorithm
src/gase/retrieval/provenance.py           # Why-tracking
tests/unit/test_fusion.py                  # Test weight combinations
```

#### Provenance Tracking
**Output per RankedResult**:
```json
{
  "chunk_id": "exec_summary_para_3",
  "text": "...",
  "rank_score": 0.87,
  "provenance": {
    "methods_used": ["BM25", "Vector", "Expansion"],
    "component_scores": {
      "vector": 0.92,
      "bm25": 0.78,
      "authority": 0.95
    },
    "breadcrumb_path": "10_K > Financial_Results > Revenue",
    "why_authority": "Executive summary section (1.5x multiplier) + high neighbor density"
  }
}
```

**Acceptance criteria**:
- E2E test: Query "net income Q3 2023" on financial doc → top_k includes Q3 section
- Fusion weights: test α=1,β=0,γ=0 → expect vector-only ranking
- Provenance: readable and accurate

---

### Phase 4: Evaluation & Benchmarking (Weeks 9–10, 2 people)

#### Evaluation Metrics
**Ragas framework**:
- Context Precision: % of retrieved chunks containing answer
- Faithfulness: % of generated context that's factually grounded
- Answer Relevance: how well answer matches query
- Context Recall: did retrieval capture all answer-related passages

**Custom metrics**:
- Structural Precision: Were high-authority chunks ranked first for structure-dependent queries?
- Noise Ratio: How many distractor chunks (similar but wrong) were pulled?

**Files**:
```
src/gase/eval/metrics.py  # Ragas + custom implementations
```

#### Benchmark Dataset (CUAD or FinanceBench)
**50 queries curated by type**:
1. **Structure-dependent** (15 queries): "What was net income in Q3 2023?" — requires section position awareness
2. **Cross-section** (15 queries): "How does Revenue 2023 vs 2022 compare?" — requires linking sections
3. **Entity** (10 queries): "Who is the CFO?" — baseline tests (vector-only should work)
4. **Entity-in-context** (10 queries): "What is CEO's compensation relative to net income?" — requires traversing sections

**Per query**: {question, document_path, expected_answer, expected_section_path}

**Files**:
```
benchmarks/datasets/cuad_50_queries.json        # Curated queries + answers
benchmarks/datasets/financebench_50_queries.json
```

#### Baselines
Implement / compare against:
1. **Naive RAG**: Vector search only (top_k) + LLM
2. **BM25 only**: Keyword search only
3. **GraphRAG** (if feasible): Microsoft's entity-first approach
4. **LightRAG** (if feasible): Speed-first approach

**Files**:
```
benchmarks/baselines/naive_rag.py
benchmarks/baselines/bm25_only.py
benchmarks/baselines/graphrag.py   # Optional
benchmarks/baselines/lightrag.py   # Optional
```

#### Benchmark Runner
**Script**: `benchmarks/run_benchmark.py`

**Process**:
1. Load 50-query dataset
2. For each query:
   - Run all retrievers (GASE + baselines)
   - Compute Ragas metrics
   - Log latency (indexing + retrieval)
3. Output:
   - CSV: metric values per query per retriever
   - Visualizations: box plots (metric distributions), scatter plot (latency vs accuracy)
   - Markdown report: summary table + winner announcement + failure case analysis

**Expected results**:
- GASE wins on Context Precision (structure-dependent queries): +15–30% vs Naive RAG
- GASE wins on Noise Ratio: fewer distractor chunks
- Latency: comparable to baselines (all sub-second)

**Acceptance criteria**:
- 50 queries executed without errors
- Metrics reasonable (Context Precision 0.3–0.9)
- Report shows GASE winning on ≥1 metric vs all baselines

---

### Phase 5: Documentation & Community (Weeks 11–12, 1 person)

#### Architecture Documentation
**File**: `docs/architecture.md`
- System diagram (ASCII or Mermaid)
- Data flow: query → retrieval → fusion → result
- Component responsibilities + interdependencies
- Extension points (custom parsers, indexers, fusion strategies)

#### API Reference
**File**: `docs/api.md` (auto-generated from docstrings)
- `GASE_Indexer` class + methods
- `GASE_Retriever` class + methods
- `DocumentTree`, `Chunk`, `RankedResult` dataclasses
- `Config` class

#### Quickstart Guide
**File**: `docs/quickstart.md`
```python
# Example: 10-line quickstart
from gase import GASE_Indexer, GASE_Retriever

# 1. Parse and index
indexer = GASE_Indexer(config=Config())
indexer.index_document("financial_report.pdf")

# 2. Retrieve
retriever = GASE_Retriever(config=Config())
results = retriever.retrieve("What was net income in Q3 2023?", top_k=5)

# 3. Results with provenance
for result in results:
    print(f"Chunk: {result.text[:100]}...")
    print(f"Rank: {result.rank_score:.2f}")
    print(f"Why: {result.provenance['why_authority']}")
```

#### Example Notebooks
**Files**: `examples/`
1. `01_parsing_and_indexing.ipynb`: Load PDF → DocumentTree → All three stores
2. `02_retrieval_and_ranking.ipynb`: Query → Retrieval → Fusion → Visualization
3. `03_evaluation.ipynb`: Run benchmark on custom document
4. `04_custom_fusion_strategy.ipynb`: Override fusion weights for domain

#### Contributing Guide
**File**: `CONTRIBUTING.md`
- Dev environment setup (virtual env, dependencies, git hooks)
- Code style: PEP 8, type hints required
- Testing: pytest, >80% coverage
- PR process: linting + tests must pass
- Contributor roles: "Parser Contributors", "Retrieval Engineers", "Benchmark Curators"
- Roadmap: v1.0 → v1.1 → v2.0

#### Community Support
- GitHub Issues: template for bug reports (doc type, query, expected vs actual)
- Discussions: Q&A board for architecture questions
- Roadmap board: public tracking (v1.1, v2.0 features)

**Acceptance criteria**:
- Docs build without errors (Sphinx or markdown)
- Quickstart runs end-to-end in CI/CD
- Example notebooks execute without errors
- First external contributor onboards successfully

---

## Parallel Workstreams Matrix

| Phase | Task | Team | Start | Duration | Dependencies |
|-------|------|------|-------|----------|--------------|
| P1 | Models + Config | 1 person | Week 1 | 2w | None |
| P2A | Docling Parser | 1 person | Week 1 | 5w | P1 ✓ |
| P2B | Qdrant Indexer | 1 person | Week 1 | 5w | P1 ✓ |
| P2C | BM25s Indexer | 1 person | Week 1 | 5w | P1 ✓ |
| P2D | NetworkX+Authority | 1 person | Week 1 | 5w | P1 ✓ |
| P2E | Orchestrator | 1 person | Week 3 | 3w | P2A+P2B+P2C+P2D |
| P3 | Retrieval+Fusion | 2 people | Week 6 | 3w | P2E ✓ |
| P4 | Benchmarking | 2 people | Week 6 | 5w | P3 (parallel) |
| P5 | Documentation | 1 person | Week 9 | 4w | P3+P4 |

**Critical path**: P1 → P2A-D → P2E → P3  
**Ideal team**: 3 critical path + 5 parallel contributors

---

## Critical Files Checklist

```
gase/
├── __init__.py
├── config.py                    ← [P1]
├── models.py                    ← [P1]
├── logging.py                   ← [P1]
├── parser/
│   ├── __init__.py
│   └── docling_parser.py        ← [P2A]
├── indexing/
│   ├── __init__.py
│   ├── qdrant_indexer.py        ← [P2B]
│   ├── bm25_indexer.py          ← [P2C]
│   ├── graph_indexer.py         ← [P2D]
│   └── indexer.py               ← [P2E] Orchestrator
├── retrieval/
│   ├── __init__.py
│   ├── bm25_retriever.py        ← [P3]
│   ├── vector_retriever.py      ← [P3]
│   ├── graph_traversal.py       ← [P3]
│   ├── fusion.py                ← [P3] ⭐ The Algorithm
│   ├── provenance.py            ← [P3]
│   └── orchestrator.py          ← [P3] Pipeline
└── eval/
    ├── __init__.py
    └── metrics.py               ← [P4]

tests/
├── unit/
│   ├── test_models.py           ← [P1]
│   ├── test_config.py           ← [P1]
│   ├── test_docling_parser.py   ← [P2A]
│   ├── test_qdrant_indexer.py   ← [P2B]
│   ├── test_bm25_indexer.py     ← [P2C]
│   ├── test_graph_indexer.py    ← [P2D]
│   └── test_fusion.py           ← [P3]
└── integration/
    ├── test_indexing_e2e.py     ← [P2E]
    └── test_retrieval_e2e.py    ← [P3]

benchmarks/
├── datasets/
│   ├── cuad_50_queries.json     ← [P4]
│   └── financebench_50_queries.json
├── baselines/
│   ├── naive_rag.py             ← [P4]
│   ├── bm25_only.py             ← [P4]
│   ├── graphrag.py              ← [P4]
│   └── lightrag.py              ← [P4]
├── run_benchmark.py             ← [P4] Main runner
└── reports/
    └── benchmark_results.md     ← [P4] Output

docs/
├── architecture.md              ← [P5]
├── api.md                       ← [P5]
├── quickstart.md                ← [P5]
└── CONTRIBUTING.md              ← [P5]

examples/
├── 01_parsing_and_indexing.ipynb    ← [P5]
├── 02_retrieval_and_ranking.ipynb   ← [P5]
├── 03_evaluation.ipynb              ← [P5]
└── 04_custom_fusion_strategy.ipynb  ← [P5]
```

---

## Key Design Decisions & Rationale

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Local-first, no LLM indexing | Cost-effective, reproducible, offline-capable | Manual authority annotations for extraction errors |
| Document structure as primary graph | Universal (all docs have headers); entity extraction is domain-specific | Entity graphs in v1.1 |
| Qdrant over ChromaDB | Superior metadata filtering for structure queries | Requires separate service (Docker/cloud) |
| Tunable fusion weights, not learned | Simple v1.0; no ML training pipeline | Auto-tuning via benchmarks in v1.1 |
| Benchmark on Finance/Legal (not generic QA) | Structure-aware queries show real differentiation | SQuAD unsuitable; Wikipedia structure doesn't matter |
| NetworkX for graph (not FalkorDB) | Battle-tested, standard Python library | Not a dedicated graph DB; fine for <100k docs |

---

## Success Metrics (Phase 4: Evaluation)

| Metric | Target | Notes |
|--------|--------|-------|
| Context Precision (structure-dependent queries) | 70–85% | vs Naive RAG 55–60% = +15–30% win |
| Noise Ratio | TBD vs baselines | How many distractors pulled; GASE should be lower |
| Latency (end-to-end) | <500ms | Indexing: one-time; Retrieval: per-query |
| Code coverage | >80% | Enforced in CI/CD |

---

## Roadmap Beyond v1.0

### v1.1 (Months 3–4)
- **Entity graph overlay**: Cross-reference extraction within sections
- **Multi-document QA**: Link between documents ("Compare revenue across reports")
- **Auto-tuning weights**: Learn α, β, γ from benchmark feedback (RASP optimizer)

### v2.0 (Months 5–6)
- **LLM-based query enrichment**: Expand query with context-awareness
- **Temporal indexing**: Version tracking within evolving documents
- **Domain plugins**: FinTech, Legal, Healthcare templates with pre-configured authorities

---

## Known Risks & Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Docling extraction errors (hierarchy) | MEDIUM | Validation tests on diverse 10-Ks; fallback OCR; manual annotation support |
| Hybrid search fusion complexity | HIGH | Start with simple linear combination; RRF (Reciprocal Rank Fusion) as fallback |
| BM25s non-English support | MEDIUM | Language-specific stemmers; note as limitation in docs |
| NetworkX perf >100k docs | LOW-MEDIUM | Hierarchical optimization; lazy load graph; pre-compute authorities |
| Cost of embedding all chunks | MEDIUM | Support local models (Ollama); batch processing; caching |

---

## Next Steps to Begin

1. **Quick spike (2 hours)**: Parse a real 10-K with Docling, verify breadcrumb extraction quality
2. **Repo setup**: GitHub Actions template (pytest, ruff, mypy) + pyproject.toml with dependencies
3. **Assign contributors**: 1 for P1, 4–5 for P2A-D in parallel
4. **Weekly sync**: Check P2E orchestrator integration points
5. **Design review at Week 3**: Before P2E orchestration, confirm all indexers compatible
