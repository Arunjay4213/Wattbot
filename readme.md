# Project Overview

This document outlines the complete implementation strategy for the **WattBot2025 Kaggle Competition** - building an advanced Retrieval-Augmented Generation (RAG) system for environmental AI research papers.

Competition URL: https://www.kaggle.com/competitions/WattBot2025/overview

---

# Part 1: Tech Stack Selection {.tabset}

## Core Infrastructure

```yaml
Language: Python 3.10+
Framework: FastAPI (for API endpoints)
Vector Database: 
  Primary: Qdrant (best for metadata filtering)
  Alternative: Weaviate (built-in hybrid search)
Database: PostgreSQL (for paper metadata, citations)
Cache: Redis (for query caching)
Message Queue: Celery + RabbitMQ (for async processing)
Containerization: Docker + Docker Compose
Orchestration: Kubernetes (if scaling needed)
```

## ML/AI Stack

```yaml
LLM Framework: 
  - LangChain (for RAG pipeline)
  - LlamaIndex (for advanced document processing)
  
Embedding Models:
  - Primary: sentence-transformers/all-MiniLM-L6-v2 (fast)
  - Secondary: BAAI/bge-large-en-v1.5 (accurate)
  - Domain-specific: SciBERT (scientific text)

LLMs:
  - Primary: GPT-4 (for answer generation)
  - Fallback: Claude-3 (for verification)
  - Local: Mistral-7B (for fast operations)

Document Processing:
  - Unstructured.io (PDF parsing)
  - Table Transformer (Microsoft)
  - Tesseract (OCR)
  - Docling (IBM's new parser)
```

## Data Processing Stack

```yaml
PDF Processing: PyPDF2, pdfplumber, pdf2image
Table Extraction: Camelot, Tabula, Table Transformer
OCR: Tesseract, EasyOCR, PaddleOCR
Graph Database: Neo4j (for knowledge graph)
ML Training: PyTorch + Transformers
Evaluation: RAGAS + custom metrics
Monitoring: Weights & Biases, MLflow
```

---

# Part 2: System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend Layer                         │
│              (API Gateway + Load Balancer)               │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────┴───────────────────────────────────┐
│                  Application Layer                       │
│   ┌──────────────────────────────────────────────┐     │
│   │  Query Processor    │    Answer Generator     │     │
│   │  Citation Manager   │    Confidence Scorer    │     │
│   └──────────────────────────────────────────────┘     │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────┴───────────────────────────────────┐
│                  Intelligence Layer                      │
│   ┌──────────────────────────────────────────────┐     │
│   │  Knowledge Graph   │    Custom Reranker       │     │
│   │  Entity Resolver   │    Numeric Reasoner      │     │
│   └──────────────────────────────────────────────┘     │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────┴───────────────────────────────────┐
│                   Data Layer                            │
│   ┌──────────────────────────────────────────────┐     │
│   │  Vector Store     │    Document Store         │     │
│   │  Graph Database   │    Metadata Store         │     │
│   └──────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

---

# Part 3: Data Processing Pipeline

## Stage 1: Document Ingestion & Analysis

```python
Document_Ingestion_Pipeline:
│
├── 1. PDF Classification
│   ├── Identify document type (research paper, report, dataset)
│   ├── Extract metadata (title, authors, year, journal)
│   └── Detect structure (sections, tables, figures)
│
├── 2. Multi-Modal Extraction
│   ├── Text Extraction
│   │   ├── Main body text
│   │   ├── Headers/footers
│   │   ├── Footnotes
│   │   └── References
│   │
│   ├── Table Extraction
│   │   ├── Detect table boundaries
│   │   ├── Extract structure (rows, columns, headers)
│   │   ├── Preserve cell relationships
│   │   └── Convert to structured format (JSON/CSV)
│   │
│   └── Figure Extraction
│       ├── Detect figures/charts
│       ├── Extract captions
│       ├── OCR text in images
│       └── Generate figure descriptions (using GPT-4V)
│
└── 3. Reference Extraction
    ├── Parse bibliography
    ├── Extract in-text citations
    └── Build citation network
```

## Stage 2: Intelligent Chunking Strategy

```python
Hierarchical_Chunking_System:
│
├── Level 1: Document Segmentation
│   ├── Split by sections (Abstract, Methods, Results)
│   ├── Preserve logical boundaries
│   └── Maintain header hierarchy
│
├── Level 2: Semantic Chunking
│   ├── Chunk Size: 512-1024 tokens (adaptive)
│   ├── Algorithm: Semantic similarity threshold
│   ├── Rules:
│   │   ├── Never split tables
│   │   ├── Never split numbered lists
│   │   ├── Keep citations with claims
│   │   └── Preserve equation context
│   └── Output: Semantic chunks with metadata
│
├── Level 3: Overlap Strategy
│   ├── 10% overlap between chunks
│   ├── Sliding window for context preservation
│   └── Boundary sentences duplicated
│
└── Level 4: Metadata Enrichment
    ├── chunk_id: UUID
    ├── source_paper: paper_id
    ├── section: "Results"
    ├── page_numbers: [5, 6]
    ├── contains_table: true
    ├── contains_numeric: true
    ├── entity_types: ["MODEL", "METRIC", "VALUE"]
    ├── citations_refs: ["strubel2019", "li2023"]
    └── confidence_score: 0.95
```

## Stage 3: Knowledge Extraction

```python
Knowledge_Extraction_Pipeline:
│
├── 1. Named Entity Recognition (NER)
│   ├── Models: ["BERT", "GPT-3", "LLAMA"]
│   ├── Metrics: ["CO2", "kWh", "PUE", "WUE"]
│   ├── Values: [{value: 1438, unit: "lbs", metric: "CO2"}]
│   └── Hardware: ["V100", "A100", "H100"]
│
├── 2. Relationship Extraction
│   ├── Model-Hardware: "BERT trained on V100"
│   ├── Metric-Value: "PUE equals 1.58"
│   ├── Temporal: "increased from 2019 to 2023"
│   └── Causal: "cooling reduces PUE"
│
├── 3. Numeric Processing
│   ├── Extract all numbers with units
│   ├── Normalize to standard units
│   ├── Identify ranges and uncertainties
│   └── Calculate derived metrics
│
└── 4. Fact Extraction
    ├── Claims with evidence
    ├── Definitions
    ├── Methodologies
    └── Contradictions detection
```

---

# Part 4: Retrieval System

## Multi-Stage Retrieval Architecture

```python
Advanced_Retrieval_System:
│
├── Stage 1: Query Processing
│   ├── Query Classification
│   │   ├── Type: [factual, comparative, definitional, calculation]
│   │   ├── Entities: extract_entities(query)
│   │   └── Intent: identify_intent(query)
│   │
│   ├── Query Expansion (RAG-Fusion)
│   │   ├── Generate 5 variant queries
│   │   ├── Include synonyms (CO2 → carbon dioxide)
│   │   ├── Add context (BERT → BERT model training)
│   │   └── Create sub-queries for complex questions
│   │
│   └── Query Decomposition
│       ├── Break multi-part questions
│       ├── Identify required evidence types
│       └── Plan retrieval strategy
│
├── Stage 2: Hybrid Retrieval
│   ├── Sparse Retrieval (BM25)
│   │   ├── Exact keyword matching
│   │   ├── High weight for numbers/units
│   │   └── Boost for exact phrases
│   │
│   ├── Dense Retrieval (Embeddings)
│   │   ├── Semantic similarity search
│   │   ├── Multiple embedding models
│   │   └── Cross-encoder reranking
│   │
│   ├── Graph Retrieval
│   │   ├── Traverse knowledge graph
│   │   ├── Find connected entities
│   │   └── Aggregate related facts
│   │
│   └── Structured Retrieval
│       ├── Table-specific search
│       ├── SQL queries on extracted data
│       └── Regex for patterns
│
├── Stage 3: Fusion & Reranking
│   ├── Reciprocal Rank Fusion (k=60)
│   ├── Custom domain reranker
│   ├── Diversity promotion
│   └── Relevance threshold filtering
│
└── Stage 4: Context Assembly
    ├── Deduplicate retrieved chunks
    ├── Order by relevance
    ├── Add supporting context
    └── Verify completeness
```

---

# Part 5: Knowledge Graph System

## Graph Schema Design

```python
Knowledge_Graph_Schema:

Nodes:
├── Paper(id, title, year, venue, doi)
├── Model(name, type, parameters, year_released)
├── Metric(name, unit, category)
├── Value(amount, unit, context, confidence)
├── Hardware(name, manufacturer, tdp, memory)
├── Institution(name, type, country)
└── Author(name, affiliation)

Edges:
├── CITES(paper → paper, year)
├── MEASURES(paper → metric, value)
├── TRAINED_ON(model → hardware, duration)
├── REQUIRES(model → metric, amount)
├── CONTRADICTS(value → value, reason)
├── SUPPORTS(paper → claim)
└── AUTHORED_BY(paper → author)

Graph_Algorithms:
├── Community Detection (research clusters)
├── PageRank (paper importance)
├── Shortest Path (fact verification)
├── Centrality (key concepts)
└── Temporal Analysis (trend detection)
```

---

# Part 6: Answer Generation System

## Multi-Component Generation Pipeline

```python
Answer_Generation_Pipeline:
│
├── 1. Evidence Validation
│   ├── Check retrieval confidence
│   ├── Verify evidence completeness
│   ├── Detect contradictions
│   └── Decision: Answer vs "Unable to answer"
│
├── 2. Answer Planning
│   ├── Structure: [answer, citations, materials]
│   ├── Identify claims needing citation
│   ├── Map evidence to claims
│   └── Plan numeric calculations
│
├── 3. Generation Strategy
│   ├── Simple Factual:
│   │   └── Direct extraction + citation
│   │
│   ├── Multi-Document:
│   │   ├── Merge evidence from sources
│   │   ├── Resolve conflicts
│   │   └── Synthesize comprehensive answer
│   │
│   ├── Calculation Required:
│   │   ├── Extract values
│   │   ├── Perform computation
│   │   └── Show work + cite sources
│   │
│   └── Definition/Explanation:
│       ├── Find authoritative definition
│       ├── Add context if helpful
│       └── Cite primary source
│
├── 4. Citation Management
│   ├── Track source for each fact
│   ├── Format: [paper_year]
│   ├── Add supporting materials
│   └── Verify citation accuracy
│
└── 5. Post-Processing
    ├── Format verification
    ├── Unit standardization
    ├── Confidence scoring
    └── Fallback handling
```

---

# Part 7: Evaluation & Monitoring

## Comprehensive Evaluation Framework

```python
Evaluation_System:
│
├── Component-Level Metrics
│   ├── Retrieval Metrics
│   │   ├── Precision@K
│   │   ├── Recall@K
│   │   ├── MRR (Mean Reciprocal Rank)
│   │   └── NDCG (Normalized Discounted Cumulative Gain)
│   │
│   ├── Generation Metrics
│   │   ├── RAGAS Faithfulness
│   │   ├── Answer Relevancy
│   │   ├── Context Precision
│   │   └── Context Recall
│   │
│   └── Domain-Specific Metrics
│       ├── Citation Accuracy
│       ├── Numeric Precision
│       ├── Unit Correctness
│       └── Fallback Precision
│
├── End-to-End Metrics
│   ├── Query Success Rate
│   ├── Average Response Time
│   ├── User Satisfaction Score
│   └── Error Rate
│
└── Monitoring Dashboard
    ├── Real-time performance
    ├── Error tracking
    ├── Query patterns
    └── System health
```

---

# Part 8: Optimization & Innovation

## Performance Optimizations

```python
Optimization_Strategy:
│
├── Caching Layer
│   ├── Query result caching
│   ├── Embedding cache
│   ├── LLM response cache
│   └── Frequent pattern cache
│
├── Indexing Strategy
│   ├── HNSW index for vectors
│   ├── Inverted index for keywords
│   ├── B-tree for numeric ranges
│   └── Graph index for relationships
│
├── Batch Processing
│   ├── Batch embed documents
│   ├── Parallel chunk processing
│   ├── Async LLM calls
│   └── GPU optimization
│
└── Model Optimization
    ├── Quantization (INT8)
    ├── Distillation
    ├── Pruning
    └── ONNX conversion
```

---

# Part 9: Implementation Timeline

## 15-Day Sprint Plan

### Days 1-3: Foundation
- Set up infrastructure
- Implement PDF processing
- Basic chunking

### Days 4-6: Data Layer
- Build vector store
- Create knowledge graph
- Implement metadata store

### Days 7-9: Retrieval System
- Hybrid search
- RAG-Fusion
- Custom reranker

### Days 10-12: Generation
- Answer generation
- Citation system
- Fallback handling

### Days 13-14: Evaluation
- RAGAS integration
- Custom metrics
- Testing

### Day 15: Polish
- UI/UX improvements
- Performance optimization
- Documentation

---

# Part 10: Winning Differentiators

## Secret Sauce Components

```python
Innovation_Stack:
│
├── 1. Contradiction Resolution System
│   ├── Detect conflicting values
│   ├── Use citation network for authority
│   ├── Show both with confidence scores
│   └── Let user choose trusted source
│
├── 2. Interactive Evidence Browser
│   ├── Show retrieved chunks
│   ├── Highlight used portions
│   ├── Allow drill-down
│   └── Export citations
│
├── 3. Confidence Calibration
│   ├── Per-statement confidence
│   ├── Source reliability score
│   ├── Temporal relevance (newer = better)
│   └── Consensus detection
│
├── 4. Smart Fallback Hierarchy
│   ├── Level 1: High-confidence answer
│   ├── Level 2: Answer with caveats
│   ├── Level 3: Partial answer
│   ├── Level 4: Related information
│   └── Level 5: "Unable to answer"
│
└── 5. Learning Component
    ├── Track failed queries
    ├── User feedback integration
    ├── Active learning for reranker
    └── Continuous improvement
```

---

# Final Architecture Summary

## Complete Data Flow

```
User Query
    ↓
[Query Processor] → Classify, Expand, Decompose
    ↓
[Multi-Stage Retrieval] → BM25 + Dense + Graph + Tables
    ↓
[Knowledge Graph] → Add related facts
    ↓
[Reranking] → Domain-specific scoring
    ↓
[Evidence Assembly] → Merge, Dedupe, Order
    ↓
[Answer Generator] → Generate with citations
    ↓
[Post-Processing] → Format, Verify, Score
    ↓
Final Answer with Citations
```

---

# Implementation Code Examples

## Example: Document Processing Pipeline

```{python, eval=FALSE}
import pdfplumber
from transformers import pipeline
import numpy as np

class DocumentProcessor:
    def __init__(self):
        self.ner = pipeline("ner", model="dslim/bert-base-NER")
        
    def process_pdf(self, pdf_path):
        """Extract and process PDF content"""
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            tables = []
            
            for page in pdf.pages:
                # Extract text
                text += page.extract_text() or ""
                
                # Extract tables
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
                    
        return {
            "text": text,
            "tables": tables,
            "metadata": self.extract_metadata(text)
        }
    
    def extract_metadata(self, text):
        """Extract metadata from document text"""
        # Extract entities
        entities = self.ner(text[:1000])  # Process first 1000 chars
        
        return {
            "entities": entities,
            "length": len(text),
            "has_tables": bool(tables)
        }
```

## Example: Hybrid Retrieval Implementation

```{python, eval=FALSE}
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.bm25 = None
        
    def index_documents(self, documents):
        """Index documents for hybrid search"""
        # Prepare for BM25
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Prepare embeddings
        self.embeddings = self.encoder.encode(documents)
        self.documents = documents
        
    def search(self, query, k=10):
        """Perform hybrid search"""
        # BM25 search
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Dense search
        query_embedding = self.encoder.encode([query])
        dense_scores = np.dot(self.embeddings, query_embedding.T).squeeze()
        
        # Reciprocal Rank Fusion
        def rrf(rank, k=60):
            return 1 / (k + rank)
        
        bm25_ranks = np.argsort(-bm25_scores)
        dense_ranks = np.argsort(-dense_scores)
        
        scores = {}
        for i, idx in enumerate(bm25_ranks[:k]):
            scores[idx] = scores.get(idx, 0) + rrf(i)
        for i, idx in enumerate(dense_ranks[:k]):
            scores[idx] = scores.get(idx, 0) + rrf(i)
            
        # Get top k results
        top_indices = sorted(scores, key=scores.get, reverse=True)[:k]
        
        return [self.documents[idx] for idx in top_indices]
```

---

# Conclusion

This comprehensive blueprint provides everything needed to build a championship-level RAG system for the WattBot2025 competition. The key to success lies in:

1. **Robust document processing** that handles complex PDF structures
2. **Intelligent chunking** that preserves semantic meaning
3. **Hybrid retrieval** combining multiple search strategies
4. **Knowledge graph integration** for relationship understanding
5. **Sophisticated answer generation** with accurate citations
6. **Continuous evaluation and monitoring** for improvement

Remember: Every component is designed to handle the specific challenges of environmental AI research papers, with multiple fallbacks and innovations to ensure top performance.

---
