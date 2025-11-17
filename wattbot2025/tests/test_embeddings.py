import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.embeddings import EmbeddingRetriever, ScientificPaperRetriever, Chunk


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_chunks():
    """Create diverse test chunks based on training data"""
    return [
        # Numeric fact chunk (from strubell2019)
        Chunk(
            text="Training BERT-base produced approximately 1,438 lbs of CO2 emissions. "
                 "This is equivalent to a trans-American flight for one person. "
                 "The model was trained on 64 V100 GPUs for 79 hours.",
            chunk_id="strubell2019_chunk_1",
            doc_id="strubell2019",
            page_num=3,
            section="Results",
            type="text",
            contains_numeric=True,
            contains_table=False,
            metadata={"entities": ["BERT-base", "V100"], "metrics": ["CO2", "lbs"]}
        ),
        
        # Table-based chunk (from patterson2021)
        Chunk(
            text="Table 2: Energy Consumption of Large Models\n\n"
                 "Model | Energy (MWh) | CO2 (tCO2e) | Hardware\n"
                 "GPT-3 | 1287 | 502 | V100\n"
                 "BERT-base | 0.96 | 1.4 | V100\n"
                 "T5-11B | 86 | 47 | TPU v3\n\n"
                 "Note: GPT-3 training consumed significantly more energy than other models.",
            chunk_id="patterson2021_chunk_23",
            doc_id="patterson2021",
            page_num=5,
            section="Table 2: Model Emissions",
            type="table",
            contains_numeric=True,
            contains_table=True,
            metadata={"table_structure": "preserved", "models": ["GPT-3", "BERT-base", "T5-11B"]}
        ),
        
        # Comparison chunk (from wu2022)
        Chunk(
            text="Google's Iowa datacenter achieved a PUE of 1.11 in 2021, representing "
                 "best-in-class efficiency. In contrast, the US national average PUE in 2020 "
                 "was 1.59. PUE (Power Usage Effectiveness) measures total facility energy "
                 "divided by IT equipment energy, where lower is better.",
            chunk_id="wu2022_chunk_45",
            doc_id="wu2022",
            page_num=8,
            section="Data Center Efficiency",
            type="text",
            contains_numeric=True,
            contains_table=False,
            metadata={"comparison": True, "entities": ["Google", "Iowa datacenter"], "metrics": ["PUE"]}
        ),
        
        # True/False context chunk (from schwartz2019)
        Chunk(
            text="A large majority of the papers target accuracy (90% of ACL papers, "
                 "80% of NeurIPS papers and 75% of CVPR papers). Moreover, for both "
                 "empirical AI conferences (ACL and CVPR) only a small portion (10% and 20% "
                 "respectively) argue for a new efficiency result.",
            chunk_id="schwartz2019_chunk_12",
            doc_id="schwartz2019",
            page_num=4,
            section="Analysis",
            type="text",
            contains_numeric=True,
            contains_table=False,
            metadata={"percentages": ["90%", "80%", "75%", "10%", "20%"]}
        ),
        
        # Definition/conceptual chunk (from li2025b)
        Chunk(
            text="Water consumption is defined as water withdrawal minus water discharge, "
                 "and means the amount of water evaporated, transpired, incorporated into "
                 "products or crops, or otherwise removed from the immediate water environment. "
                 "This is distinct from water withdrawal, which may be returned to the source.",
            chunk_id="li2025b_chunk_8",
            doc_id="li2025b",
            page_num=2,
            section="Methodology",
            type="text",
            contains_numeric=False,
            contains_table=False,
            metadata={"definition": "water consumption"}
        ),
        
        # Multi-document synthesis chunk (from chung2025)
        Chunk(
            text="We present the ML.ENERGY Benchmark, a benchmark suite and tool for "
                 "measuring inference energy consumption under realistic service environments. "
                 "Recent estimates suggest inference can account for up to 90% of a model's "
                 "total lifecycle energy use.",
            chunk_id="chung2025_chunk_5",
            doc_id="chung2025",
            page_num=1,
            section="Introduction",
            type="text",
            contains_numeric=True,
            contains_table=False,
            metadata={"benchmark_name": "ML.ENERGY Benchmark", "focus": "inference"}
        ),
        
        # Calculation-based chunk (from dodge2022)
        Chunk(
            text="The 6.1B parameter model consumed 13.8 MWh during training. "
                 "Given that average U.S. household consumption is approximately 10,715 kWh/yr "
                 "(or 10.7 MWh/yr), this is equivalent to 1.3 household-years of electricity consumption.",
            chunk_id="dodge2022_chunk_30",
            doc_id="dodge2022",
            page_num=6,
            section="Energy Analysis",
            type="text",
            contains_numeric=True,
            contains_table=False,
            metadata={"calculation": True, "values": ["13.8 MWh", "10.7 MWh/yr", "1.3 years"]}
        ),
        
        # Hardware specification chunk (from chen2024)
        Chunk(
            text="Table 3: Large language models used for evaluation.\n"
                 "Model | Size (GB) | Min GPUs (A100 80GB)\n"
                 "LLaMA-7B | 13.5 | 1\n"
                 "LLaMA-13B | 26.0 | 1\n"
                 "LLaMA-33B | 64.7 | 1\n"
                 "LLaMA-65B | 129.4 | 2",
            chunk_id="chen2024_chunk_15",
            doc_id="chen2024",
            page_num=5,
            section="Table 3",
            type="table",
            contains_numeric=True,
            contains_table=True,
            metadata={"table_type": "hardware_requirements"}
        )
    ]


@pytest.fixture
def retriever(temp_cache_dir):
    """Create a basic retriever instance"""
    return EmbeddingRetriever(
        model_name="BAAI/bge-large-en-v1.5",
        cache_dir=temp_cache_dir,
        device="cpu"
    )


@pytest.fixture
def scientific_retriever(temp_cache_dir):
    """Create a scientific paper retriever instance"""
    return ScientificPaperRetriever(cache_dir=temp_cache_dir)


class TestChunkDataclass:
    """Test the Chunk dataclass structure"""
    
    def test_chunk_creation_minimal(self):
        """Test creating a chunk with minimal required fields"""
        chunk = Chunk(
            text="Test text",
            chunk_id="test_1",
            doc_id="doc_1"
        )
        assert chunk.text == "Test text"
        assert chunk.chunk_id == "test_1"
        assert chunk.doc_id == "doc_1"
        assert chunk.type == "text"
        assert chunk.contains_numeric is False
        assert chunk.contains_table is False
    
    def test_chunk_creation_full(self):
        """Test creating a chunk with all fields"""
        chunk = Chunk(
            text="Training BERT produces 1438 lbs CO2",
            chunk_id="test_1",
            doc_id="strubell2019",
            page_num=3,
            section="Results",
            type="text",
            contains_numeric=True,
            contains_table=False,
            metadata={"source": "paper"}
        )
        assert chunk.page_num == 3
        assert chunk.section == "Results"
        assert chunk.contains_numeric is True
        assert chunk.metadata["source"] == "paper"
    
    def test_chunk_table_type(self):
        """Test chunk with table type"""
        chunk = Chunk(
            text="Model | Energy\nBERT | 0.96",
            chunk_id="table_1",
            doc_id="test",
            type="table",
            contains_table=True
        )
        assert chunk.type == "table"
        assert chunk.contains_table is True


class TestEmbeddingRetrieverBasics:
    """Test basic retriever functionality"""
    
    def test_retriever_initialization(self, retriever):
        """Test retriever initializes correctly"""
        assert retriever.model_name == "BAAI/bge-large-en-v1.5"
        assert retriever.device in ["cpu", "cuda"]
        assert retriever.chunks == []
        assert retriever.embeddings is None
    
    def test_cache_directory_creation(self, temp_cache_dir):
        """Test that cache directory is created"""
        retriever = EmbeddingRetriever(cache_dir=temp_cache_dir)
        assert Path(temp_cache_dir).exists()
    
    def test_bge_instruction_detection(self, temp_cache_dir):
        """Test BGE model instruction prefix detection"""
        bge_retriever = EmbeddingRetriever(
            model_name="BAAI/bge-large-en-v1.5",
            cache_dir=temp_cache_dir
        )
        assert bge_retriever.use_instruction is True
        assert "Represent" in bge_retriever.query_instruction


class TestIndexing:
    """Test document indexing functionality"""
    
    def test_index_single_chunk(self, retriever):
        """Test indexing a single chunk"""
        chunks = [Chunk(text="Test", chunk_id="1", doc_id="doc1")]
        retriever.index_documents(chunks, batch_size=1)
        
        assert len(retriever.chunks) == 1
        assert retriever.embeddings is not None
        assert retriever.embeddings.shape[0] == 1
        assert "1" in retriever.chunk_map
    
    def test_index_multiple_chunks(self, retriever, sample_chunks):
        """Test indexing multiple diverse chunks"""
        retriever.index_documents(sample_chunks, batch_size=4)
        
        assert len(retriever.chunks) == len(sample_chunks)
        assert retriever.embeddings.shape[0] == len(sample_chunks)
        assert len(retriever.chunk_map) == len(sample_chunks)
    
    def test_chunk_map_consistency(self, retriever, sample_chunks):
        """Test that chunk_map correctly maps IDs to indices"""
        retriever.index_documents(sample_chunks)
        
        for i, chunk in enumerate(sample_chunks):
            assert retriever.chunk_map[chunk.chunk_id] == i
            assert retriever.chunks[i].chunk_id == chunk.chunk_id
    
    def test_embedding_normalization(self, retriever, sample_chunks):
        """Test that embeddings are normalized"""
        retriever.index_documents(sample_chunks)
        
        # Check that embeddings have unit norm (normalized)
        norms = np.linalg.norm(retriever.embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(sample_chunks)), decimal=5)


class TestSearch:
    """Test search functionality"""
    
    def test_search_numeric_query(self, retriever, sample_chunks):
        """Test searching for numeric facts"""
        retriever.index_documents(sample_chunks)
        
        results = retriever.search("What is the CO2 emission of BERT?", top_k=3)
        
        assert len(results) > 0
        assert all(isinstance(chunk, Chunk) for chunk, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        
        # BERT emission chunk should be in top results
        top_docs = [chunk.doc_id for chunk, _ in results]
        assert "strubell2019" in top_docs or "patterson2021" in top_docs
    
    def test_search_table_query(self, retriever, sample_chunks):
        """Test searching for table-based information"""
        retriever.index_documents(sample_chunks)
        
        results = retriever.search("energy consumption comparison of models", top_k=3)
        
        # Should find the table chunk
        table_chunks = [chunk for chunk, _ in results if chunk.contains_table]
        assert len(table_chunks) > 0
    
    def test_search_definition_query(self, retriever, sample_chunks):
        """Test searching for definitions"""
        retriever.index_documents(sample_chunks)
        
        results = retriever.search("What is water consumption?", top_k=3)
        
        # Should find the definition chunk
        top_chunk_ids = [chunk.chunk_id for chunk, _ in results]
        assert "li2025b_chunk_8" in top_chunk_ids[:3]
    
    def test_search_comparison_query(self, retriever, sample_chunks):
        """Test searching for comparisons"""
        retriever.index_documents(sample_chunks)
        
        results = retriever.search("Compare PUE of Google datacenter to national average", top_k=3)
        
        # Should find the comparison chunk
        assert any(chunk.doc_id == "wu2022" for chunk, _ in results)
    
    def test_search_top_k(self, retriever, sample_chunks):
        """Test that top_k parameter works correctly"""
        retriever.index_documents(sample_chunks)
        
        for k in [1, 3, 5]:
            results = retriever.search("energy consumption", top_k=k)
            assert len(results) <= k
    
    def test_search_min_score(self, retriever, sample_chunks):
        """Test minimum score filtering"""
        retriever.index_documents(sample_chunks)
        
        results = retriever.search("energy", top_k=10, min_score=0.5)
        
        # All results should have score >= 0.5
        assert all(score >= 0.5 for _, score in results)
    
    def test_search_filter_doc_ids(self, retriever, sample_chunks):
        """Test filtering by document IDs"""
        retriever.index_documents(sample_chunks)
        
        results = retriever.search(
            "CO2 emissions",
            top_k=5,
            filter_doc_ids=["strubell2019", "patterson2021"]
        )
        
        # All results should be from specified docs
        result_doc_ids = {chunk.doc_id for chunk, _ in results}
        assert result_doc_ids.issubset({"strubell2019", "patterson2021"})
    
    def test_search_empty_query(self, retriever, sample_chunks):
        """Test search with empty query"""
        retriever.index_documents(sample_chunks)
        
        results = retriever.search("", top_k=3)
        assert len(results) > 0  # Should still return results
    
    def test_search_before_indexing(self, retriever):
        """Test that search fails before indexing"""
        with pytest.raises(ValueError, match="No documents indexed"):
            retriever.search("test query")


class TestHybridSearch:
    """Test hybrid search functionality"""
    
    def test_hybrid_search_basic(self, retriever, sample_chunks):
        """Test basic hybrid search"""
        retriever.index_documents(sample_chunks)
        
        results = retriever.hybrid_search("BERT CO2 emissions", top_k=3, alpha=0.5)
        
        assert len(results) > 0
        assert len(results) <= 3
    
    def test_hybrid_search_alpha_dense(self, retriever, sample_chunks):
        """Test hybrid search favoring dense (semantic) search"""
        retriever.index_documents(sample_chunks)
        
        results = retriever.hybrid_search(
            "environmental impact of training",
            top_k=3,
            alpha=0.9  # Favor dense
        )
        
        assert len(results) > 0
    
    def test_hybrid_search_alpha_sparse(self, retriever, sample_chunks):
        """Test hybrid search favoring sparse (keyword) search"""
        retriever.index_documents(sample_chunks)
        
        results = retriever.hybrid_search(
            "BERT GPU V100",
            top_k=3,
            alpha=0.1  # Favor keywords
        )
        
        assert len(results) > 0
    
    def test_keyword_score_numeric_boost(self, retriever, sample_chunks):
        """Test that numeric chunks get boosted for quantitative queries"""
        retriever.index_documents(sample_chunks)
        
        results = retriever.hybrid_search("how much CO2 emission", top_k=5, alpha=0.3)
        
        # Top results should contain numeric chunks
        top_3_numeric = sum(1 for chunk, _ in results[:3] if chunk.contains_numeric)
        assert top_3_numeric >= 2


class TestSimilarChunks:
    """Test finding similar chunks"""
    
    def test_find_similar_chunks(self, retriever, sample_chunks):
        """Test finding similar chunks"""
        retriever.index_documents(sample_chunks)
        
        reference_chunk_id = "strubell2019_chunk_1"
        similar = retriever.find_similar_chunks(reference_chunk_id, top_k=3)
        
        assert len(similar) > 0
        assert len(similar) <= 3
        assert all(chunk.chunk_id != reference_chunk_id for chunk, _ in similar)
    
    def test_find_similar_chunks_invalid_id(self, retriever, sample_chunks):
        """Test error handling for invalid chunk ID"""
        retriever.index_documents(sample_chunks)
        
        with pytest.raises(ValueError, match="not found in index"):
            retriever.find_similar_chunks("nonexistent_chunk", top_k=3)
    
    def test_similar_chunks_score_order(self, retriever, sample_chunks):
        """Test that similar chunks are ordered by score"""
        retriever.index_documents(sample_chunks)
        
        similar = retriever.find_similar_chunks("patterson2021_chunk_23", top_k=5)
        
        # Scores should be in descending order
        scores = [score for _, score in similar]
        assert scores == sorted(scores, reverse=True)


class TestSaveLoad:
    """Test index persistence"""
    
    def test_save_and_load_index(self, retriever, sample_chunks, temp_cache_dir):
        """Test saving and loading index"""
        retriever.index_documents(sample_chunks)
        
        index_path = Path(temp_cache_dir) / "test_index.pkl"
        retriever.save_index(str(index_path))
        
        assert index_path.exists()
        
        # Create new retriever and load
        new_retriever = EmbeddingRetriever(cache_dir=temp_cache_dir)
        new_retriever.load_index(str(index_path))
        
        assert len(new_retriever.chunks) == len(sample_chunks)
        assert new_retriever.embeddings.shape == retriever.embeddings.shape
        assert new_retriever.chunk_map == retriever.chunk_map
    
    def test_load_preserves_search(self, retriever, sample_chunks, temp_cache_dir):
        """Test that loaded index can perform searches"""
        retriever.index_documents(sample_chunks)
        
        index_path = Path(temp_cache_dir) / "test_index.pkl"
        retriever.save_index(str(index_path))
        
        # Use same model as original retriever
        new_retriever = EmbeddingRetriever(
            model_name="BAAI/bge-large-en-v1.5",
            cache_dir=temp_cache_dir
        )
        new_retriever.load_index(str(index_path))
        
        results = new_retriever.search("BERT emissions", top_k=3)
        assert len(results) > 0


class TestEmbeddingCache:
    """Test embedding caching functionality"""
    
    def test_cache_saves_embeddings(self, retriever, sample_chunks):
        """Test that embeddings are cached"""
        retriever.index_documents(sample_chunks)
        
        # Check that cache files were created
        cache_files = list(retriever.cache_dir.glob("*.npy"))
        assert len(cache_files) > 0
    
    def test_cache_loads_embeddings(self, temp_cache_dir, sample_chunks):
        """Test that cached embeddings are loaded"""
        # First run - creates cache
        retriever1 = EmbeddingRetriever(
            model_name="BAAI/bge-large-en-v1.5",
            cache_dir=temp_cache_dir
        )
        retriever1.index_documents(sample_chunks)
        
        # Second run - should load from cache
        retriever2 = EmbeddingRetriever(
            model_name="BAAI/bge-large-en-v1.5",
            cache_dir=temp_cache_dir
        )
        retriever2.index_documents(sample_chunks)
        
        # Embeddings should be identical
        np.testing.assert_array_equal(retriever1.embeddings, retriever2.embeddings)


class TestScientificPaperRetriever:
    """Test specialized scientific paper retriever"""
    
    def test_scientific_retriever_initialization(self, scientific_retriever):
        """Test scientific retriever initialization"""
        assert "scibert" in scientific_retriever.model_name.lower()
        assert len(scientific_retriever.boost_keywords) > 0
    
    def test_scientific_retriever_keyword_boost(self, scientific_retriever, sample_chunks):
        """Test that domain keywords boost results"""
        scientific_retriever.index_documents(sample_chunks)
        
        # Query with domain-specific terms
        results = scientific_retriever.search("CO2 emissions energy consumption", top_k=5)
        
        # Results should contain numeric/technical chunks
        numeric_count = sum(1 for chunk, _ in results if chunk.contains_numeric)
        assert numeric_count >= 3
    
    def test_scientific_retriever_numeric_boost(self, scientific_retriever, sample_chunks):
        """Test boost for numeric chunks on quantitative queries"""
        scientific_retriever.index_documents(sample_chunks)
        
        results = scientific_retriever.search("how much energy does GPT-3 consume", top_k=3)
        
        # Top result should be numeric
        assert results[0][0].contains_numeric is True


class TestRealQuestionPatterns:
    """Test retrieval patterns matching actual train_QA.csv questions"""
    
    def test_true_false_question(self, retriever, sample_chunks):
        """Test True/False questions like q075"""
        retriever.index_documents(sample_chunks)
        
        # q075: "True or False: Hyperscale data centers in 2020 achieved 
        #        more than 40% higher efficiency..."
        results = retriever.search(
            "hyperscale data centers efficiency 2020 traditional comparison",
            top_k=3
        )
        
        # Should find wu2022 or patterson2021 chunks about PUE/efficiency
        assert len(results) > 0
        assert any(chunk.contains_numeric for chunk, _ in results)
    
    def test_calculation_question(self, retriever, sample_chunks):
        """Test calculation questions like q091"""
        retriever.index_documents(sample_chunks)
        
        # q091: "difference between the percentage of CVPR papers 
        #        that target accuracy and efficiency"
        results = retriever.search(
            "CVPR papers accuracy efficiency percentage",
            top_k=3
        )
        
        # Should find schwartz2019 chunk with percentages
        assert any(chunk.doc_id == "schwartz2019" for chunk, _ in results)
    
    def test_numeric_fact_question(self, retriever, sample_chunks):
        """Test numeric fact questions like q170"""
        retriever.index_documents(sample_chunks)
        
        # q170: "How many days of CO₂ emissions from an average American life 
        #        are equivalent to training BERT base?"
        results = retriever.search(
            "BERT training CO2 emissions days American life",
            top_k=3
        )
        
        # Should find strubell2019 chunk
        top_doc_ids = [chunk.doc_id for chunk, _ in results[:2]]
        assert "strubell2019" in top_doc_ids
    
    def test_table_lookup_question(self, retriever, sample_chunks):
        """Test table-based questions like energy consumption lookups"""
        retriever.index_documents(sample_chunks)
        
        # Similar to questions asking for model energy consumption from tables
        results = retriever.search(
            "GPT-3 energy consumption MWh table",
            top_k=3
        )
        
        # Should find table chunks
        assert any(chunk.contains_table for chunk, _ in results[:3])
    
    def test_comparison_calculation(self, retriever, sample_chunks):
        """Test comparison questions requiring calculation"""
        retriever.index_documents(sample_chunks)
        
        # Similar to questions comparing values between models/systems
        results = retriever.search(
            "household electricity consumption model training equivalent",
            top_k=3
        )
        
        # Should find dodge2022 chunk with household comparison
        assert any(chunk.doc_id == "dodge2022" for chunk, _ in results)
    
    def test_named_entity_question(self, retriever, sample_chunks):
        """Test questions asking for names/identifiers"""
        retriever.index_documents(sample_chunks)
        
        # Similar to q003 asking for benchmark name
        results = retriever.search(
            "benchmark measuring inference energy consumption",
            top_k=3
        )
        
        assert len(results) > 0
    
    def test_hardware_specification(self, retriever, sample_chunks):
        """Test questions about hardware specs"""
        retriever.index_documents(sample_chunks)
        
        # Questions about GPU requirements, model sizes, etc.
        results = retriever.search(
            "LLaMA model size gigabytes GPU requirements",
            top_k=3
        )
        
        # Should find chen2024 hardware table
        assert any(chunk.doc_id == "chen2024" for chunk, _ in results)
    """Test retrieval across different chunk types"""
    
    def test_retrieve_numeric_chunks(self, retriever, sample_chunks):
        """Test retrieving chunks with numeric content"""
        retriever.index_documents(sample_chunks)
        
        numeric_chunks = [c for c in sample_chunks if c.contains_numeric]
        assert len(numeric_chunks) > 0
        
        results = retriever.search("1438 lbs CO2", top_k=5)
        assert any(chunk.contains_numeric for chunk, _ in results)
    
    def test_retrieve_table_chunks(self, retriever, sample_chunks):
        """Test retrieving table chunks"""
        retriever.index_documents(sample_chunks)
        
        table_chunks = [c for c in sample_chunks if c.contains_table]
        assert len(table_chunks) > 0
        
        results = retriever.search("table energy consumption models", top_k=5)
        assert any(chunk.contains_table for chunk, _ in results)
    
    def test_retrieve_named_entities(self, retriever, sample_chunks):
        """Test retrieving chunks with named entities (like q003 asking for benchmark name)"""
        retriever.index_documents(sample_chunks)
        
        # Similar to q003: "What is the name of the benchmark suite..."
        results = retriever.search("benchmark suite measuring inference energy consumption", top_k=3)
        
        # Should find the chung2025 chunk about ML.ENERGY Benchmark
        assert any(chunk.doc_id == "chung2025" for chunk, _ in results)
    
    def test_retrieve_from_multiple_documents(self, retriever, sample_chunks):
        """Test that search can retrieve from multiple documents"""
        retriever.index_documents(sample_chunks)
        
        results = retriever.search("energy efficiency", top_k=5)
        
        # Should get results from multiple docs
        doc_ids = {chunk.doc_id for chunk, _ in results}
        assert len(doc_ids) > 1


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_chunk_list(self, retriever):
        """Test indexing empty chunk list"""
        retriever.index_documents([])
        assert len(retriever.chunks) == 0
        assert retriever.embeddings is not None
        assert retriever.embeddings.shape[0] == 0
    
    def test_very_long_text(self, retriever):
        """Test chunk with very long text"""
        long_text = "energy " * 1000
        chunk = Chunk(text=long_text, chunk_id="long", doc_id="test")
        
        retriever.index_documents([chunk])
        results = retriever.search("energy", top_k=1)
        
        assert len(results) == 1
    
    def test_special_characters(self, retriever):
        """Test chunk with special characters"""
        special_text = "CO₂ emissions: 1,438 lbs (≈ 652 kg)"
        chunk = Chunk(text=special_text, chunk_id="special", doc_id="test")
        
        retriever.index_documents([chunk])
        results = retriever.search("CO2 emissions", top_k=1)
        
        assert len(results) == 1
    
    def test_duplicate_chunk_ids(self, retriever):
        """Test handling of duplicate chunk IDs"""
        chunks = [
            Chunk(text="First", chunk_id="dup", doc_id="doc1"),
            Chunk(text="Second", chunk_id="dup", doc_id="doc2")
        ]
        
        retriever.index_documents(chunks)
        
        # Last chunk should be in the map (overwrites)
        assert "dup" in retriever.chunk_map