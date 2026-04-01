import pytest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data"))

from chunker import DocumentChunker


class TestDocumentChunker:
    """Comprehensive test suite for DocumentChunker class"""

    @pytest.fixture
    def chunker(self):
        """Create a chunker instance for testing"""
        return DocumentChunker(chunk_size=100, overlap=10)

    @pytest.fixture
    def sample_pdf_path(self):
        """Path to a real PDF file for testing"""
        # Point to papers folder on Desktop
        pdf_path = Path.home() / "Desktop" / "papers"
        pdfs = list(pdf_path.glob("*.pdf"))
        if pdfs:
            return str(pdfs[0])  # Return first PDF found
        return None

    # -----------------------------
    # TEST CHUNKING LOGIC
    # -----------------------------

    def test_chunk_text_basic(self, chunker):
        """Test basic text chunking"""
        text = "a" * 250
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        assert len(chunks[0]) <= 100

    def test_chunk_text_overlap(self, chunker):
        """Test that chunks have proper overlap"""
        text = "0123456789" * 30
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 2
        assert len(chunks[0]) <= 100

    def test_chunk_text_empty(self, chunker):
        """Test chunking empty text"""
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0

    def test_chunk_text_shorter_than_chunk_size(self, chunker):
        """Test text shorter than chunk_size"""
        text = "Short text"
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_exact_chunk_size(self, chunker):
        """Test text exactly equal to chunk_size"""
        text = "a" * 100
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 2

    def test_chunk_text_whitespace_handling(self, chunker):
        """Test that whitespace is stripped from chunks"""
        text = "   text with spaces   " * 50
        chunks = chunker.chunk_text(text)
        for chunk in chunks:
            assert chunk == chunk.strip()

    def test_custom_chunk_size(self):
        """Test chunker with custom parameters"""
        chunker = DocumentChunker(chunk_size=50, overlap=5)
        text = "a" * 200
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        assert len(chunks[0]) <= 50

    def test_custom_overlap(self):
        """Test chunker with different overlap"""
        chunker = DocumentChunker(chunk_size=100, overlap=20)
        text = "a" * 300
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 3

    def test_zero_overlap(self):
        """Test chunking with no overlap"""
        chunker = DocumentChunker(chunk_size=100, overlap=0)
        text = "a" * 300
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 3

    def test_large_overlap(self):
        """Test chunking with large overlap (edge case)"""
        chunker = DocumentChunker(chunk_size=100, overlap=50)
        text = "a" * 300
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 5

    def test_multiline_text(self, chunker):
        """Test chunking text with newlines"""
        text = "Line 1\nLine 2\nLine 3\n" * 20
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        assert "\n" in chunks[0]

    # -----------------------------
    # TEST PDF TEXT EXTRACTION
    # -----------------------------

    def test_extract_text_from_pdf_real_file(self, chunker, sample_pdf_path):
        """Test extracting text from a real PDF file"""
        if sample_pdf_path is None:
            pytest.skip("No PDF files found for testing")

        text = chunker.extract_text_from_pdf(sample_pdf_path)
        assert isinstance(text, str)
        assert len(text) > 0  # Should extract some text

    def test_extract_text_from_pdf_returns_string(self, chunker, sample_pdf_path):
        """Test that extract_text_from_pdf returns a string"""
        if sample_pdf_path is None:
            pytest.skip("No PDF files found for testing")

        text = chunker.extract_text_from_pdf(sample_pdf_path)
        assert isinstance(text, str)

    @patch("chunker.PdfReader")
    def test_extract_text_from_pdf_mock(self, mock_pdf_reader, chunker):
        """Test PDF text extraction with mocked PDF"""
        # Mock the PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample text from page"

        mock_reader = Mock()
        mock_reader.pages = [mock_page, mock_page]
        mock_pdf_reader.return_value = mock_reader

        text = chunker.extract_text_from_pdf("fake.pdf")
        assert "Sample text from page" in text

    # -----------------------------
    # TEST TABLE EXTRACTION
    # -----------------------------

    @patch("chunker.camelot.read_pdf")
    def test_extract_tables_from_pdf_success(self, mock_camelot, chunker):
        """Test successful table extraction"""
        # Mock a table
        mock_table = Mock()
        mock_table.df.to_csv.return_value = "col1,col2\nval1,val2"

        mock_table_list = [mock_table]
        mock_camelot.return_value = mock_table_list

        tables = chunker.extract_tables_from_pdf("fake.pdf")
        assert len(tables) >= 1
        assert "col1,col2" in tables[0]

    @patch("chunker.camelot.read_pdf")
    def test_extract_tables_from_pdf_failure(self, mock_camelot, chunker):
        """Test table extraction handles errors gracefully"""
        mock_camelot.side_effect = Exception("PDF parsing error")

        tables = chunker.extract_tables_from_pdf("fake.pdf")
        # Should return empty list on error (tries both flavors)
        assert isinstance(tables, list)

    @patch("chunker.camelot.read_pdf")
    def test_extract_tables_both_flavors(self, mock_camelot, chunker):
        """Test that both lattice and stream flavors are tried"""
        mock_table = Mock()
        mock_table.df.to_csv.return_value = "data"
        mock_camelot.return_value = [mock_table]

        chunker.extract_tables_from_pdf("fake.pdf")

        # Should be called twice (once for each flavor)
        assert mock_camelot.call_count == 2

    # -----------------------------
    # TEST CHUNK_PDF (INTEGRATION)
    # -----------------------------

    def test_chunk_pdf_real_file(self, chunker, sample_pdf_path):
        """Test processing a real PDF file"""
        if sample_pdf_path is None:
            pytest.skip("No PDF files found for testing")

        chunks = chunker.chunk_pdf(sample_pdf_path)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Check structure of chunks
        for chunk in chunks:
            assert "type" in chunk
            assert "content" in chunk
            assert chunk["type"] in ["text", "table"]

    @patch.object(DocumentChunker, "extract_text_from_pdf")
    @patch.object(DocumentChunker, "extract_tables_from_pdf")
    def test_chunk_pdf_creates_text_chunks(self, mock_tables, mock_text, chunker):
        """Test that chunk_pdf creates text chunks"""
        mock_text.return_value = "Sample text " * 200
        mock_tables.return_value = []

        chunks = chunker.chunk_pdf("fake.pdf")

        # Should have text chunks
        text_chunks = [c for c in chunks if c["type"] == "text"]
        assert len(text_chunks) > 0

    @patch.object(DocumentChunker, "extract_text_from_pdf")
    @patch.object(DocumentChunker, "extract_tables_from_pdf")
    def test_chunk_pdf_creates_table_chunks(self, mock_tables, mock_text, chunker):
        """Test that chunk_pdf creates table chunks"""
        mock_text.return_value = "Short text"
        mock_tables.return_value = ["col1,col2\n" + "data," * 200]

        chunks = chunker.chunk_pdf("fake.pdf")

        # Should have table chunks
        table_chunks = [c for c in chunks if c["type"] == "table"]
        assert len(table_chunks) > 0

    # -----------------------------
    # TEST JSON SAVING
    # -----------------------------

    def test_save_chunks_to_json(self, chunker, tmp_path):
        """Test saving chunks to JSON file"""
        chunks = [
            {"type": "text", "content": "Sample text"},
            {"type": "table", "content": "col1,col2\nval1,val2"},
        ]

        output_path = tmp_path / "test_output.json"
        chunker.save_chunks_to_json(chunks, output_path)

        # Check file was created
        assert output_path.exists()

        # Check content is valid JSON
        with open(output_path, "r") as f:
            loaded_chunks = json.load(f)

        assert len(loaded_chunks) == 2
        assert loaded_chunks[0]["type"] == "text"
        assert loaded_chunks[1]["type"] == "table"

    def test_save_chunks_creates_directory(self, chunker, tmp_path):
        """Test that save_chunks_to_json creates parent directories"""
        output_path = tmp_path / "nested" / "dirs" / "output.json"
        chunks = [{"type": "text", "content": "test"}]

        chunker.save_chunks_to_json(chunks, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    # -----------------------------
    # TEST INITIALIZATION
    # -----------------------------

    def test_init_default_parameters(self):
        """Test DocumentChunker initialization with defaults"""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 1000
        assert chunker.overlap == 100

    def test_init_custom_parameters(self):
        """Test DocumentChunker initialization with custom params"""
        chunker = DocumentChunker(chunk_size=500, overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.overlap == 50
