"""Document chunking module for PDFs (text + tables) with JSON export."""

import re
import json
from pathlib import Path
from PyPDF2 import PdfReader
import camelot


class DocumentChunker:
    def __init__(self, chunk_size=1000, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    # -----------------------------
    # PDF PARSING
    # -----------------------------
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from all pages in a PDF."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n\n"
        return text.strip()

    def extract_tables_from_pdf(self, pdf_path):
        """Extract tables using Camelot (both lattice and stream)."""
        tables = []
        for flavor in ["lattice", "stream"]:
            try:
                table_list = camelot.read_pdf(pdf_path, flavor=flavor, pages="all")
                for t in list(table_list):
                    tables.append(t.df.to_csv(index=False))
            except Exception as e:
                print(f"⚠️ {flavor} extraction failed for {pdf_path}: {e}")
        return tables


    # -----------------------------
    # CHUNKING LOGIC
    # -----------------------------
    def chunk_text(self, text):
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        return chunks

    def chunk_pdf(self, pdf_path):
        """Process one PDF into structured text/table chunks."""
        print(f"📄 Processing {pdf_path} ...")

        text = self.extract_text_from_pdf(pdf_path)
        tables = self.extract_tables_from_pdf(pdf_path)
        chunks = []

        # Text chunks
        for chunk in self.chunk_text(text):
            chunks.append({"type": "text", "content": chunk})

        # Table chunks
        for table in tables:
            for chunk in self.chunk_text(table):
                chunks.append({"type": "table", "content": chunk})

        return chunks

    def save_chunks_to_json(self, chunks, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    input_dir = Path("data/raw")           # Directory with your downloaded PDFs
    output_dir = Path("data/chunks")

    chunker = DocumentChunker(chunk_size=800, overlap=100)

    for pdf_path in input_dir.glob("*.pdf"):
        chunks = chunker.chunk_pdf(str(pdf_path))
        json_path = output_dir / f"{pdf_path.stem}_chunks.json"
        chunker.save_chunks_to_json(chunks, json_path)
