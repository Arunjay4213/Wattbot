"""Document chunking module for PDFs (text + tables) with JSON export."""

import re
import json
from pathlib import Path
from PyPDF2 import PdfReader
import camelot


class DocumentChunker:
    def __init__(self, chunk_size=1200, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = 100

        # Patterns for detecting structure
        self.section_pattern = re.compile(
            r'^(?:\d+\.?\s*)?(?:Abstract|Introduction|Methods?|Results?|Discussion|'
            r'Conclusion|References|Appendix|Background|Related Work|Experiments?|'
            r'Evaluation|Analysis|Data|Model|Architecture|Training|Inference)',
            re.IGNORECASE | re.MULTILINE
        )

        self.numeric_pattern = re.compile(
            r'\b\d+\.?\d*\s*(?:kwh|mwh|gwh|twh|co2e?|tco2e?|'
            r'percent|%|lbs?|kg|tons?|gpu|tpu|flops?|'
            r'pue|wue|watts?|hours?|days?|years?)\b',
            re.IGNORECASE
        )

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
        """Extract tables using Camelot (both lattice and stream), return as markdown."""
        tables = []
        seen_hashes = set()

        for flavor in ["lattice", "stream"]:
            try:
                table_list = camelot.read_pdf(pdf_path, flavor=flavor, pages="all")
                for idx, t in enumerate(list(table_list)):
                    df = t.df

                    # Convert to markdown table for better LLM understanding
                    if len(df) > 1:
                        headers = df.iloc[0].tolist()
                        data_rows = df.iloc[1:].values.tolist()

                        md_table = "| " + " | ".join(str(h) for h in headers) + " |\n"
                        md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

                        for row in data_rows:
                            md_table += "| " + " | ".join(str(c) for c in row) + " |\n"
                    else:
                        md_table = df.to_csv(index=False)

                    # Deduplicate tables
                    content_hash = hash(md_table[:100])
                    if content_hash not in seen_hashes:
                        seen_hashes.add(content_hash)
                        tables.append({
                            "content": md_table,
                            "page": getattr(t, 'page', idx + 1),
                            "table_id": f"table_{len(tables) + 1}"
                        })

            except Exception as e:
                print(f"⚠️ {flavor} extraction failed for {pdf_path}: {e}")

        return tables

    # -----------------------------
    # CHUNKING LOGIC
    # -----------------------------
    def detect_sections(self, text):
        """Detect section headers and their positions."""
        sections = []
        for match in self.section_pattern.finditer(text):
            section_name = match.group().strip()
            start = match.start()
            sections.append((section_name, start, match.end()))
        return sections

    def split_by_sentences(self, text):
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text, doc_id=None):
        """Split text into overlapping chunks with section awareness."""
        chunks = []

        # Detect sections
        sections = self.detect_sections(text)

        # If no sections found, treat as single section
        if not sections:
            sections = [("content", 0, 0)]

        # Build section ranges
        section_ranges = []
        for i, (name, start, _) in enumerate(sections):
            end = sections[i + 1][1] if i + 1 < len(sections) else len(text)
            section_ranges.append((name, start, end))

        # Process each section
        for section_name, start, end in section_ranges:
            section_text = text[start:end].strip()

            if len(section_text) < self.min_chunk_size:
                continue

            # Split by paragraphs
            paragraphs = re.split(r'\n\s*\n', section_text)

            current_chunk = ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # If adding paragraph exceeds chunk size, save current and start new
                if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                    contains_numeric = bool(self.numeric_pattern.search(current_chunk))

                    chunks.append({
                        "type": "numeric" if contains_numeric else "text",
                        "content": current_chunk.strip(),
                        "section": section_name,
                        "contains_numeric": contains_numeric
                    })

                    # Start new chunk with overlap from last sentences
                    sentences = self.split_by_sentences(current_chunk)
                    overlap_text = " ".join(sentences[-2:]) if len(sentences) > 2 else ""
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = current_chunk + "\n\n" + para if current_chunk else para

            # Don't forget last chunk
            if current_chunk and len(current_chunk) >= self.min_chunk_size:
                contains_numeric = bool(self.numeric_pattern.search(current_chunk))
                chunks.append({
                    "type": "numeric" if contains_numeric else "text",
                    "content": current_chunk.strip(),
                    "section": section_name,
                    "contains_numeric": contains_numeric
                })

        return chunks

    def chunk_pdf(self, pdf_path):
        """Process one PDF into structured text/table chunks."""
        print(f"📄 Processing {pdf_path} ...")

        doc_id = Path(pdf_path).stem

        text = self.extract_text_from_pdf(pdf_path)
        tables = self.extract_tables_from_pdf(pdf_path)
        chunks = []

        # Text chunks with section awareness
        text_chunks = self.chunk_text(text, doc_id)
        for chunk in text_chunks:
            chunks.append(chunk)

        # Table chunks - keep tables intact when possible
        for table in tables:
            table_content = table["content"]
            contains_numeric = bool(self.numeric_pattern.search(table_content))

            # If table is too large, chunk it
            if len(table_content) > self.chunk_size * 1.5:
                for chunk in self._chunk_large_table(table_content):
                    chunks.append({
                        "type": "table",
                        "content": f"[TABLE {table['table_id']}]\n{chunk}\n[/TABLE]",
                        "section": f"Table (Page {table['page']})",
                        "contains_numeric": contains_numeric
                    })
            else:
                chunks.append({
                    "type": "table",
                    "content": f"[TABLE {table['table_id']}]\n{table_content}\n[/TABLE]",
                    "section": f"Table (Page {table['page']})",
                    "contains_numeric": contains_numeric
                })

        print(f"   → {len(chunks)} chunks ({sum(1 for c in chunks if c['type'] == 'table')} tables)")
        return chunks

    def _chunk_large_table(self, table_content):
        """Split large tables while preserving header."""
        lines = table_content.strip().split('\n')
        if len(lines) < 3:
            return [table_content]

        # Keep header (first 2 lines for markdown tables)
        header = '\n'.join(lines[:2])
        data_lines = lines[2:]

        chunks = []
        current_chunk_lines = []
        current_size = len(header)

        for line in data_lines:
            if current_size + len(line) > self.chunk_size and current_chunk_lines:
                chunks.append(header + '\n' + '\n'.join(current_chunk_lines))
                current_chunk_lines = []
                current_size = len(header)

            current_chunk_lines.append(line)
            current_size += len(line) + 1

        if current_chunk_lines:
            chunks.append(header + '\n' + '\n'.join(current_chunk_lines))

        return chunks

    def save_chunks_to_json(self, chunks, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    input_dir = Path("data/raw")
    output_dir = Path("data/chunks")

    chunker = DocumentChunker(chunk_size=1200, overlap=200)

    for pdf_path in input_dir.glob("*.pdf"):
        chunks = chunker.chunk_pdf(str(pdf_path))
        json_path = output_dir / f"{pdf_path.stem}_chunks.json"
        chunker.save_chunks_to_json(chunks, json_path)
