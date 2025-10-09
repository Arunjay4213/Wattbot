import os
import csv
import re
import requests
from pathlib import Path
from PyPDF2 import PdfReader
import camelot

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("data/processed")
METADATA_PATH = RAW_DIR / "metadata.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def download_pdf(pdf_id, url):
    """Download a PDF from a URL and save it to data/raw."""
    pdf_path = RAW_DIR / f"{pdf_id}.pdf"
    if pdf_path.exists():
        print(f"✅ {pdf_id}.pdf already exists, skipping download.")
        return pdf_path

    print(f"⬇️ Downloading {pdf_id} from {url} ...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an error for bad status codes

        with open(pdf_path, "wb") as f:
            f.write(response.content)

        print(f"📄 Saved to {pdf_path}")
        return pdf_path

    except Exception as e:
        print(f"❌ Failed to download {pdf_id} from {url}: {e}")
        return None



def extract_text(pdf_path):
    """Extract text from a PDF using PyPDF2."""
    reader = PdfReader(str(pdf_path))
    all_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            all_text.append(text)
        except Exception as e:
            print(f"⚠️ Error extracting text from page {i+1} of {pdf_path.name}: {e}")
    return "\n".join(all_text)

def extract_tables(pdf_path):
    """Extract tables using Camelot (lattice + stream)."""
    tables_text = []

    # Try lattice first
    try:
        tables_lattice = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
        for table in tables_lattice:
            tables_text.append(table.df.to_string())
    except Exception as e:
        print(f"⚠️ Lattice extraction failed for {pdf_path.name}: {e}")

    # Fallback to stream
    try:
        tables_stream = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
        for table in tables_stream:
            tables_text.append(table.df.to_string())
    except Exception as e:
        print(f"⚠️ Stream extraction failed for {pdf_path.name}: {e}")

    return "\n".join(tables_text)


def clean_text(text):
    """Basic cleaning for whitespace, newlines, etc."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def process_pdf(pdf_id, url):
    pdf_path = download_pdf(pdf_id, url)
    if pdf_path is None:
        return  # Skip processing if download failed

    text = extract_text(pdf_path)
    tables = extract_tables(pdf_path)
    combined = clean_text(text + "\n" + tables)

    out_path = PROCESSED_DIR / f"{pdf_id}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"📝 Processed {pdf_id} → saved to {out_path}")

def main():
    with open("data/raw/metadata.csv", "r", encoding="latin-1") as f:
        reader = csv.DictReader(f)

        for row in reader:
            pdf_id = row["id"].strip()
            url = row["url"].strip()  # remove leading/trailing spaces
            process_pdf(pdf_id, url)


if __name__ == "__main__":
    main()
