"""
01_documents.py
----------------
Step 1 of the RAG pipeline.

Reads all raw documents from data/raw/ (supports .txt and .pdf)
and saves them as a single JSON file: data/documents.json

Each document is stored as:
{
    "source": "filename.txt",
    "text": "raw text content..."
}

Run:
    python 01_documents.py
"""

import os
import json

RAW_DIR = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join("data", "documents.json")


def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is not installed. Run: pip install pypdf"
        )
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def load_documents(raw_dir=RAW_DIR):
    documents = []
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Folder '{raw_dir}' not found. Put your .txt/.pdf files there."
        )

    for filename in sorted(os.listdir(raw_dir)):
        path = os.path.join(raw_dir, filename)
        if not os.path.isfile(path):
            continue

        if filename.lower().endswith(".txt"):
            text = read_txt(path)
        elif filename.lower().endswith(".pdf"):
            text = read_pdf(path)
        else:
            continue

        if text.strip():
            documents.append({"source": filename, "text": text})

    return documents


def main():
    documents = load_documents()
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Loaded {len(documents)} documents -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
