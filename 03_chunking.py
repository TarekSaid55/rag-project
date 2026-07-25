"""
03_chunking.py
---------------
Step 3 of the RAG pipeline.

Reads data/documents_clean.json (from step 2), splits each document
into overlapping word-based chunks, and saves them to data/chunks.json

Each chunk is stored as:
{
    "chunk_id": "filename.txt::0",
    "source": "filename.txt",
    "text": "chunk text..."
}

Run:
    python 03_chunking.py
"""

import os
import json

INPUT_FILE = os.path.join("data", "documents_clean.json")
OUTPUT_FILE = os.path.join("data", "chunks.json")

CHUNK_SIZE = 200   # words per chunk
CHUNK_OVERLAP = 50  # overlapping words between consecutive chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap  # move forward with overlap

    return chunks


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"'{INPUT_FILE}' not found. Run 02_preprocessing.py first."
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"])
        for i, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['source']}::{i}",
                    "source": doc["source"],
                    "text": piece,
                }
            )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Created {len(all_chunks)} chunks -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
