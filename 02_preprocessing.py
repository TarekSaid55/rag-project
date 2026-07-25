"""
02_preprocessing.py
--------------------
Step 2 of the RAG pipeline.

Reads data/documents.json (from step 1), cleans the text of each
document, and saves the result to data/documents_clean.json

Run:
    python 02_preprocessing.py
"""

import os
import json
import re

INPUT_FILE = os.path.join("data", "documents.json")
OUTPUT_FILE = os.path.join("data", "documents_clean.json")


def clean_text(text):
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove control characters (keep newlines/tabs)
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u0600-\u06FF\u0750-\u077F]", " ", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse repeated spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"'{INPUT_FILE}' not found. Run 01_documents.py first."
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    cleaned = []
    for doc in documents:
        cleaned_text = clean_text(doc["text"])
        if cleaned_text:
            cleaned.append({"source": doc["source"], "text": cleaned_text})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"Cleaned {len(cleaned)} documents -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
