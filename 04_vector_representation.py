"""
04_vector_representation.py
-----------------------------
Step 4 of the RAG pipeline.

Reads data/chunks.json (from step 3), computes a vector embedding for
each chunk using a local, free sentence-transformers model, and saves
everything to data/embeddings.json

Using a local embedding model here (instead of a paid API) keeps this
step free and reproducible. You can swap EMBEDDING_MODEL for any other
sentence-transformers model if you like.

Run:
    python 04_vector_representation.py
"""

import os
import json

INPUT_FILE = os.path.join("data", "chunks.json")
OUTPUT_FILE = os.path.join("data", "embeddings.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"'{INPUT_FILE}' not found. Run 03_chunking.py first."
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks found. Nothing to embed.")
        return

    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"Embedded {len(chunks)} chunks -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
