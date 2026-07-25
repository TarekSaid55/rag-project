"""
05_create_chroma_store.py
---------------------------
Step 5 of the RAG pipeline.

Reads data/embeddings.json (from step 4) and stores the chunks +
embeddings in a persistent ChromaDB collection on disk (./chroma_db).

This only needs to be run once (or whenever your source documents
change). The Streamlit app just reads from this store afterwards.

Run:
    python 05_create_chroma_store.py
"""

import os
import json
import chromadb

INPUT_FILE = os.path.join("data", "embeddings.json")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"'{INPUT_FILE}' not found. Run 04_vector_representation.py first."
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("No embedded chunks found. Nothing to store.")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Start fresh each time this script runs
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    # Chroma has a batch size limit, so insert in batches
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"Stored {len(ids)} chunks in Chroma collection '{COLLECTION_NAME}' at ./{CHROMA_DIR}")


if __name__ == "__main__":
    main()
