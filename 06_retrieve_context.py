"""
06_retrieve_context.py
------------------------
Step 6 of the RAG pipeline.

Given a user query, embeds it with the same embedding model used in
step 4, then retrieves the most relevant chunks from the Chroma
collection created in step 5.

This module is imported by 07_prompting.py / streamlit_app.py — it is
not meant to be run directly (though you can, for a quick test).
"""

import os
import chromadb

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model = None
_collection = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_collection():
    global _collection
    if _collection is None:
        if not os.path.isdir(CHROMA_DIR):
            raise FileNotFoundError(
                f"'{CHROMA_DIR}' not found. Run 05_create_chroma_store.py first."
            )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve_context(query, top_k=4):
    """
    Returns a list of dicts:
    [{"text": "...", "source": "filename.txt", "score": 0.12}, ...]
    sorted from most to least relevant.
    """
    model = get_embedding_model()
    collection = get_collection()

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    contexts = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, distance in zip(documents, metadatas, distances):
        contexts.append(
            {
                "text": text,
                "source": meta.get("source", "unknown"),
                "score": distance,
            }
        )

    return contexts


if __name__ == "__main__":
    query = input("Test query: ")
    for i, ctx in enumerate(retrieve_context(query), start=1):
        print(f"\n--- Result {i} (source: {ctx['source']}, score: {ctx['score']:.4f}) ---")
        print(ctx["text"][:300])
