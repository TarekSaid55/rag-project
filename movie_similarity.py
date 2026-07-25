"""
movie_similarity.py
----------------------
Given a story/plot description typed by the user, ranks the movies in
the "movies" Chroma collection (built by 08_movie_documents.py and
09_movies_vector_store.py) by embedding similarity, closest first.

This is pure vector search -- no LLM call needed, so it's fast and
free. Used by streamlit_app.py for the "Find Similar Movies" tab.
"""

import os
import chromadb

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "movies"
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
                f"'{CHROMA_DIR}' not found. Run 09_movies_vector_store.py first."
            )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def rank_similar_movies(story, top_k=10):
    """
    Returns a ranked list (closest first):
    [{"rank": 1, "title": "...", "year": "...", "genre": "...",
      "director": "...", "plot": "...", "distance": 0.42,
      "similarity": 0.70}, ...]

    `distance` is the raw vector distance from Chroma (lower = closer).
    `similarity` is a simple 0-1 conversion (1 / (1 + distance)) meant
    only for display -- it is not a calibrated probability.
    """
    model = get_embedding_model()
    collection = get_collection()

    query_embedding = model.encode([story]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    ranked = []
    for i, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        ranked.append(
            {
                "rank": i,
                "title": meta.get("title", "Unknown"),
                "year": meta.get("year", ""),
                "genre": meta.get("genre", ""),
                "director": meta.get("director", ""),
                "plot": doc,
                "distance": distance,
                "similarity": round(1 / (1 + distance), 3),
            }
        )

    return ranked


if __name__ == "__main__":
    story = input("Describe a movie plot: ")
    for movie in rank_similar_movies(story, top_k=10):
        print(f"{movie['rank']}. {movie['title']} ({movie['year']}) "
              f"- similarity: {movie['similarity']}")
