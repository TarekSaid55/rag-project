"""
09_movies_vector_store.py
----------------------------
Embeds each movie's plot (from data/movies.json, step 8) and stores
it in its own Chroma collection ("movies"), separate from the generic
document collection used by the Q&A pipeline (01-05).

Run:
    python 09_movies_vector_store.py
"""

import os
import json
import chromadb

INPUT_FILE = os.path.join("data", "movies.json")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "movies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"'{INPUT_FILE}' not found. Run 08_movie_documents.py first."
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        movies = json.load(f)

    if not movies:
        print("No movies found. Nothing to embed.")
        return

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Embed title + plot together so the title's wording also contributes
    texts = [f"{m['title']}. {m['plot']}" for m in movies]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    ids = [str(i) for i in range(len(movies))]
    documents = [m["plot"] for m in movies]
    metadatas = [
        {
            "title": m["title"],
            "year": "" if m["year"] is None else str(m["year"]),
            "genre": m["genre"],
            "director": m["director"],
        }
        for m in movies
    ]

    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"Stored {len(ids)} movies in Chroma collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
