"""
08_movie_documents.py
------------------------
Prepares data for the "similar movies" feature.

Unlike the generic 01-03 pipeline (which splits long text into small
chunks), here each MOVIE is kept as ONE unit, because we want to rank
whole movies against each other -- not fragments of movies.

Reads data/movies.csv (columns: Release Year, Title, Origin/Ethnicity,
Director, Cast, Genre, Wiki Page, Plot) and saves a clean list to
data/movies.json:

{
    "title": "...",
    "year": 1939,
    "genre": "drama",
    "director": "...",
    "plot": "..."
}

Run:
    python 08_movie_documents.py
"""

import os
import json
import pandas as pd

INPUT_CSV = os.path.join("data", "movies.csv")
OUTPUT_FILE = os.path.join("data", "movies.json")


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"'{INPUT_CSV}' not found. Put your movies CSV there "
            "(must have Title and Plot columns at minimum)."
        )

    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["Title", "Plot"])

    movies = []
    for _, row in df.iterrows():
        movies.append(
            {
                "title": str(row.get("Title", "")).strip(),
                "year": row.get("Release Year", None),
                "genre": str(row.get("Genre", "")).strip(),
                "director": str(row.get("Director", "")).strip(),
                "plot": str(row.get("Plot", "")).strip(),
            }
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)

    print(f"Prepared {len(movies)} movies -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
