"""
streamlit_app.py
------------------
Streamlit UI for the RAG assistant.

Locally: reads OPENROUTER_API_KEY / OPENROUTER_MODEL from a .env file
(via 07_prompting.py).

On Streamlit Cloud: reads them from Streamlit secrets (Manage app ->
Secrets), and overrides the values coming from 07_prompting.py.
"""

import os
import importlib.util
import streamlit as st
import movie_similarity

# ---------------------------------------------------------------------
# Import 07_prompting.py despite its filename starting with a digit.
# We alias it as `rag`, matching the required project convention.
# ---------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "rag", os.path.join(_this_dir, "07_prompting.py")
)
rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rag)

# ---------------------------------------------------------------------
# Load secrets when deployed on Streamlit Cloud (overrides local .env)
# ---------------------------------------------------------------------
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="RAG Assistant", page_icon="🔎")
st.title("🔎 RAG Assistant")

tab_qa, tab_movies = st.tabs(["💬 Ask a question", "🎬 Find similar movies"])

# ---------------------------------------------------------------------
# Tab 1: generic Q&A over documents (steps 01-07)
# ---------------------------------------------------------------------
with tab_qa:
    st.caption(f"Model: {rag.OPENROUTER_MODEL}")

    if not rag.OPENROUTER_API_KEY:
        st.warning(
            "No OPENROUTER_API_KEY found. Add it to a local .env file, "
            "or to Streamlit secrets when deployed."
        )

    query = st.text_input("Ask a question about your documents:")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=4)

    if st.button("Ask") and query.strip():
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                answer, contexts = rag.ask_llm(query, top_k=top_k)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
            else:
                st.subheader("Answer")
                st.write(answer)

                if contexts:
                    st.subheader("Sources")
                    seen = set()
                    for c in contexts:
                        if c["source"] not in seen:
                            st.markdown(f"- `{c['source']}`")
                            seen.add(c["source"])

                    with st.expander("Show retrieved context chunks"):
                        for i, c in enumerate(contexts, start=1):
                            st.markdown(f"**Chunk {i} — source: `{c['source']}`**")
                            st.write(c["text"])
                            st.divider()

# ---------------------------------------------------------------------
# Tab 2: rank movies by plot similarity (steps 08-09, pure vector search,
# no LLM call needed)
# ---------------------------------------------------------------------
with tab_movies:
    st.caption("Type a story / plot idea and get the closest matching movies, ranked by similarity.")

    story = st.text_area(
        "Describe a movie plot or story:",
        placeholder="e.g. A detective investigates a murder in a Hollywood studio...",
        height=120,
    )
    movie_top_k = st.slider("Number of movies to show", min_value=3, max_value=25, value=10)

    if st.button("Find similar movies") and story.strip():
        with st.spinner("Ranking movies by similarity..."):
            try:
                ranked = movie_similarity.rank_similar_movies(story, top_k=movie_top_k)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
            else:
                if not ranked:
                    st.info("No movies found in the store.")
                else:
                    st.subheader("Ranked results")
                    st.dataframe(
                        [
                            {
                                "Rank": m["rank"],
                                "Title": m["title"],
                                "Year": m["year"],
                                "Genre": m["genre"],
                                "Director": m["director"],
                                "Similarity": m["similarity"],
                            }
                            for m in ranked
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                    with st.expander("Show plots"):
                        for m in ranked:
                            st.markdown(f"**{m['rank']}. {m['title']} ({m['year']})** — similarity {m['similarity']}")
                            st.write(m["plot"])
                            st.divider()
