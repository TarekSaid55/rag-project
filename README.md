# RAG Project

A simple Retrieval-Augmented Generation pipeline built with plain Python
files (no notebooks), plus a Streamlit chat UI.

Pipeline:

```
documents -> preprocessing -> chunking -> vector representation ->
vector store -> context retrieval -> prompting -> Streamlit UI
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your real OpenRouter key in it:

```bash
cp .env.example .env
```

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

`.env` is already in `.gitignore` — never commit it.

## 2. Add your documents

Put your `.txt` / `.pdf` files in `data/raw/` (a sample file is already
there — replace or remove it).

## 3. Build the knowledge base (run once, in order)

```bash
python 01_documents.py
python 02_preprocessing.py
python 03_chunking.py
python 04_vector_representation.py
python 05_create_chroma_store.py
```

Re-run these 5 steps any time you change the documents in `data/raw/`.

## 4. Run the app locally

```bash
streamlit run streamlit_app.py
```

## 5. Deploy on Streamlit Cloud

1. Push this project to a GitHub repository (make sure `.env` and
   `chroma_db/` are NOT included — check `.gitignore`).
2. **Important:** `chroma_db/` is git-ignored, but Streamlit Cloud needs
   it to answer questions. Either:
   - remove `chroma_db/` from `.gitignore` and commit it (simplest for a
     small demo project), or
   - run the 5 pipeline steps as a one-time setup step in your deployed
     app / a build script.
3. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing to `streamlit_app.py` in your repo.
4. In the app dashboard: **Manage app -> Secrets**, add:

```toml
OPENROUTER_API_KEY = "your_openrouter_key_here"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

5. Save — the app will restart and read the key from `st.secrets`.

## Final checklist

- [ ] All required Python files exist (01 through 07, streamlit_app.py)
- [ ] `requirements.txt` exists
- [ ] Real API key is NOT in the ZIP or in GitHub
- [ ] Streamlit secrets configured in valid TOML
- [ ] App runs successfully
- [ ] Answers use retrieved context
- [ ] Answers cite sources
