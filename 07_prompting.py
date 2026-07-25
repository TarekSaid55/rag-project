"""
07_prompting.py
-----------------
Step 7 of the RAG pipeline.

This module:
1. Retrieves relevant context for a query (using 06_retrieve_context.py)
2. Builds a prompt that includes that context
3. Calls an LLM through OpenRouter
4. Returns an answer that cites its sources

IMPORTANT: Do not hardcode your real API key here. Set it via:
- a local .env file (for local development, never commit this file), or
- Streamlit secrets (for deployment) — see streamlit_app.py

This file is imported dynamically by streamlit_app.py as `rag`,
since its filename starts with a digit and can't be imported normally
with `import 07_prompting`.
"""

import os
import importlib.util
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------
# Config (read from environment / .env locally, or Streamlit secrets
# when deployed — see streamlit_app.py)
# ---------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------
# Import 06_retrieve_context.py despite its filename starting with a digit
# ---------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "retrieve_context_module", os.path.join(_this_dir, "06_retrieve_context.py")
)
_retrieve_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_retrieve_module)

retrieve_context = _retrieve_module.retrieve_context


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the answer is not contained in the context, "
    "say you don't know. Always cite the source file(s) you used, "
    "in the format [source: filename]."
)


def build_prompt(query, contexts):
    context_block = "\n\n".join(
        f"[source: {c['source']}]\n{c['text']}" for c in contexts
    )
    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context above, "
        "and cite the source file(s) you used."
    )
    return user_prompt


def call_openrouter(system_prompt, user_prompt):
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file "
            "locally, or to Streamlit secrets when deployed."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def ask_llm(query, top_k=4):
    """
    Full RAG call: retrieve -> build prompt -> call LLM.
    Returns: (answer_text, contexts_used)
    """
    contexts = retrieve_context(query, top_k=top_k)

    if not contexts:
        return (
            "I couldn't find any relevant context in the knowledge base "
            "to answer this question.",
            [],
        )

    user_prompt = build_prompt(query, contexts)
    answer = call_openrouter(SYSTEM_PROMPT, user_prompt)
    return answer, contexts


if __name__ == "__main__":
    q = input("Ask a question: ")
    answer, contexts = ask_llm(q)
    print("\nAnswer:\n", answer)
    print("\nSources used:")
    for c in contexts:
        print(" -", c["source"])
