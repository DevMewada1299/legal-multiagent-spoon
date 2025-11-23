## Goal
Run the Streamlit app locally, identify runtime errors, and implement fixes so upload → analyze → results works reliably.

## What I Found
- Entrypoint: `streamlit_app.py` (`streamlit run streamlit_app.py`).
- Pipeline modules: ingestion/classification/extraction/retrieval/summarization under `app/`.
- Summarization requires `GEMINI_API_KEY`; otherwise raises `ValueError` at `app/summarization/gemini_summarizer.py:16`.
- Retrieval initializes `SentenceTransformer` at `app/retrieval/faiss_retriever.py:10`; this commonly fails if `torch` is missing.
- Dependencies listed in `requirements.txt` do not include `torch`.

## Execution Plan
1. Environment Setup
- Create a fresh virtual environment.
- Install deps: `pip install -r requirements.txt`.
- Ensure `.env` has `GEMINI_API_KEY` set; if not, expect summarization error.

2. Run and Reproduce
- Start the app: `streamlit run streamlit_app.py`.
- Upload a sample PDF and click Analyze.
- Capture and categorize errors (missing keys, missing packages, model download failures, FAISS issues).

3. Fixes
- Handle missing `GEMINI_API_KEY` gracefully: catch at `streamlit_app.py:146-148` and show a clear UI message, continue to display classification/retrieval results.
- Add `torch` to dependencies if `SentenceTransformer` fails at `app/retrieval/faiss_retriever.py:10`; alternatively, lazy-load or guard with an informative error.
- Add a startup check for critical deps (PyMuPDF, FAISS, SentenceTransformer) with user-friendly guidance.

4. Validation
- Re-run the app, verify:
  - PDF upload works and clauses render.
  - Retrieval tab populates.
  - Summary tab either shows output (with `GEMINI_API_KEY`) or a graceful notice.

## Commands I’ll Run
- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- `streamlit run streamlit_app.py`

## Potential Code Changes
- Update `requirements.txt` to include `torch` if needed.
- Improve error handling and user messaging around GEMINI setup at `streamlit_app.py:146-148` and `app/summarization/gemini_summarizer.py:14-17`.
- Optional: add a preflight check component for environment readiness.

Confirm and I’ll execute, diagnose, and implement the fixes.