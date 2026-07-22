# Engineering Journal

## 2026-07-22 — RAG chatbot deployment: Render OOM → dropped torch for Gemini embeddings

### Context
`app.py` (Session 3 RAG chatbot) was implemented on `feature/rag`: chunk `menu_kb.md`,
embed with `sentence-transformers`, index with FAISS, retrieve top-k, answer with
Gemini. Added a `Dockerfile`/`.dockerignore` to deploy on Render.

### Problem
After deploying to Render (free tier), the app returned `502 Bad Gateway`, then later
loaded the page shell but hung forever on `Running load_index()` with no error.
Render eventually sent an official notification: **the service exceeded its memory
limit and was auto-restarted.**

### Diagnosis
Free Render instances are capped at 512MB RAM. The build logs showed
`sentence-transformers` was pulling in the **full CUDA build of `torch`**
(`nvidia-cublas`, `nvidia-cudnn`, `cuda-toolkit`, `triton`, etc.) even though Render
has no GPU — this bloated both the image and, more importantly, resident memory when
`torch` probed for CUDA at import time. Loading the multilingual MiniLM model on top
of that regularly exceeded the memory limit.

### Things we tried, in order
1. **Pin `torch` to the CPU-only wheel** in the Dockerfile
   (`--index-url https://download.pytorch.org/whl/cpu`) before installing
   `sentence-transformers`, so pip wouldn't pull the CUDA build. Shrank image size but
   didn't fully solve the OOM — the embedding model itself was still heavy for 512MB.
2. **Considered Hugging Face Spaces** (the original intended deploy target per the
   `app.py` docstring). Turned out this account's Space creator only offers
   Gradio / Static / Docker, and Docker requires a paid plan — no native Streamlit
   SDK option was available.
3. **Ported the UI to Gradio** (`gr.ChatInterface`) to fit HF's free SDK path, keeping
   the Streamlit UI commented out in `app.py` for reversibility. Turned out Gradio on
   this HF account also hit a payment requirement, so we reverted: Streamlit restored
   as the active UI, Gradio version commented out instead.
4. **Considered Streamlit Community Cloud** as another free host — same underlying
   `torch`/CUDA bloat problem would apply there too (no custom Dockerfile control),
   with the equivalent fix being an `--extra-index-url` line at the top of
   `requirements.txt`.
5. **Considered AWS EC2 free tier** — ruled out: free instance sizes have a similar or
   worse memory ceiling, the free tier is time-limited (12 months) and needs a credit
   card, and it's more ops overhead than Render/HF/Streamlit Cloud.

### Root fix
Replaced local embeddings entirely: `load_index()` and `retrieve_top_k()` now call
**Gemini's `embed_content` API** (`gemini-embedding-001`) instead of running
`sentence-transformers` locally. `faiss` stays (it doesn't depend on `torch`) for the
actual vector search. This removes `torch` and `sentence-transformers` from the
dependency tree entirely, eliminating the memory problem at the source instead of
working around it. Reuses the same `GOOGLE_API_KEY` already used for answer
generation — no new secret needed.

The old `sentence-transformers`-based `load_index`/`retrieve_top_k` are kept
commented out at the bottom of `app.py` in case we ever want to go back to local
embeddings.

### Verification
Wrote a standalone script, installed a minimal venv (`google-genai`, `faiss-cpu`,
`numpy`, `python-dotenv`), and ran the pipeline end-to-end against the real API key:
- Embedded all 5 KB chunks → `(5, 3072)` vectors
- Retrieved sensible top-3 chunks for a Thai test query
- Generated a valid answer via `gemini-2.5-flash`

No `torch` involved anywhere in the run.

### Current state
- `app.py`: Streamlit UI + Gemini embeddings is the active path.
- Commented out (not deleted) for future reference: Gradio UI version, old
  `sentence-transformers` embedding version.
- `requirements.txt`: `sentence-transformers` and `gradio` commented out, `numpy`
  added (now imported directly), `streamlit`/`faiss-cpu` active.
- `Dockerfile`: back to a plain `pip install -r requirements.txt` — the CPU-only
  `torch` wheel workaround is no longer needed since `torch` isn't pulled in at all
  anymore.

### Open items
- Commit and push, then redeploy on Render to confirm the OOM is actually gone.
- `.env` has real secrets checked in locally (Gemini key, Telegram token, Sheets
  service account key) — gitignored, but rotate any of these if they were ever
  shared/pasted outside this machine.
- Still undecided: stick with Render, or move to Streamlit Community Cloud /
  Hugging Face Spaces long-term.
