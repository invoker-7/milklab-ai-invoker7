"""MilkLab RAG Chatbot (S3).

Run locally: streamlit run app.py
Deploy: push to GitHub then deploy to Render (see Dockerfile)

Gradio version is kept commented out below (main_gradio/chat_fn) in case
we switch to a HuggingFace Space later.
"""

import os
import re

import faiss
import numpy as np

# import gradio as gr
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig

# from sentence_transformers import SentenceTransformer

load_dotenv()

KB_PATH = "menu_kb.md"
EMBED_MODEL = "gemini-embedding-001"

ANSWER_PROMPT = """\
ตอบจากข้อมูลต่อไปนี้เท่านั้น ถ้าไม่มีใน context ให้บอกว่าไม่รู้

Context:
{context}

คำถาม: {query}
"""


def _load_chunks(path: str = KB_PATH) -> list[str]:
    text = open(path, encoding="utf-8").read()
    sections = re.split(r"\n(?=## )", text)
    return [s.strip() for s in sections if s.strip()]


def _get_client() -> genai.Client:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env")
    return genai.Client(api_key=key)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / norms).astype("float32")


@st.cache_resource
def load_index():
    """โหลด menu_kb.md, split เป็น chunk, encode ด้วย Gemini embedding API, สร้าง faiss index.
    Cache เพราะเรียก embedding API ครั้งแรกใช้เวลาสักพัก

    Returns: (client, index, chunks_list)
    """
    chunks = _load_chunks()
    client = _get_client()
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=chunks,
        config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    embeddings = _normalize(np.array([e.values for e in response.embeddings]))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return client, index, chunks


def retrieve_top_k(query: str, client: genai.Client, index, chunks: list[str], k: int = 3) -> list[str]:
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query],
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_embedding = _normalize(np.array([response.embeddings[0].values]))
    _, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0] if i != -1]


def generate_answer(query: str, context_chunks: list[str]) -> str:
    client = _get_client()
    prompt = ANSWER_PROMPT.format(context="\n\n".join(context_chunks), query=query)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text or ""


# --- sentence-transformers version (disabled) ---
# Kept in case we go back to local embeddings (torch was too heavy for Render's free 512MB tier).
# To re-enable: uncomment `from sentence_transformers import SentenceTransformer` near the top,
# uncomment load_index_local/retrieve_top_k_local below, and use those instead.
#
# @st.cache_resource
# def load_index_local():
#     chunks = _load_chunks()
#     model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#     embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
#     index = faiss.IndexFlatIP(embeddings.shape[1])
#     index.add(embeddings.astype("float32"))
#     return model, index, chunks
#
#
# def retrieve_top_k_local(query: str, model, index, chunks: list[str], k: int = 3) -> list[str]:
#     query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
#     _, indices = index.search(query_embedding.astype("float32"), k)
#     return [chunks[i] for i in indices[0] if i != -1]


def main():
    st.set_page_config(page_title="MilkLab° RAG", page_icon="🥛")
    st.title("MilkLab° RAG Chatbot")
    st.caption("ถามอะไรเกี่ยวกับ MilkLab ได้ ตอบจาก menu_kb.md")

    try:
        client, index, chunks = load_index()
    except NotImplementedError as exc:
        st.error(f"TODO not implemented: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("ถามอะไรเกี่ยวกับ MilkLab"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นข้อมูล..."):
                context = retrieve_top_k(prompt, client, index, chunks)
                answer = generate_answer(prompt, context)
            st.write(answer)
            with st.expander("Source chunks"):
                for i, c in enumerate(context, 1):
                    st.markdown(f"**[{i}]** {c}")
        st.session_state.messages.append({"role": "assistant", "content": answer})


# --- Gradio version (disabled) ---
# Kept in case we switch to a HuggingFace Space later (Docker/Gradio SDK required payment on this account).
# To re-enable: uncomment `import gradio as gr` near the top, uncomment
# chat_fn/main_gradio below, and call main_gradio() instead of main() at the bottom.
#
# def chat_fn(message: str, history) -> str:
#     model, index, chunks = load_index()
#     context = retrieve_top_k(message, model, index, chunks)
#     return generate_answer(message, context)
#
#
# def main_gradio():
#     demo = gr.ChatInterface(
#         fn=chat_fn,
#         title="MilkLab° RAG Chatbot",
#         description="ถามอะไรเกี่ยวกับ MilkLab ได้ ตอบจาก menu_kb.md",
#     )
#     demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
