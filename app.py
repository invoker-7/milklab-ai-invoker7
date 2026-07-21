"""MilkLab RAG Chatbot (S3).

Run locally: streamlit run app.py
Deploy: push to GitHub then Actions deploys to HuggingFace Space
"""

import os
import re

import faiss
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()

KB_PATH = "menu_kb.md"
EMBED_MODEL = "all-MiniLM-L6-v2"

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


@st.cache_resource
def load_index():
    """โหลด menu_kb.md, split เป็น chunk, encode ด้วย sentence-transformers, สร้าง faiss index.
    Cache เพราะโหลด model ครั้งแรกใช้เวลา 30 วินาที

    Returns: (model, index, chunks_list)
    """
    chunks = _load_chunks()
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return model, index, chunks


def retrieve_top_k(query: str, model, index, chunks: list[str], k: int = 3) -> list[str]:
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, indices = index.search(query_embedding.astype("float32"), k)
    return [chunks[i] for i in indices[0] if i != -1]


def generate_answer(query: str, context_chunks: list[str]) -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env")
    client = genai.Client(api_key=key)
    prompt = ANSWER_PROMPT.format(context="\n\n".join(context_chunks), query=query)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text or ""


def main():
    st.set_page_config(page_title="MilkLab° RAG", page_icon="🥛")
    st.title("MilkLab° RAG Chatbot")
    st.caption("ถามอะไรเกี่ยวกับ MilkLab ได้ ตอบจาก menu_kb.md")

    try:
        model, index, chunks = load_index()
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
                context = retrieve_top_k(prompt, model, index, chunks)
                answer = generate_answer(prompt, context)
            st.write(answer)
            with st.expander("Source chunks"):
                for i, c in enumerate(context, 1):
                    st.markdown(f"**[{i}]** {c}")
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
