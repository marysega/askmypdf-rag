from __future__ import annotations

import streamlit as st

from src.askmypdf_rag.rag_pipeline import build_vector_store, answer_question


st.set_page_config(page_title="Maryse App | AskMyPDF RAG", page_icon="📄", layout="wide")

st.title("Welcome to Maryse App")
st.subheader("AskMyPDF RAG")
st.caption(
    "Question answering over PDF documents using ingestion, chunking, embeddings, "
    "FAISS vector search, retrieval, and LLM-based answer generation."
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "document_stats" not in st.session_state:
    st.session_state.document_stats = None
if "source_documents" not in st.session_state:
    st.session_state.source_documents = []

with st.sidebar:
    st.header("Pipeline")
    st.markdown(
        """
        - PDF ingestion
        - Text extraction
        - Chunking
        - Embeddings
        - FAISS vector store
        - Retrieval
        - Prompt with context
        - Answer generation
        """
    )
    top_k = st.slider("Top-k retrieved chunks", min_value=2, max_value=8, value=4)
    st.divider()
    st.caption("Created by Maryse G.A")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    if st.button("Process PDF", type="primary"):
        try:
            with st.spinner("Building embeddings and FAISS index..."):
                vector_store, stats = build_vector_store(uploaded_file)
                st.session_state.vector_store = vector_store
                st.session_state.document_stats = stats
                st.session_state.source_documents = []
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    if st.session_state.document_stats:
        stats = st.session_state.document_stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Pages", stats["pages"])
        col2.metric("Chunks", stats["chunks"])
        col3.metric("Characters", stats["characters"])

if st.session_state.vector_store is not None:
    question = st.text_input(
        "Ask a question about the PDF",
        placeholder="What are the main findings of this document?",
    )

    if question:
        try:
            with st.spinner("Retrieving relevant chunks and generating the answer..."):
                result = answer_question(
                    question=question,
                    vector_store=st.session_state.vector_store,
                    top_k=top_k,
                )
                st.session_state.source_documents = result["sources"]

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Retrieved Sources")
            for idx, source in enumerate(result["sources"], start=1):
                st.markdown(
                    f"**Source {idx}**  \n"
                    f"Page: {source['page']}  \n"
                    f"Chunk: `{source['chunk_id']}`"
                )
                st.code(source["content"], language="text")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
else:
    st.info("Upload a PDF and process it to initialize the RAG pipeline.")

st.divider()
st.caption("Created by Maryse G.A")
