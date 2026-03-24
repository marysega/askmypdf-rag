from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from src.askmypdf_rag.rag_pipeline import answer_question, build_vector_store


st.set_page_config(page_title="AskMyPDF RAG", page_icon="📄", layout="wide")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "document_stats" not in st.session_state:
    st.session_state.document_stats = None
if "source_documents" not in st.session_state:
    st.session_state.source_documents = []

logo_path = Path(__file__).parent / "assets" / "AskMyPdfLOGO.png"
avatar_path = Path(__file__).parent / "assets" / "maryse-avatar.svg"


def to_data_uri(path: Path) -> str:
    mime = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


logo_data_uri = to_data_uri(logo_path) if logo_path.exists() else ""
avatar_data_uri = to_data_uri(avatar_path) if avatar_path.exists() else ""

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

        :root {
            --page: #ffffff;
            --shell: #ffffff;
            --card: #ffffff;
            --ink: #243f67;
            --muted: #58708f;
            --line: #f1f4f8;
            --line-strong: #edf1f6;
            --red: #e76171;
            --red-dark: #d64c5f;
            --blue: #4d97ea;
            --blue-dark: #2f75cb;
            --shadow-main: 0 14px 30px rgba(95, 123, 164, 0.06);
            --shadow-soft: 0 8px 18px rgba(111, 137, 174, 0.06);
        }

        .stApp {
            background: #ffffff;
            color: var(--ink);
            font-family: "Nunito", sans-serif;
        }

        .block-container {
            max-width: 1440px;
            padding-top: 1.25rem;
            padding-bottom: 2.2rem;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        h1, h2, h3, h4 {
            font-family: "Nunito", sans-serif;
            color: var(--ink);
        }

        .shell {
            background: var(--shell);
            border: 1px solid var(--line);
            border-radius: 34px;
            box-shadow: var(--shadow-main);
            overflow: hidden;
        }

        .topbar {
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: center;
            padding: 0.8rem 1.15rem;
            border-bottom: 1px solid var(--line);
            background: #ffffff;
        }

        .topbar-center {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .topbar-right {
            display: flex;
            justify-content: flex-end;
        }

        .avatar {
            width: 3.15rem;
            height: 3.15rem;
            border-radius: 50%;
            border: 1px solid var(--line-strong);
            background: #ffffff;
            overflow: hidden;
            box-shadow: var(--shadow-soft);
        }

        .avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        .hero {
            padding: 0 1.15rem 0.12rem 1.15rem;
            background: #ffffff;
        }

        .logo-slot {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 118px;
            overflow: hidden;
            margin-bottom: 0;
        }

        .logo-slot img {
            width: 540px;
            max-width: 100%;
            display: block;
            transform: scale(1.38);
            transform-origin: center center;
        }

        .hero-copy {
            max-width: 1060px;
            margin: -0.1rem auto 0.3rem auto;
            text-align: center;
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.6;
        }

        .hero-copy strong {
            color: var(--ink);
            font-weight: 800;
        }

        .step-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1.2rem;
            margin-top: 0.2rem;
        }

        .step-card {
            position: relative;
            background: #ffffff;
            border: 1px solid var(--line-strong);
            border-radius: 26px;
            padding: 1.05rem 1rem 0.95rem 1rem;
            box-shadow: var(--shadow-soft);
        }

        .step-badge {
            position: absolute;
            top: -1rem;
            left: 1rem;
            width: 3.25rem;
            height: 3.25rem;
            border-radius: 50%;
            display: grid;
            place-items: center;
            color: #ffffff;
            font-size: 1.5rem;
            font-weight: 800;
            box-shadow: 0 8px 16px rgba(111, 137, 174, 0.14);
        }

        .badge-red {
            background: linear-gradient(180deg, #ef7b89, var(--red-dark));
        }

        .badge-blue {
            background: linear-gradient(180deg, #62aaf2, var(--blue-dark));
        }

        .step-card h3 {
            margin: 0.8rem 0 0.6rem 0;
            font-size: 1.02rem;
            line-height: 1.15;
            padding-bottom: 0.55rem;
            border-bottom: 1px solid var(--line);
        }

        .step-card p {
            margin: 0;
            font-size: 0.92rem;
            line-height: 1.48;
            color: var(--muted);
        }

        .content-grid {
            display: grid;
            grid-template-columns: 0.92fr 1.18fr;
            gap: 1.25rem;
            padding: 1rem 2rem 1.2rem 2rem;
            background: #ffffff;
            border-top: 1px solid #f7f9fc;
        }

        .panel {
            background: #ffffff;
            border: 1px solid var(--line-strong);
            border-radius: 30px;
            box-shadow: var(--shadow-soft);
            padding: 1.35rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid var(--line-strong);
            border-radius: 30px;
            box-shadow: var(--shadow-soft);
            padding: 1.2rem;
        }

        .upload-panel {
            min-height: 25rem;
        }

        .upload-icon {
            font-size: 3.2rem;
            color: var(--red-dark);
            line-height: 1;
            margin-bottom: 0.75rem;
        }

        .upload-text {
            color: var(--muted);
            font-size: 1rem;
            margin-bottom: 1.25rem;
        }

        .creator {
            margin-top: 1.8rem;
            padding-top: 0.9rem;
            border-top: 1px solid var(--line);
            color: #8a9ab1;
            font-size: 0.98rem;
            width: 100%;
            text-align: center;
        }

        .detail-list {
            display: flex;
            flex-direction: column;
            gap: 0.95rem;
        }

        .detail-item {
            display: grid;
            grid-template-columns: 3.3rem 1fr;
            gap: 0.9rem;
            align-items: start;
            padding-bottom: 0.95rem;
            border-bottom: 1px solid var(--line);
        }

        .detail-item:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }

        .detail-icon {
            width: 3rem;
            height: 3rem;
            border-radius: 18px;
            display: grid;
            place-items: center;
            color: #ffffff;
            font-size: 1.35rem;
            font-weight: 800;
            box-shadow: 0 10px 22px rgba(111, 137, 174, 0.16);
        }

        .detail-icon.red {
            background: linear-gradient(180deg, #ef7b89, var(--red-dark));
        }

        .detail-icon.blue {
            background: linear-gradient(180deg, #62aaf2, var(--blue-dark));
        }

        .detail-item h4 {
            margin: 0;
            font-size: 1.08rem;
        }

        .detail-item p {
            margin: 0.22rem 0 0 0;
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.58;
        }

        .qa-panel {
            margin: 0 2rem 1.55rem 2rem;
            background: #ffffff;
            border: 1px solid var(--line-strong);
            border-radius: 30px;
            box-shadow: var(--shadow-soft);
            padding: 1.35rem;
        }

        .stats-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-bottom: 1rem;
        }

        .stat-box {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.95rem 1rem;
        }

        .stat-box strong {
            display: block;
            color: #8a9ab1;
            font-size: 0.78rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.28rem;
        }

        .stat-box span {
            color: var(--ink);
            font-size: 1.15rem;
            font-weight: 800;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .chip-label {
            color: #8a9ab1;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }

        .chip {
            background: #ffffff;
            border: 1px solid var(--line-strong);
            border-radius: 14px;
            padding: 0.72rem 1rem;
            color: var(--muted);
            font-size: 0.92rem;
            font-weight: 700;
        }

        .source-block {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.95rem;
            margin-bottom: 0.75rem;
        }

        .footer-note {
            text-align: center;
            color: #8a9ab1;
            font-size: 0.94rem;
            padding: 0 0 1rem 0;
        }

        div[data-testid="stFileUploader"] section,
        div[data-testid="stTextInput"] input {
            background: #ffffff;
            border-radius: 18px;
            border: 1px solid var(--line-strong);
            min-height: 3.35rem;
        }

        section[data-testid="stFileUploaderDropzone"] {
            min-height: 15rem;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid var(--line-strong) !important;
            border-radius: 24px !important;
            background: #ffffff !important;
        }

        section[data-testid="stFileUploaderDropzone"] div {
            color: var(--muted);
        }

        div[data-testid="stButton"] > button {
            border: none;
            border-radius: 999px;
            background: linear-gradient(180deg, #ee6d79, #da5868);
            color: #ffffff;
            font-weight: 800;
            font-size: 1rem;
            min-height: 3.2rem;
            box-shadow: 0 12px 24px rgba(218, 88, 104, 0.22);
        }

        div[data-testid="stButton"] > button:hover {
            background: linear-gradient(180deg, #e45f6d, #cf4e5f);
        }

        .send-button div[data-testid="stButton"] > button {
            border-radius: 18px;
            background: linear-gradient(180deg, #3f92f2, #256dc8);
            box-shadow: 0 12px 24px rgba(60, 109, 194, 0.20);
        }

        @media (max-width: 1180px) {
            .step-grid {
                grid-template-columns: 1fr 1fr;
            }

            .content-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 780px) {
            .step-grid,
            .stats-row {
                grid-template-columns: 1fr;
            }

            .hero,
            .content-grid,
            .qa-panel {
                padding-left: 1rem;
                padding-right: 1rem;
                margin-left: 0;
                margin-right: 0;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="shell">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="topbar">
        <div class="topbar-center"></div>
        <div class="topbar-right">
            <div class="avatar"><img src="{avatar_data_uri}" alt="Maryse avatar" /></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero">', unsafe_allow_html=True)
if logo_data_uri:
    st.markdown(
        f'<div class="logo-slot"><img src="{logo_data_uri}" alt="AskMyPDF logo" /></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown("## AskMyPDF RAG")

st.markdown(
    """
    <p class="hero-copy">
        Upload a PDF, split it into chunks, generate embeddings,
        <strong>search relevant passages</strong>, and answer questions with
        <strong>grounded context</strong> using a clean
        <strong>retrieval-augmented generation</strong> workflow.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="step-grid">
        <div class="step-card">
            <div class="step-badge badge-red">1</div>
            <h3>Document<br/>Ingestion</h3>
            <p>Upload the PDF and extract text page by page before processing.</p>
        </div>
        <div class="step-card">
            <div class="step-badge badge-blue">2</div>
            <h3>Chunking +<br/>Embeddings</h3>
            <p>Split content into meaningful chunks and convert them into vectors.</p>
        </div>
        <div class="step-card">
            <div class="step-badge badge-blue">3</div>
            <h3>FAISS<br/>Retrieval</h3>
            <p>Search the vector store to recover the most relevant passages.</p>
        </div>
        <div class="step-card">
            <div class="step-badge badge-red">4</div>
            <h3>Grounded<br/>Answer</h3>
            <p>Inject retrieved context into the prompt and display the answer with sources.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

left_col, right_col = st.columns([0.92, 1.18], gap="large")

with left_col:
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], label_visibility="collapsed")
        if st.button("Upload PDF", type="primary", use_container_width=True):
            if uploaded_file is None:
                st.warning("Upload a PDF first.")
            else:
                try:
                    with st.spinner("Building embeddings and FAISS index..."):
                        vector_store, stats = build_vector_store(uploaded_file)
                        st.session_state.vector_store = vector_store
                        st.session_state.document_stats = stats
                        st.session_state.source_documents = []
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        st.markdown('<div class="creator">Created by Maryse G.A.</div>', unsafe_allow_html=True)

with right_col:
    with st.container(border=True):
        st.markdown(
            """
            <div class="detail-list">
                <div class="detail-item">
                    <div class="detail-icon red">&#128196;</div>
                    <div>
                        <h4>Document Ingestion</h4>
                        <p>Upload the PDF and extract text page by page before processing.</p>
                    </div>
                </div>
                <div class="detail-item">
                    <div class="detail-icon blue">&#8801;</div>
                    <div>
                        <h4>Chunking + Embeddings</h4>
                        <p>Split content into meaningful chunks and convert them into vectors.</p>
                    </div>
                </div>
                <div class="detail-item">
                    <div class="detail-icon blue">&#8981;</div>
                    <div>
                        <h4>FAISS Retrieval</h4>
                        <p>Search the vector store to recover the most relevant passages.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.container(border=True):
    if st.session_state.document_stats:
        stats = st.session_state.document_stats
        st.markdown(
            f"""
            <div class="stats-row">
                <div class="stat-box"><strong>Pages</strong><span>{stats["pages"]}</span></div>
                <div class="stat-box"><strong>Chunks</strong><span>{stats["chunks"]}</span></div>
                <div class="stat-box"><strong>Characters</strong><span>{stats["characters"]}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="chip-label">Example prompts:</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chip-row">
            <div class="chip">Summarize the document</div>
            <div class="chip">What are the key points?</div>
            <div class="chip">Find the publication date</div>
            <div class="chip">Compare important sections</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ask_col, send_col = st.columns([4.8, 1], gap="small")
    with ask_col:
        question = st.text_input(
            "Ask a question about the PDF",
            placeholder="Ask your question here...",
            label_visibility="collapsed",
        )
    with send_col:
        st.markdown('<div class="send-button">', unsafe_allow_html=True)
        ask_clicked = st.button("Send", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if ask_clicked:
        if st.session_state.vector_store is None:
            st.warning("Upload and process a PDF before asking a question.")
        elif not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Retrieving relevant chunks and generating the answer..."):
                    result = answer_question(
                        question=question,
                        vector_store=st.session_state.vector_store,
                        top_k=4,
                    )
                    st.session_state.source_documents = result["sources"]

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Sources")
                for idx, source in enumerate(result["sources"], start=1):
                    st.markdown(
                        f"""
                        <div class="source-block">
                            <strong>Source {idx}</strong><br/>
                            Page: {source['page']}<br/>
                            Chunk: {source['chunk_id']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.code(source["content"], language="text")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
st.markdown('<p class="footer-note">AskMyPDF RAG | Created by Maryse G.A</p>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
