from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import get_openai_settings
from .pdf_ingestion import chunk_documents, extract_pdf_documents


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful assistant for question answering over a PDF document.
Use only the retrieved context below to answer the user question.
If the answer is not in the context, say that the document does not provide enough information.

Question:
{question}

Context:
{context}

Answer:
"""
)


def build_vector_store(uploaded_file) -> tuple[FAISS, dict[str, int]]:
    settings = get_openai_settings()
    file_bytes = uploaded_file.getvalue()
    documents, total_pages = extract_pdf_documents(file_bytes, uploaded_file.name)
    if not documents:
        raise ValueError("No extractable text was found in this PDF.")

    chunks = chunk_documents(documents)
    embeddings = OpenAIEmbeddings(
        model=settings["embedding_model"],
        api_key=settings["api_key"],
        base_url=settings["base_url"],
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    stats = {
        "pages": total_pages,
        "chunks": len(chunks),
        "characters": sum(len(doc.page_content) for doc in documents),
    }
    return vector_store, stats


def answer_question(question: str, vector_store: FAISS, top_k: int = 4) -> dict[str, object]:
    settings = get_openai_settings()
    retrieved_docs = vector_store.similarity_search(question, k=top_k)
    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 'N/A')} | {doc.metadata.get('chunk_id', 'N/A')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    llm = ChatOpenAI(
        model=settings["chat_model"],
        api_key=settings["api_key"],
        base_url=settings["base_url"],
        temperature=0,
    )
    chain = ANSWER_PROMPT | llm
    response = chain.invoke({"question": question, "context": context})

    sources = [
        {
            "page": doc.metadata.get("page", "N/A"),
            "chunk_id": doc.metadata.get("chunk_id", "N/A"),
            "content": doc.page_content,
        }
        for doc in retrieved_docs
    ]

    return {"answer": response.content, "sources": sources}
