from __future__ import annotations

from io import BytesIO

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader


def extract_pdf_documents(file_bytes: bytes, file_name: str) -> tuple[list[Document], int]:
    reader = PdfReader(BytesIO(file_bytes))
    documents: list[Document] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={"source": file_name, "page": page_index},
            )
        )

    return documents, len(reader.pages)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = f"chunk-{idx}"

    return chunks
