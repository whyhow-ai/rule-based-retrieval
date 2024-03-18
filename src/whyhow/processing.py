"""Collection of utilities for extracting and processing text."""

import copy
import pathlib
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_and_split(
    path: str | pathlib.Path,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Parse a PDF and split it into chunks.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the document to process.

    chunk_size : int
        Size of the chunks.

    chunk_overlap : int
        Overlap between chunks.

    Returns
    -------
    list[Document]
        The chunks of the pdf.
    """
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # Assign the change number (within a page) to each chunk
    i_page = 0
    i_chunk = 0

    for chunk in chunks:
        if chunk.metadata["page"] != i_page:
            i_page = chunk.metadata["page"]
            i_chunk = 0

        chunk.metadata["chunk"] = i_chunk
        i_chunk += 1

    return chunks


def clean_chunks(
    chunks: list[Document],
) -> list[Document]:
    """Clean the chunks of a pdf.

    No modifications in-place.

    Parameters
    ----------
    chunks : list[Document]
        The chunks of the pdf.

    Returns
    -------
    list[Document]
        The cleaned chunks.
    """
    pattern = re.compile(r"(\r\n|\n|\r)")
    clean_chunks: list[Document] = []

    for chunk in chunks:
        text = re.sub(pattern, "", chunk.page_content)
        new_chunk = Document(
            page_content=text,
            metadata=copy.deepcopy(chunk.metadata),
        )

        clean_chunks.append(new_chunk)

    return clean_chunks
