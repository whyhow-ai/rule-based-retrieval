"""Collection of tests for the processing module."""

import pathlib

import pytest
from fpdf import FPDF
from langchain_core.documents import Document
from llmsherpa.parsing.pdf_utils import extract_metadata

from whyhow_rbr.processing import clean_chunks, parse_and_split


@pytest.fixture
def dummy_pdf(tmp_path) -> pathlib.Path:
    """Create a dummy PDF file."""
    output_path = tmp_path / "dummy.pdf"

    pdf = FPDF()
    pdf.set_font("Arial", size=12)

    pdf.add_page()
    pdf.cell(200, 10, txt="Welcome to the dummy PDF", ln=True, align="C")

    pdf.add_page()
    pdf.cell(200, 10, txt="This is the second page", ln=True, align="C")

    pdf.output(str(output_path))

    return output_path


def test_parse_and_split(dummy_pdf):
    """Test the dummy PDF."""
    result = parse_and_split(dummy_pdf, chunk_size=512, extract_extra_metadata=True)

    assert len(result) == 2
    assert result[0].page_content == "Welcome to the dummy PDF"
    assert result[0].metadata["page"] == 0
    assert result[0].metadata["chunk"] == 0
    assert result[0].metadata.get("author") is None
    assert result[0].metadata.get("subject") is None
    assert result[0].metadata.get("keywords") is None
    assert result[0].metadata.get("creation_date") is None
    assert result[0].metadata.get("modification_date") is None
    assert result[0].metadata.get("pages") is None
    assert result[0].metadata.get("file_size") is None
    assert result[0].metadata.get("pdf_version") is None

    assert result[1].page_content == "This is the second page"
    assert result[1].metadata["page"] == 1
    assert result[0].metadata["chunk"] == 0
    assert result[1].metadata.get("author") is None
    assert result[1].metadata.get("subject") is None
    assert result[1].metadata.get("keywords") is None
    assert result[1].metadata.get("creation_date") is None
    assert result[1].metadata.get("modification_date") is None
    assert result[1].metadata.get("pages") is None
    assert result[1].metadata.get("file_size") is None
    assert result[1].metadata.get("pdf_version") is None


def test_parse_and_split_small_chunks(dummy_pdf):
    """Test the dummy PDF with small chunks."""
    result = parse_and_split(dummy_pdf, chunk_size=7, chunk_overlap=0, extract_extra_metadata=True)

    assert len(result) == 8
    assert result[0].page_content == "Welcome"
    assert result[0].metadata["page"] == 0
    assert result[0].metadata["chunk"] == 0
    assert result[0].metadata.get("author") is None
    assert result[0].metadata.get("subject") is None
    assert result[0].metadata.get("keywords") is None
    assert result[0].metadata.get("creation_date") is None
    assert result[0].metadata.get("modification_date") is None
    assert result[0].metadata.get("pages") is None
    assert result[0].metadata.get("file_size") is None
    assert result[0].metadata.get("pdf_version") is None

    assert result[1].page_content == "to the"
    assert result[1].metadata["page"] == 0
    assert result[1].metadata["chunk"] == 1
    assert result[1].metadata.get("author") is None
    assert result[1].metadata.get("subject") is None
    assert result[1].metadata.get("keywords") is None
    assert result[1].metadata.get("creation_date") is None
    assert result[1].metadata.get("modification_date") is None
    assert result[1].metadata.get("pages") is None
    assert result[1].metadata.get("file_size") is None
    assert result[1].metadata.get("pdf_version") is None

    assert result[2].page_content == "dummy"
    assert result[2].metadata["page"] == 0
    assert result[2].metadata["chunk"] == 2
    assert result[2].metadata.get("author") is None
    assert result[2].metadata.get("subject") is None
    assert result[2].metadata.get("keywords") is None
    assert result[2].metadata.get("creation_date") is None
    assert result[2].metadata.get("modification_date") is None
    assert result[2].metadata.get("pages") is None
    assert result[2].metadata.get("file_size") is None
    assert result[2].metadata.get("pdf_version") is None

    assert result[3].page_content == "PDF"
    assert result[3].metadata["page"] == 0
    assert result[3].metadata["chunk"] == 3
    assert result[3].metadata.get("author") is None
    assert result[3].metadata.get("subject") is None
    assert result[3].metadata.get("keywords") is None
    assert result[3].metadata.get("creation_date") is None
    assert result[3].metadata.get("modification_date") is None
    assert result[3].metadata.get("pages") is None
    assert result[3].metadata.get("file_size") is None
    assert result[3].metadata.get("pdf_version") is None

    assert result[4].page_content == "This is"
    assert result[4].metadata["page"] == 1
    assert result[4].metadata["chunk"] == 0
    assert result[4].metadata.get("author") is None
    assert result[4].metadata.get("subject") is None
    assert result[4].metadata.get("keywords") is None
    assert result[4].metadata.get("creation_date") is None
    assert result[4].metadata.get("modification_date") is None
    assert result[4].metadata.get("pages") is None
    assert result[4].metadata.get("file_size") is None
    assert result[4].metadata.get("pdf_version") is None

    assert result[5].page_content == "the"
    assert result[5].metadata["page"] == 1
    assert result[5].metadata["chunk"] == 1
    assert result[5].metadata.get("author") is None
    assert result[5].metadata.get("subject") is None
    assert result[5].metadata.get("keywords") is None
    assert result[5].metadata.get("creation_date") is None
    assert result[5].metadata.get("modification_date") is None
    assert result[5].metadata.get("pages") is None
    assert result[5].metadata.get("file_size") is None
    assert result[5].metadata.get("pdf_version") is None

    assert result[6].page_content == "second"
    assert result[6].metadata["page"] == 1
    assert result[6].metadata["chunk"] == 2
    assert result[6].metadata.get("author") is None
    assert result[6].metadata.get("subject") is None
    assert result[6].metadata.get("keywords") is None
    assert result[6].metadata.get("creation_date") is None
    assert result[6].metadata.get("modification_date") is None
    assert result[6].metadata.get("pages") is None
    assert result[6].metadata.get("file_size") is None
    assert result[6].metadata.get("pdf_version") is None

    assert result[7].page_content == "page"
    assert result[7].metadata["page"] == 1
    assert result[7].metadata["chunk"] == 3
    assert result[7].metadata.get("author") is None
    assert result[7].metadata.get("subject") is None
    assert result[7].metadata.get("keywords") is None
    assert result[7].metadata.get("creation_date") is None
    assert result[7].metadata.get("modification_date") is None
    assert result[7].metadata.get("pages") is None
    assert result[7].metadata.get("file_size") is None
    assert result[7].metadata.get("pdf_version") is None


def test_clean_chunks():
    """Test the clean_chunks function."""
    chunks = [
        Document("This is a \n\ntest", metadata={"page": 0}),
        Document("Nothing changes here", metadata={"page": 1}),
        Document("Hor\nrible", metadata={"page": 1}),
    ]
    result = clean_chunks(chunks)

    assert len(result) == 3
    assert result[0].page_content == "This is a test"
    assert result[1].page_content == "Nothing changes here"
    assert result[2].page_content == "Horrible"

    assert result[0].metadata["page"] == 0
    assert result[1].metadata["page"] == 1
    assert result[2].metadata["page"] == 1

    # make sure the original chunks are not modified
    assert chunks[0].page_content == "This is a \n\ntest"