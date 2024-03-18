"""Tests for the embedding module."""

from unittest.mock import Mock

import pytest
from langchain_openai import OpenAIEmbeddings

from whyhow.embedding import generate_embeddings


@pytest.mark.parametrize("model", ["whatever", "else"])
def test_generate_embeddings(monkeypatch, model):
    chunks = ["hello there", "today is a great day"]

    fake_inst = Mock(spec=OpenAIEmbeddings)
    fake_inst.embed_documents.side_effect = lambda x: [[2.2, 5.5] for _ in x]
    fake_class = Mock(return_value=fake_inst)

    monkeypatch.setattr("whyhow.embedding.OpenAIEmbeddings", fake_class)
    embeddings = generate_embeddings(
        chunks=chunks, openai_api_key="test", model=model
    )

    assert fake_class.call_count == 1
    assert fake_class.call_args.kwargs["openai_api_key"] == "test"
    assert fake_class.call_args.kwargs["model"] == model

    assert fake_inst.embed_documents.call_count == 1

    assert len(embeddings) == 2
    assert embeddings[0] == [2.2, 5.5]
    assert embeddings[1] == [2.2, 5.5]
