import logging
from unittest.mock import Mock
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from qdrant_client import QdrantClient
from qdrant_client.http import models

from whyhow_rbr.exceptions import (
    CollectionAlreadyExistsException,
    CollectionNotFoundException,
)
from whyhow_rbr.rag_qdrant import (
    Client,
    Metadata,
    Output,
    QdrantDocument,
    Rule,
)


class TestRule:
    def test_default(self):
        rule = Rule()

        assert rule.filename is None
        assert rule.page_numbers is None
        assert rule.uuid is None

    def test_empty_page_numbers(self):
        rule = Rule(page_numbers=[])

        assert rule.filename is None
        assert rule.page_numbers is None
        assert rule.uuid is None

    def test_to_filter(self):
        # no conditions
        rule = Rule()
        assert rule.to_filter() is None

        # only filename
        rule = Rule(filename="hello.pdf")
        assert rule.to_filter() == models.Filter(
            must=[
                models.FieldCondition(
                    key="filename", match=models.MatchValue(value="hello.pdf")
                )
            ]
        )

        # only page_numbers
        rule = Rule(page_numbers=[1, 2])
        assert rule.to_filter() == models.Filter(
            must=[
                models.FieldCondition(
                    key="page_number", match=models.MatchAny(any=[1, 2])
                )
            ]
        )


class TestQdrantDocument:
    def test_auto_id(self):
        metadata = Metadata(
            text="hello world",
            page_number=1,
            chunk_number=0,
            filename="hello.pdf",
        )
        doc = QdrantDocument(
            vector=[0.2, 0.3],
            metadata=metadata,
        )

        assert doc.id is not None

    def test_provide_id(self):
        _id = str(uuid4())
        metadata = Metadata(
            text="hello world",
            page_number=1,
            chunk_number=0,
            filename="hello.pdf",
        )
        doc = QdrantDocument(
            vector=[0.2, 0.3],
            metadata=metadata,
            id=_id,
        )

        assert doc.id == _id


@pytest.fixture(name="client")
def patched_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret_openai")

    fake_qdrant_instance = Mock(spec=QdrantClient)
    fake_qdrant_class = Mock(return_value=fake_qdrant_instance)

    fake_openai_instance = Mock(spec=OpenAI)
    fake_openai_class = Mock(return_value=fake_openai_instance)

    monkeypatch.setattr(
        "whyhow_rbr.rag_qdrant.QdrantClient", fake_qdrant_class
    )
    monkeypatch.setattr("whyhow_rbr.rag.OpenAI", fake_openai_class)

    client = Client(fake_openai_instance, fake_qdrant_instance)

    assert isinstance(client.openai_client, Mock)
    assert isinstance(client.qdrant_client, Mock)

    return client


class TestClient:
    def test_collection(self, client):
        def side_effect(*args, **kwargs):
            raise CollectionNotFoundException()

        client.qdrant_client.collection_exists.side_effect = side_effect

        with pytest.raises(CollectionNotFoundException):
            client.query("some question", "some collection")

    def test_create_index(self, client, monkeypatch):
        client.qdrant_client.collection_exists.return_value = False
        client.create_collection("some name")
        assert client.qdrant_client.collection_exists.call_count == 1
        assert client.qdrant_client.create_collection.call_count == 1

        client.qdrant_client.collection_exists.return_value = True
        with pytest.raises(CollectionAlreadyExistsException):
            client.create_collection("some name")

    def test_upload_documents_nothing(self, client, caplog):
        caplog.set_level(logging.INFO)
        client.upload_documents(
            "some collection",
            documents=[],
        )

        captured = caplog.records[0]

        assert captured.levelname == "INFO"
        assert "No documents to upload" in captured.message

    def test_upload_document(self, client, caplog, monkeypatch):
        caplog.set_level(logging.INFO)
        documents = ["doc1.pdf", "doc2.pdf"]

        client.openai_client.api_key = "fake"
        client.qdrant_client.upload_points = Mock(return_value=None)

        parsed_docs = [
            Document(
                page_content="hello there",
                metadata={
                    "page": 0,
                    "chunk": 0,
                    "source": "something",
                },
            ),
            Document(
                page_content="again",
                metadata={
                    "page": 0,
                    "chunk": 1,
                    "source": "something",
                },
            ),
            Document(
                page_content="it is cold",
                metadata={
                    "page": 1,
                    "chunk": 0,
                    "source": "something",
                },
            ),
        ]
        fake_parse_and_split = Mock(return_value=parsed_docs)
        fake_clean_chunks = Mock(return_value=parsed_docs)
        fake_generate_embeddings = Mock(return_value=6 * [[2.2, 0.6]])

        monkeypatch.setattr(
            "whyhow_rbr.rag_qdrant.parse_and_split", fake_parse_and_split
        )
        monkeypatch.setattr(
            "whyhow_rbr.rag_qdrant.clean_chunks", fake_clean_chunks
        )
        monkeypatch.setattr(
            "whyhow_rbr.rag_qdrant.generate_embeddings",
            fake_generate_embeddings,
        )

        client.upload_documents(
            "some collection",
            documents=documents,
        )

        assert fake_parse_and_split.call_count == 2
        assert fake_clean_chunks.call_count == 2
        assert fake_generate_embeddings.call_count == 1

        assert "Parsing 2 documents" == caplog.records[0].message
        assert "Embedding 6 chunks" == caplog.records[1].message
        assert "Upserted 6 documents" == caplog.records[2].message

    def test_upload_document_inconsistent(self, client, caplog, monkeypatch):
        documents = ["doc1.pdf"]
        caplog.set_level(logging.INFO)
        client.openai_client.api_key = "fake"

        parsed_docs = [
            Document(
                page_content="hello there",
                metadata={
                    "page": 0,
                    "chunk": 0,
                    "source": "something",
                },
            ),
            Document(
                page_content="again",
                metadata={
                    "page": 0,
                    "chunk": 1,
                    "source": "something",
                },
            ),
            Document(
                page_content="it is cold",
                metadata={
                    "page": 1,
                    "chunk": 0,
                    "source": "something",
                },
            ),
        ]
        fake_parse_and_split = Mock(return_value=parsed_docs)
        fake_clean_chunks = Mock(return_value=parsed_docs)
        fake_generate_embeddings = Mock(return_value=5 * [[2.2, 0.6]])

        monkeypatch.setattr(
            "whyhow_rbr.rag_qdrant.parse_and_split", fake_parse_and_split
        )
        monkeypatch.setattr(
            "whyhow_rbr.rag_qdrant.clean_chunks", fake_clean_chunks
        )
        monkeypatch.setattr(
            "whyhow_rbr.rag_qdrant.generate_embeddings",
            fake_generate_embeddings,
        )

        with pytest.raises(
            ValueError, match="Number of embeddings does not match"
        ):
            client.upload_documents(
                "some collection",
                documents=documents,
            )

        assert fake_parse_and_split.call_count == 1
        assert fake_clean_chunks.call_count == 1
        assert fake_generate_embeddings.call_count == 1

        assert "Parsing 1 documents" == caplog.records[0].message
        assert "Embedding 3 chunks" == caplog.records[1].message

    def test_query_documents(self, client, monkeypatch):
        client.openai_client.api_key = "fake"
        client.qdrant_client.collection_exists.return_value = False
        with pytest.raises(CollectionNotFoundException):
            client.query("some question", "some collection")

        assert client.qdrant_client.collection_exists.call_count == 1
        assert client.qdrant_client.query_points.call_count == 0

        client.qdrant_client.collection_exists.return_value = True
        client.qdrant_client.query_points.return_value = models.QueryResponse(
            points=[]
        )
        fake_generate_embeddings = Mock(
            return_value=10 * [[0.525, 0.532, 0.5321]]
        )
        monkeypatch.setattr(
            "whyhow_rbr.rag_qdrant.generate_embeddings",
            fake_generate_embeddings,
        )
        content = Output(
            answer="Hello world",
            contexts=[0, 1],
        )
        fake_openai_response_rv = ChatCompletion(
            id="whatever",
            choices=[
                dict(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=dict(
                        content="```json\n"
                        + content.model_dump_json()
                        + "\n```",
                        role="assistant",
                        function_call=None,
                        tool_calls=None,
                    ),
                )
            ],
            created=1710065537,
            model="gpt-4o",
            object="chat.completion",
            system_fingerprint="whatever",
            usage=dict(
                completion_tokens=20, prompt_tokens=679, total_tokens=699
            ),
        )

        client.openai_client = Mock()
        client.openai_client.api_key = "fake"
        client.openai_client.chat.completions.create.return_value = (
            fake_openai_response_rv
        )
        client.query("some question", "some collection", top_k=4)

        assert client.qdrant_client.collection_exists.call_count == 2
        assert client.qdrant_client.query_points.call_count == 1
        client.qdrant_client.query_points.assert_called_with(
            collection_name="some collection",
            limit=4,
            query=[0.525, 0.532, 0.5321],
            with_payload=True,
        )
