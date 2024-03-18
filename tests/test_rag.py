import logging
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pinecone import Index, NotFoundException, Pinecone

from whyhow.exceptions import (
    IndexAlreadyExistsException,
    IndexNotFoundException,
    OpenAIException,
)
from whyhow.rag import Client, Output, PineconeDocument, PineconeMetadata, Rule


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
        assert rule.to_filter() == {
            "$and": [
                {"filename": {"$eq": "hello.pdf"}},
            ]
        }

        # only page_numbers
        rule = Rule(page_numbers=[1, 2])
        assert rule.to_filter() == {
            "$and": [
                {"page_number": {"$in": [1, 2]}},
            ]
        }

        # everything
        rule = Rule(filename="hello.pdf", page_numbers=[1, 2], uuid="123")

        assert rule.to_filter() == {
            "$and": [
                {"filename": {"$eq": "hello.pdf"}},
                {"uuid": {"$eq": "123"}},
                {"page_number": {"$in": [1, 2]}},
            ]
        }


class TestPineconeDocument:
    def test_generate_id(self):
        metadata = PineconeMetadata(
            text="hello world",
            page_number=1,
            chunk_number=0,
            filename="hello.pdf",
        )
        doc = PineconeDocument(
            values=[0.2, 0.3],
            metadata=metadata,
        )

        assert doc.id == "hello.pdf-1-0"

    def test_provide_id(self):
        metadata = PineconeMetadata(
            text="hello world",
            page_number=1,
            chunk_number=0,
            filename="hello.pdf",
        )
        doc = PineconeDocument(
            values=[0.2, 0.3],
            metadata=metadata,
            id="custom_id",
        )

        assert doc.id == "custom_id"


@pytest.fixture(name="client")
def patched_client(monkeypatch):
    """Generate a client instance with patched OpenAI and Pinecone clients."""
    monkeypatch.setenv("OPENAI_API_KEY", "secret_openai")
    monkeypatch.setenv("PINECONE_API_KEY", "secret_pinecone")

    fake_pinecone_instance = Mock(spec=Pinecone)
    fake_pinecone_class = Mock(return_value=fake_pinecone_instance)

    fake_openai_instance = Mock(spec=OpenAI)
    fake_openai_class = Mock(return_value=fake_openai_instance)

    monkeypatch.setattr("whyhow.rag.Pinecone", fake_pinecone_class)
    monkeypatch.setattr("whyhow.rag.OpenAI", fake_openai_class)

    client = Client()

    assert isinstance(client.openai_client, Mock)
    assert isinstance(client.pinecone_client, Mock)

    return client


class TestClient:
    def test_no_openai_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("PINECONE_API_KEY", "whatever")

        with pytest.raises(ValueError, match="No OPENAI_API_KEY"):
            Client()

    def test_no_pinecone_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "whatever")
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)

        with pytest.raises(ValueError, match="No PINECONE_API_KEY"):
            Client()

    def test_correct_instantiation(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "secret_openai")
        monkeypatch.setenv("PINECONE_API_KEY", "secret_pinecone")

        fake_pinecone_instance = Mock(spec=Pinecone)
        fake_pinecone_class = Mock(return_value=fake_pinecone_instance)

        fake_openai_instance = Mock(spec=OpenAI)
        fake_openai_class = Mock(return_value=fake_openai_instance)

        monkeypatch.setattr("whyhow.rag.Pinecone", fake_pinecone_class)
        monkeypatch.setattr("whyhow.rag.OpenAI", fake_openai_class)

        client = Client()

        assert client.openai_client == fake_openai_instance
        assert client.pinecone_client == fake_pinecone_instance

        assert fake_openai_class.call_count == 1
        args, kwargs = fake_openai_class.call_args
        assert args == ()
        assert kwargs == {"api_key": "secret_openai"}

        assert fake_pinecone_class.call_count == 1
        args, kwargs = fake_pinecone_class.call_args
        assert args == ()
        assert kwargs == {"api_key": "secret_pinecone"}

    def test_get_index(self, client):
        client.pinecone_client.Index.return_value = Index("foo", "bar")

        index = client.get_index("something")
        assert isinstance(index, Index)

        def side_effect(*args, **kwargs):
            raise NotFoundException("Index not found")

        client.pinecone_client.Index.side_effect = side_effect

        with pytest.raises(IndexNotFoundException, match="Index something"):
            client.get_index("something")

    def test_create_index(self, client, monkeypatch):
        # index does not exist
        def side_effect(*args, **kwargs):
            raise IndexNotFoundException("Index not found")

        monkeypatch.setattr(client, "get_index", Mock(side_effect=side_effect))
        client.create_index("new_index")

        assert client.pinecone_client.create_index.call_count == 1
        assert client.pinecone_client.Index.call_count == 1

        # index exists already
        monkeypatch.setattr(client, "get_index", Mock())

        with pytest.raises(
            IndexAlreadyExistsException, match="Index new_index"
        ):
            client.create_index("new_index")

        assert client.pinecone_client.create_index.call_count == 1
        assert client.pinecone_client.Index.call_count == 1

    def test_upload_documents_nothing(self, client, caplog):
        caplog.set_level(logging.INFO)
        client.upload_documents(
            index="index",
            namespace="namespace",
            documents=[],
        )

        captured = caplog.records[0]

        assert captured.levelname == "INFO"
        assert "No documents to upload" in captured.message

    def test_upload_document(self, client, caplog, monkeypatch):
        caplog.set_level(logging.INFO)
        documents = ["doc1.pdf", "doc2.pdf"]

        # mocking
        client.openai_client.api_key = "fake"
        fake_index = Mock()
        fake_index.upsert = Mock(return_value={"upserted_count": 6})

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

        monkeypatch.setattr("whyhow.rag.parse_and_split", fake_parse_and_split)
        monkeypatch.setattr("whyhow.rag.clean_chunks", fake_clean_chunks)
        monkeypatch.setattr(
            "whyhow.rag.generate_embeddings", fake_generate_embeddings
        )

        client.upload_documents(
            index=fake_index,
            namespace="great_namespace",
            documents=documents,
        )

        # assertions mocks
        assert fake_parse_and_split.call_count == 2
        assert fake_clean_chunks.call_count == 2
        assert fake_generate_embeddings.call_count == 1
        assert fake_index.upsert.call_count == 1

        # assertions logging
        assert "Parsing 2 documents" == caplog.records[0].message
        assert "Embedding 6 chunks" == caplog.records[1].message
        assert "Upserted 6 documents" == caplog.records[2].message

    def test_upload_document_inconsistent(self, client, caplog, monkeypatch):
        documents = ["doc1.pdf"]
        caplog.set_level(logging.INFO)

        # mocking
        client.openai_client.api_key = "fake"
        fake_index = Mock()

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

        monkeypatch.setattr("whyhow.rag.parse_and_split", fake_parse_and_split)
        monkeypatch.setattr("whyhow.rag.clean_chunks", fake_clean_chunks)
        monkeypatch.setattr(
            "whyhow.rag.generate_embeddings", fake_generate_embeddings
        )

        with pytest.raises(
            ValueError, match="Number of embeddings does not match"
        ):
            client.upload_documents(
                index=fake_index,
                namespace="great_namespace",
                documents=documents,
            )

        # assertions mocks
        assert fake_parse_and_split.call_count == 1
        assert fake_clean_chunks.call_count == 1
        assert fake_generate_embeddings.call_count == 1
        assert fake_index.upsert.call_count == 0

        # assertions logging
        assert "Parsing 1 documents" == caplog.records[0].message
        assert "Embedding 3 chunks" == caplog.records[1].message

    def test_query_no_rules_json_header(self, client, monkeypatch):
        # mocking embedding
        fake_generate_embeddings = Mock(return_value=[[0.2, 0.3]])

        # mocking pinecone related stuff
        fake_index = Mock()
        fake_match = Mock()
        fake_match.to_dict.return_value = {
            "id": "doc1",
            "score": 0.8,
            "metadata": {
                "filename": "hello.pdf",
                "page_number": 1,
                "chunk_number": 0,
                "text": "hello world",
                "uuid": "123",
            },
        }
        fake_query_response = {
            "matches": [fake_match, fake_match, fake_match],
        }
        fake_index.query = Mock(return_value=fake_query_response)

        # mocking openai related stuff
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
            model="gpt-3.5-turbo-0125",
            object="chat.completion",
            system_fingerprint="whatever",
            usage=dict(
                completion_tokens=20, prompt_tokens=679, total_tokens=699
            ),
        )

        client.openai_client = (
            Mock()
        )  # for some reason spec is not working correctly
        client.openai_client.api_key = "whatever"
        client.openai_client.chat.completions.create.return_value = (
            fake_openai_response_rv
        )

        monkeypatch.setattr(client, "get_index", Mock(return_value=fake_index))
        monkeypatch.setattr(
            "whyhow.rag.generate_embeddings", fake_generate_embeddings
        )

        final_result = client.query(
            question="How are you?",
            index=fake_index,
            namespace="great_namespace",
        )

        assert fake_index.query.call_count == 1

        expected_final_result = {
            "answer": "Hello world",
            "matches": [
                {
                    "id": "doc1",
                    "metadata": {
                        "chunk_number": 0,
                        "filename": "hello.pdf",
                        "page_number": 1,
                        "text": "hello world",
                        "uuid": "123",
                    },
                    "score": 0.8,
                },
                {
                    "id": "doc1",
                    "metadata": {
                        "chunk_number": 0,
                        "filename": "hello.pdf",
                        "page_number": 1,
                        "text": "hello world",
                        "uuid": "123",
                    },
                    "score": 0.8,
                },
                {
                    "id": "doc1",
                    "metadata": {
                        "chunk_number": 0,
                        "filename": "hello.pdf",
                        "page_number": 1,
                        "text": "hello world",
                        "uuid": "123",
                    },
                    "score": 0.8,
                },
            ],
            "used_contexts": [0, 1],
        }

        assert final_result == expected_final_result

    def test_query_no_rules_no_json_header(self, client, monkeypatch):
        # mocking embedding
        fake_generate_embeddings = Mock(return_value=[[0.2, 0.3]])

        # mocking pinecone related stuff
        fake_index = Mock()
        fake_match = Mock()
        fake_match.to_dict.return_value = {
            "id": "doc1",
            "score": 0.8,
            "metadata": {
                "filename": "hello.pdf",
                "page_number": 1,
                "chunk_number": 0,
                "text": "hello world",
                "uuid": "123",
            },
        }
        fake_query_response = {
            "matches": [fake_match, fake_match, fake_match],
        }
        fake_index.query = Mock(return_value=fake_query_response)

        # mocking openai related stuff
        content = Output(
            answer="The answer is 42",
            contexts=[0, 2],
        )
        fake_openai_response_rv = ChatCompletion(
            id="whatever",
            choices=[
                dict(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=dict(
                        content=content.model_dump_json(),
                        role="assistant",
                        function_call=None,
                        tool_calls=None,
                    ),
                )
            ],
            created=1710065537,
            model="gpt-3.5-turbo-0125",
            object="chat.completion",
            system_fingerprint="whatever",
            usage=dict(
                completion_tokens=20, prompt_tokens=679, total_tokens=699
            ),
        )

        client.openai_client = (
            Mock()
        )  # for some reason spec is not working correctly
        client.openai_client.api_key = "whatever"
        client.openai_client.chat.completions.create.return_value = (
            fake_openai_response_rv
        )

        monkeypatch.setattr(client, "get_index", Mock(return_value=fake_index))
        monkeypatch.setattr(
            "whyhow.rag.generate_embeddings", fake_generate_embeddings
        )

        final_result = client.query(
            question="How are you?",
            index=fake_index,
            namespace="great_namespace",
        )

        assert fake_index.query.call_count == 1

        expected_final_result = {
            "answer": "The answer is 42",
            "matches": [
                {
                    "id": "doc1",
                    "metadata": {
                        "chunk_number": 0,
                        "filename": "hello.pdf",
                        "page_number": 1,
                        "text": "hello world",
                        "uuid": "123",
                    },
                    "score": 0.8,
                },
                {
                    "id": "doc1",
                    "metadata": {
                        "chunk_number": 0,
                        "filename": "hello.pdf",
                        "page_number": 1,
                        "text": "hello world",
                        "uuid": "123",
                    },
                    "score": 0.8,
                },
                {
                    "id": "doc1",
                    "metadata": {
                        "chunk_number": 0,
                        "filename": "hello.pdf",
                        "page_number": 1,
                        "text": "hello world",
                        "uuid": "123",
                    },
                    "score": 0.8,
                },
            ],
            "used_contexts": [0, 2],
        }

        assert final_result == expected_final_result
        prompt = client.openai_client.chat.completions.create.call_args.kwargs[
            "messages"
        ][0]["content"]

        assert prompt.count("hello world") == 3

    def test_query_with_rules_no_json_header(self, client, monkeypatch):
        # mocking embedding
        fake_generate_embeddings = Mock(return_value=[[0.2, 0.3]])

        # mocking pinecone related stuff
        fake_index = Mock()
        fake_match = Mock()
        fake_match.to_dict.return_value = {
            "id": "doc1",
            "score": 0.8,
            "metadata": {
                "filename": "hello.pdf",
                "page_number": 1,
                "chunk_number": 0,
                "text": "hello world",
                "uuid": "123",
            },
        }
        fake_query_response = {
            "matches": [fake_match, fake_match, fake_match],
        }
        fake_index.query = Mock(return_value=fake_query_response)

        # mocking openai related stuff
        content = Output(
            answer="The answer is 42",
            contexts=[0, 2],
        )
        fake_openai_response_rv = ChatCompletion(
            id="whatever",
            choices=[
                dict(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=dict(
                        content=content.model_dump_json(),
                        role="assistant",
                        function_call=None,
                        tool_calls=None,
                    ),
                )
            ],
            created=1710065537,
            model="gpt-3.5-turbo-0125",
            object="chat.completion",
            system_fingerprint="whatever",
            usage=dict(
                completion_tokens=20, prompt_tokens=679, total_tokens=699
            ),
        )

        client.openai_client = (
            Mock()
        )  # for some reason spec is not working correctly
        client.openai_client.api_key = "whatever"
        client.openai_client.chat.completions.create.return_value = (
            fake_openai_response_rv
        )

        monkeypatch.setattr(client, "get_index", Mock(return_value=fake_index))
        monkeypatch.setattr(
            "whyhow.rag.generate_embeddings", fake_generate_embeddings
        )

        _ = client.query(
            question="How are you?",
            index=fake_index,
            namespace="great_namespace",
            rules=[
                Rule(
                    filename="hello.pdf",
                    page_numbers=[1],
                ),
                Rule(
                    page_numbers=[0],
                ),
            ],
        )

        assert fake_index.query.call_count == 1
        kwargs = fake_index.query.call_args.kwargs

        assert kwargs["filter"] == {
            "$or": [
                {
                    "$and": [
                        {"filename": {"$eq": "hello.pdf"}},
                        {"page_number": {"$in": [1]}},
                    ],
                },
                {"$and": [{"page_number": {"$in": [0]}}]},
            ]
        }

    def test_query_impossible_to_decode(self, client, monkeypatch):
        # mocking embedding
        fake_generate_embeddings = Mock(return_value=[[0.2, 0.3]])

        # mocking pinecone related stuff
        fake_index = Mock()
        fake_match = Mock()
        fake_match.to_dict.return_value = {
            "id": "doc1",
            "score": 0.8,
            "metadata": {
                "filename": "hello.pdf",
                "page_number": 1,
                "chunk_number": 0,
                "text": "hello world",
                "uuid": "123",
            },
        }
        fake_query_response = {
            "matches": [fake_match, fake_match, fake_match],
        }
        fake_index.query = Mock(return_value=fake_query_response)

        # mocking openai related stuff
        fake_openai_response_rv = ChatCompletion(
            id="whatever",
            choices=[
                dict(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=dict(
                        content="This is not a JSON",
                        role="assistant",
                        function_call=None,
                        tool_calls=None,
                    ),
                )
            ],
            created=1710065537,
            model="gpt-3.5-turbo-0125",
            object="chat.completion",
            system_fingerprint="whatever",
            usage=dict(
                completion_tokens=20, prompt_tokens=679, total_tokens=699
            ),
        )

        client.openai_client = (
            Mock()
        )  # for some reason spec is not working correctly
        client.openai_client.api_key = "whatever"
        client.openai_client.chat.completions.create.return_value = (
            fake_openai_response_rv
        )

        monkeypatch.setattr(client, "get_index", Mock(return_value=fake_index))
        monkeypatch.setattr(
            "whyhow.rag.generate_embeddings", fake_generate_embeddings
        )

        with pytest.raises(OpenAIException, match="OpenAI did not return"):
            client.query(
                question="How are you?",
                index=fake_index,
                namespace="great_namespace",
            )

    def test_query_wrong_reason(self, client, monkeypatch):
        # mocking embedding
        fake_generate_embeddings = Mock(return_value=[[0.2, 0.3]])

        # mocking pinecone related stuff
        fake_index = Mock()
        fake_match = Mock()
        fake_match.to_dict.return_value = {
            "id": "doc1",
            "score": 0.8,
            "metadata": {
                "filename": "hello.pdf",
                "page_number": 1,
                "chunk_number": 0,
                "text": "hello world",
                "uuid": "123",
            },
        }
        fake_query_response = {
            "matches": [fake_match, fake_match, fake_match],
        }
        fake_index.query = Mock(return_value=fake_query_response)

        # mocking openai related stuff
        content = Output(
            answer="The answer is 42",
            contexts=[0, 2],
        )
        fake_openai_response_rv = ChatCompletion(
            id="whatever",
            choices=[
                dict(
                    finish_reason="length",
                    index=0,
                    logprobs=None,
                    message=dict(
                        content=content.model_dump_json(),
                        role="assistant",
                        function_call=None,
                        tool_calls=None,
                    ),
                )
            ],
            created=1710065537,
            model="gpt-3.5-turbo-0125",
            object="chat.completion",
            system_fingerprint="whatever",
            usage=dict(
                completion_tokens=20, prompt_tokens=679, total_tokens=699
            ),
        )

        client.openai_client = (
            Mock()
        )  # for some reason spec is not working correctly
        client.openai_client.api_key = "whatever"
        client.openai_client.chat.completions.create.return_value = (
            fake_openai_response_rv
        )

        monkeypatch.setattr(client, "get_index", Mock(return_value=fake_index))
        monkeypatch.setattr(
            "whyhow.rag.generate_embeddings", fake_generate_embeddings
        )

        with pytest.raises(OpenAIException, match="Chat did not finish"):
            client.query(
                question="How are you?",
                index=fake_index,
                namespace="great_namespace",
            )
