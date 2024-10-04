"""Rule based RAG with Qdrant."""

import logging
import pathlib
import re
import uuid
from typing import Any, TypedDict, cast

from langchain_core.documents import Document
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator
from qdrant_client import QdrantClient, models

from whyhow_rbr.embedding import generate_embeddings
from whyhow_rbr.exceptions import (
    CollectionAlreadyExistsException,
    CollectionNotFoundException,
    OpenAIException,
)
from whyhow_rbr.processing import clean_chunks, parse_and_split

logger = logging.getLogger(__name__)


class Metadata(BaseModel, extra="forbid"):
    """The metadata to be stored in Qdrant.

    Attributes
    ----------
    text : str
        The text of the document.

    page_number : int
        The page number of the document.

    chunk_number : int
        The chunk number of the document.

    filename : str
        The filename of the document.
    """

    text: str
    page_number: int
    chunk_number: int
    filename: str


class QdrantDocument(BaseModel, extra="forbid"):
    """The actual document to be stored in Qdrant.

    Attributes
    ----------
    metadata : Metadata
        The metadata of the document.

    vector : list[float] | None
        The vector embedding representing the document.

    id : str | None
        UUID of the document.
    """

    metadata: Metadata
    vector: list[float]
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class QdrantMatch(BaseModel, extra="ignore"):
    """The match returned from Qdrant.

    Attributes
    ----------
    id : str
        The ID of the document.

    score : float
        The score of the match. Its meaning depends on the distance used for
        the collection.

    metadata : Metadata
        The metadata of the document.

    """

    id: str | int
    score: float
    metadata: Metadata


class Rule(BaseModel):
    """Retrieval rule.

    The rule is used to filter the documents in the collection.

    Attributes
    ----------
    filename : str | None
        The filename of the document.

    uuid : str | None
        The UUID of the document.

    page_numbers : list[int] | None
        The page numbers of the document.

    keywords : list[str] | None
        The keywords to trigger a rule.
    """

    filename: str | None = None
    uuid: str | None = None
    page_numbers: list[int] | None = None
    keywords: list[str] | None = None

    @field_validator("page_numbers", mode="before")
    @classmethod
    def convert_empty_to_none(cls, v: list[int] | None) -> list[int] | None:
        """Convert empty list to None."""
        if v is not None and not v:
            return None
        return v

    @field_validator("keywords", mode="before")
    @classmethod
    def convert_empty_str_to_none(
        cls, s: list[str] | None
    ) -> list[str] | None:
        """Convert empty string list to None."""
        if s is not None and not s:
            return None
        return s

    def to_filter(self) -> models.Filter | None:
        """Convert rule to Qdrant filter format."""
        if not any([self.filename, self.uuid, self.page_numbers]):
            return None

        conditions: list[models.Condition] = []
        if self.filename is not None:
            conditions.append(
                models.FieldCondition(
                    key="filename",
                    match=models.MatchValue(value=self.filename),
                )
            )
        if self.uuid is not None:
            conditions.append(
                models.HasIdCondition(has_id=[self.uuid]),
            )
        if self.page_numbers is not None:
            conditions.append(
                models.FieldCondition(
                    key="page_number",
                    match=models.MatchAny(any=self.page_numbers),
                )
            )

        filter_ = models.Filter(must=conditions)
        return filter_


class Input(BaseModel):
    """Example input for the prompt.

    Attributes
    ----------
    question : str
        The question to ask.

    contexts : list[str]
        The contexts to use for answering the question.
    """

    question: str
    contexts: list[str]


class Output(BaseModel):
    """Example output for the prompt.

    Attributes
    ----------
    answer : str
        The answer to the question.

    contexts : list[int]
        The indices of the contexts that were used to answer the question.
    """

    answer: str
    contexts: list[int]


input_example_1 = Input(
    question="What is the capital of France?",
    contexts=[
        "The capital of France is Paris.",
        "The capital of France is not London.",
        "Paris is beautiful and it is also the capital of France.",
    ],
)
output_example_1 = Output(answer="Paris", contexts=[0, 2])

input_example_2 = Input(
    question="What are the impacts of climate change on global agriculture?",
    contexts=[
        "Climate change can lead to more extreme weather patterns, affecting crop yields.",
        "Rising sea levels due to climate change can inundate agricultural lands in coastal areas, reducing arable land.",
        "Changes in temperature and precipitation patterns can shift agricultural zones, impacting food security.",
    ],
)

output_example_2 = Output(
    answer="Variable impacts including altered weather patterns, reduced arable land, shifting agricultural zones, increased pests and diseases, with potential mitigation through technology and sustainable practices",
    contexts=[0, 1, 2],
)

input_example_3 = Input(
    question="How has the concept of privacy evolved with the advent of digital technology?",
    contexts=[
        "Digital technology has made it easier to collect, store, and analyze personal data, raising privacy concerns.",
        "Social media platforms and smartphones often track user activity and preferences, leading to debates over consent and data ownership.",
        "Encryption and secure communication technologies have evolved as means to protect privacy in the digital age.",
        "Legislation like the GDPR in the EU has been developed to address privacy concerns and regulate data handling by companies.",
        "The concept of privacy is increasingly being viewed through the lens of digital rights and cybersecurity.",
    ],
)

output_example_3 = Output(
    answer="Evolving with challenges due to data collection and analysis, changes in legislation, and advancements in encryption and security, amidst ongoing debates over consent and data ownership",
    contexts=[0, 1, 2, 3, 4],
)


class QueryReturnType(TypedDict):
    """The return type of the query method.

    Attributes
    ----------
    answer : str
        The answer to the question.

    matches : list[dict[str, Any]]
        The retrieved documents from the collection.

    used_contexts : list[int]
        The indices of the matches that were actually used to answer the question.
    """

    answer: str
    matches: list[dict[str, Any]]
    used_contexts: list[int]


PROMPT_START = f"""\
You are a helpful assistant. I will give you a question and provide multiple
context documents. You will need to answer the question based on the contexts
and also specify in which context(s) you found the answer.
If you don't find the answer in the context, you can use your own knowledge, however,
in that case, the contexts array should be empty.

Both the input and the output are JSON objects.

# EXAMPLE INPUT
# ```json
# {input_example_1.model_dump_json()}
# ```

# EXAMPLE OUTPUT
# ```json
# {output_example_1.model_dump_json()}

# EXAMPLE INPUT
# ```json
# {input_example_2.model_dump_json()}
# ```

# EXAMPLE OUTPUT
# ```json
# {output_example_2.model_dump_json()}

# EXAMPLE INPUT
# ```json
# {input_example_3.model_dump_json()}
# ```

# EXAMPLE OUTPUT
# ```json
# {output_example_3.model_dump_json()}

"""


class Client:
    """RBR client for Qdrant."""

    def __init__(
        self,
        oclient: OpenAI,
        qclient: QdrantClient,
    ):
        self.openai_client = oclient
        self.qdrant_client = qclient

    def create_collection(
        self,
        collection_name: str,
        size: int = 1536,
        distance: models.Distance = models.Distance.COSINE,
        **collection_kwargs: Any,
    ) -> None:
        """Create a new collection.

        Parameters
        ----------
        collection_name : str
            The name of the collection.

        size : int
            The dimension of the vectors.

        distance : Distance
            The distance metric to use for the collection.

        collection_kwargs : Any
            Additional arguments to pass to QdrantClient#create_collection.

        Raises
        ------
        CollectionAlreadyExistsException
            If the collection already exists.

        """
        if self.qdrant_client.collection_exists(collection_name):
            raise CollectionAlreadyExistsException()

        collection_opts = {
            "collection_name": collection_name,
            "vectors_config": models.VectorParams(
                size=size, distance=distance
            ),
            **collection_kwargs,
        }

        self.qdrant_client.create_collection(**collection_opts)

    def upload_documents(
        self,
        collection_name: str,
        documents: list[str | pathlib.Path],
        embedding_model: str = "text-embedding-3-small",
        batch_size: int = 64,
    ) -> None:
        """Upload documents to the collection.

        Parameters
        ----------
        collection_name : str
            The name of the collection.

        documents : list[str | pathlib.Path]
            The documents to upload.

        embedding_model : str
            The OpenAI embedding model to use.

        batch_size : int
            The number of documents to upload at a time.
        """
        if not self.qdrant_client.collection_exists(collection_name):
            raise CollectionNotFoundException()

        documents = list(set(documents))
        if not documents:
            logger.info("No documents to upload")
            return

        logger.info(f"Parsing {len(documents)} documents")
        all_chunks: list[Document] = []
        for document in documents:
            chunks_ = parse_and_split(document)
            chunks = clean_chunks(chunks_)
            all_chunks.extend(chunks)

        logger.info(f"Embedding {len(all_chunks)} chunks")
        embeddings = generate_embeddings(
            openai_api_key=self.openai_client.api_key,
            chunks=[c.page_content for c in all_chunks],
            model=embedding_model,
        )

        if len(embeddings) != len(all_chunks):
            raise ValueError(
                "Number of embeddings does not match number of chunks"
            )

        qdrant_documents: list[QdrantDocument] = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            metadata = Metadata(
                text=chunk.page_content,
                page_number=chunk.metadata["page"],
                chunk_number=chunk.metadata["chunk"],
                filename=chunk.metadata["source"],
            )
            qdrant_document = QdrantDocument(
                vector=embedding, metadata=metadata
            )
            qdrant_documents.append(qdrant_document)

        points = [
            models.PointStruct(
                id=d.id, vector=d.vector, payload=d.metadata.model_dump()
            )
            for d in qdrant_documents
        ]

        self.qdrant_client.upload_points(
            collection_name, points, batch_size=batch_size
        )

        logger.info(f"Upserted {len(points)} documents")

    def clean_text(self, text: str) -> str:
        """Return a lower case version of text with punctuation removed.

        Parameters
        ----------
        text : str
            The raw text to be cleaned.

        Returns
        -------
        str: The cleaned text string.
        """
        text_processed = re.sub("[^0-9a-zA-Z ]+", "", text.lower())
        text_processed_further = re.sub(" +", " ", text_processed)
        return text_processed_further

    def query(
        self,
        question: str,
        collection_name: str,
        rules: list[Rule] | None = None,
        top_k: int = 5,
        chat_model: str = "gpt-4o",
        chat_temperature: float = 0.0,
        chat_max_tokens: int = 1000,
        chat_seed: int = 2,
        embedding_model: str = "text-embedding-3-small",
        process_rules_separately: bool = False,
        keyword_trigger: bool = False,
    ) -> QueryReturnType:
        """Query the collection.

        Parameters
        ----------
        question : str
            The question to ask.

        collection_name : str
            The name of the collection.

        rules : list[Rule] | None
            The rules to use for filtering the documents.

        top_k : int
            The number of matches to return per rule.

        chat_model : str
            The OpenAI chat model to use.

        chat_temperature : float
            The temperature for the chat model.

        chat_max_tokens : int
            The maximum number of tokens for the chat model.

        chat_seed : int
            The seed for the chat model.

        embedding_model : str
            The OpenAI embedding model to use.

        process_rules_separately : bool, optional
            Whether to process each rule individually and combine the results at the end.
            When set to True, each rule will be run independently, ensuring that every rule
            returns results. When set to False (default), all rules will be run as one joined
            query, potentially allowing one rule to dominate the others.
            Default is False.

        keyword_trigger : bool, optional
            Whether to trigger rules based on keyword matches in the question.
            Default is False.

        Returns
        -------
        QueryReturnType
            Dictionary with keys "answer", "matches", and "used_contexts".
            The "answer" is the answer to the question.
            The "matches" are the "top_k" matches from the collection.
            The "used_contexts" are the indices of the matches
            that were actually used to answer the question.

        Raises
        ------
        OpenAIException
            If there is an error with the OpenAI API. Some possible reasons
            include the chat model not finishing or the response not being
            valid JSON.

        CollectionNotFoundException
            If the collection does not exist in Qdrant.
        """
        if not self.qdrant_client.collection_exists(collection_name):
            raise CollectionNotFoundException()

        logger.info(f"Raw rules: {rules}")

        if rules is None:
            rules = []

        if keyword_trigger:
            clean_question = set(self.clean_text(question).split(" "))
            rules = [
                rule
                for rule in rules
                if rule.keywords
                and set(map(self.clean_text, rule.keywords)) & clean_question
            ]

        rule_filters = [rule.to_filter() for rule in rules if rule.to_filter()]

        question_embedding = generate_embeddings(
            openai_api_key=self.openai_client.api_key,
            chunks=[question],
            model=embedding_model,
        )[0]

        matches, match_texts = [], []

        if not rule_filters:
            query_response = self.qdrant_client.query_points(
                collection_name=collection_name,
                limit=top_k,
                query=question_embedding,
                with_payload=True,
            ).points
            matches = [
                QdrantMatch(
                    id=p.id,
                    score=p.score,
                    metadata=Metadata(**p.payload),  # type: ignore
                )
                for p in query_response
            ]
            match_texts = [m.metadata.text for m in matches]
        else:
            if process_rules_separately:
                for rule_filter in rule_filters:
                    query_response = self.qdrant_client.query_points(
                        collection_name=collection_name,
                        limit=top_k,
                        query=question_embedding,
                        query_filter=rule_filter,
                        with_payload=True,
                    ).points
                    matches.extend(
                        QdrantMatch(
                            id=p.id,
                            score=p.score,
                            metadata=Metadata(**p.payload),  # type: ignore
                        )
                        for p in query_response
                    )
                    match_texts.extend(m.metadata.text for m in matches)
                match_texts = list(
                    set(match_texts)
                )  # Ensure unique match texts
            else:
                combined_filter = models.Filter(must=rule_filters)  # type: ignore
                query_response = self.qdrant_client.query_points(
                    collection_name=collection_name,
                    limit=top_k,
                    query=question_embedding,
                    query_filter=combined_filter,
                    with_payload=True,
                ).points
                matches = [
                    QdrantMatch(
                        id=p.id,
                        score=p.score,
                        metadata=Metadata(**p.payload),  # type: ignore
                    )
                    for p in query_response
                ]
                match_texts = [m.metadata.text for m in matches]

        prompt = self.create_prompt(question, match_texts)
        response = self.openai_client.chat.completions.create(
            model=chat_model,
            seed=chat_seed,
            temperature=chat_temperature,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=chat_max_tokens,
        )

        output = self.process_response(response)

        return {
            "answer": output.answer,
            "matches": [m.model_dump() for m in matches],
            "used_contexts": output.contexts,
        }

    def create_prompt(self, question: str, match_texts: list[str]) -> str:
        """Create the prompt for the OpenAI chat completion.

        Parameters
        ----------
        question : str
            The question to ask.

        match_texts : list[str]
            The list of context strings to include in the prompt.

        Returns
        -------
        str
            The generated prompt.
        """
        input_actual = Input(question=question, contexts=match_texts)
        prompt_end = f"""
        ACTUAL INPUT
        ```json
        {input_actual.model_dump_json()}
        ```

        ACTUAL OUTPUT
        """
        return f"{PROMPT_START}\n{prompt_end}"

    def process_response(self, response: Any) -> Output:
        """Process the OpenAI chat completion response.

        Parameters
        ----------
        response : Any
            The OpenAI chat completion response.

        Returns
        -------
        Output
            The processed output.

        Raises
        ------
        OpenAIException
            If the chat model did not finish or the response is not valid JSON.
        """
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise OpenAIException(
                f"Chat did not finish. Reason: {choice.finish_reason}"
            )

        response_raw = cast(str, response.choices[0].message.content)

        if response_raw.startswith("```json"):
            start_i = response_raw.index("{")
            end_i = response_raw.rindex("}")
            response_raw = response_raw[start_i : end_i + 1]

        try:
            output = Output.model_validate_json(response_raw)
        except ValidationError as e:
            raise OpenAIException(
                f"OpenAI did not return a valid JSON: {response_raw}"
            ) from e

        return output
