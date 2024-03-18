"""Retrieval augmented generation logic."""

import logging
import os
import pathlib
import re
import uuid
from typing import Any, Literal, TypedDict, cast

from langchain_core.documents import Document
from openai import OpenAI
from pinecone import (
    Index,
    NotFoundException,
    Pinecone,
    PodSpec,
    ServerlessSpec,
)
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from whyhow.embedding import generate_embeddings
from whyhow.exceptions import (
    IndexAlreadyExistsException,
    IndexNotFoundException,
    OpenAIException,
)
from whyhow.processing import clean_chunks, parse_and_split

logger = logging.getLogger(__name__)


# Defaults
DEFAULT_SPEC = ServerlessSpec(cloud="aws", region="us-west-2")


# Custom classes
class PineconeMetadata(BaseModel, extra="forbid"):
    """The metadata to be stored in Pinecone.

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

    uuid : str
        The UUID of the document. Note that this is not required to be
        provided when creating the metadata. It is generated automatically
        when creating the PineconeDocument.
    """

    text: str
    page_number: int
    chunk_number: int
    filename: str
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class PineconeDocument(BaseModel, extra="forbid"):
    """The actual document to be stored in Pinecone.

    Attributes
    ----------
    metadata : PineconeMetadata
        The metadata of the document.

    values : list[float] | None
        The embedding of the document. The None is used when querying
        the index since the values are not needed. At upsert time, the
        values are required.

    id : str | None
        The human-readable identifier of the document. This is generated
        automatically when creating the PineconeDocument unless it is
        provided.

    """

    metadata: PineconeMetadata
    values: list[float] | None = None
    id: str | None = None

    @model_validator(mode="after")
    def generate_human_readable_id(self) -> "PineconeDocument":
        """Generate a human-readable identifier for the document."""
        if self.id is None:
            meta = self.metadata
            hr_id = f"{meta.filename}-{meta.page_number}-{meta.chunk_number}"
            self.id = hr_id

        return self


class PineconeMatch(BaseModel, extra="ignore"):
    """The match returned from Pinecone.

    Attributes
    ----------
    id : str
        The ID of the document.

    score : float
        The score of the match. Its meaning depends on the metric used for
        the index.

    metadata : PineconeMetadata
        The metadata of the document.

    """

    id: str
    score: float
    metadata: PineconeMetadata


class Rule(BaseModel):
    """Retrieval rule.

    The rule is used to filter the documents in the index.

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

    def convert_empty_str_to_none(cls, s: list[str] | None) -> list[str] | None:
        """Convert empty string list to None."""
        if s is not None and not s:
            return None
        return s

    def to_filter(self) -> dict[str, list[dict[str, Any]]] | None:
        """Convert rule to Pinecone filter format."""
        if not any([self.filename, self.uuid, self.page_numbers]):
            return None

        conditions: list[dict[str, Any]] = []
        if self.filename is not None:
            conditions.append({"filename": {"$eq": self.filename}})
        if self.uuid is not None:
            conditions.append({"uuid": {"$eq": self.uuid}})
        if self.page_numbers is not None:
            conditions.append({"page_number": {"$in": self.page_numbers}})

        filter_ = {"$and": conditions}
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

# Custom types
Metric = Literal["cosine", "euclidean", "dotproduct"]


class QueryReturnType(TypedDict):
    """The return type of the query method.

    Attributes
    ----------
    answer : str
        The answer to the question.

    matches : list[dict[str, Any]]
        The retrieved documents from the index.

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
    """Synchronous client."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        pinecone_api_key: str | None = None,
    ):
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key is None:
                raise ValueError(
                    "No OPENAI_API_KEY provided must be provided."
                )

        if pinecone_api_key is None:
            pinecone_api_key = os.environ.get("PINECONE_API_KEY")
            if pinecone_api_key is None:
                raise ValueError("No PINECONE_API_KEY provided")

        self.openai_client = OpenAI(api_key=openai_api_key)
        self.pinecone_client = Pinecone(api_key=pinecone_api_key)

    def get_index(self, name: str) -> Index:
        """Get an existing index.

        Parameters
        ----------
        name : str
            The name of the index.


        Returns
        -------
        Index
            The index.

        Raises
        ------
        IndexNotFoundException
            If the index does not exist.

        """
        try:
            index = self.pinecone_client.Index(name)
        except NotFoundException as e:
            raise IndexNotFoundException(f"Index {name} does not exist") from e

        return index

    def create_index(
        self,
        name: str,
        dimension: int = 1536,
        metric: Metric = "cosine",
        spec: ServerlessSpec | PodSpec | None = None,
    ) -> Index:
        """Create a new index.

        If the index does not exist, it creates a new index with the specified.

        Parameters
        ----------
        name : str
            The name of the index.

        dimension : int
            The dimension of the index.

        metric : Metric
            The metric of the index.

        spec : ServerlessSpec | PodSpec | None
            The spec of the index. If None, it uses the default spec.

        Raises
        ------
        IndexAlreadyExistsException
            If the index already exists.

        """
        try:
            self.get_index(name)
        except IndexNotFoundException:
            pass
        else:
            raise IndexAlreadyExistsException(f"Index {name} already exists")

        if spec is None:
            spec = DEFAULT_SPEC
            logger.info(f"Using default spec {spec}")

        self.pinecone_client.create_index(
            name=name, dimension=dimension, metric=metric, spec=spec
        )
        index = self.pinecone_client.Index(name)

        return index

    def upload_documents(
        self,
        index: Index,
        documents: list[str | pathlib.Path],
        namespace: str,
        embedding_model: str = "text-embedding-3-small",
        batch_size: int = 100,
    ) -> None:
        """Upload documents to the index.

        Parameters
        ----------
        index : Index
            The index.

        documents : list[str | pathlib.Path]
            The documents to upload.

        namespace : str
            The namespace within the index to use.

        batch_size : int
            The number of documents to upload at a time.

        embedding_model : str
            The OpenAI embedding model to use.

        """
        # don't allow for duplicate documents
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

        # create PineconeDocuments
        pinecone_documents = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            metadata = PineconeMetadata(
                text=chunk.page_content,
                page_number=chunk.metadata["page"],
                chunk_number=chunk.metadata["chunk"],
                filename=chunk.metadata["source"],
            )
            pinecone_document = PineconeDocument(
                values=embedding,
                metadata=metadata,
            )
            pinecone_documents.append(pinecone_document)

        upsert_documents = [d.model_dump() for d in pinecone_documents]

        response = index.upsert(
            upsert_documents, namespace=namespace, batch_size=batch_size
        )
        n_upserted = response["upserted_count"]
        logger.info(f"Upserted {n_upserted} documents")

    def clean_text(
        self,
        text: str
    ) -> str:
        """Return a lower case version of text with punctuation removed.

        Parameters
        ----------
        text : str
            The raw text to be cleaned.

        Returns
        -------
        str: The cleaned text string.
        """
        text_processed = re.sub('[^0-9a-zA-Z ]+', '', text.lower())
        text_processed_further = re.sub(' +', ' ', text_processed)
        return text_processed_further

    def query(
        self,
        question: str,
        index: Index,
        namespace: str,
        rules: list[Rule] | None = None,
        top_k: int = 5,
        chat_model: str = "gpt-4-1106-preview",
        chat_temperature: float = 0.0,
        chat_max_tokens: int = 1000,
        chat_seed: int = 2,
        embedding_model: str = "text-embedding-3-small",
        process_rules_separately: bool = False,
        keyword_trigger: bool = False
    ) -> QueryReturnType:
        """Query the index.

        Parameters
        ----------
        question : str
            The question to ask.

        index : Index
            The index to query.

        namespace : str
            The namespace within the index to use.

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
            The "matches" are the "top_k" matches from the index.
            The "used_contexts" are the indices of the matches
            that were actually used to answer the question.

        Raises
        ------
        OpenAIException
            If there is an error with the OpenAI API. Some possible reasons
            include the chat model not finishing or the response not being
            valid JSON.
        """
        logger.info(f'Raw rules: {rules}')

        if rules is None:
            rules = []

        if keyword_trigger:
            triggered_rules = []
            clean_question = self.clean_text(question).split(' ')

            for rule in rules:
                if rule.keywords:
                    clean_keywords = [self.clean_text(keyword) for keyword in rule.keywords]

                    if bool(set(clean_keywords) & set(clean_question)):
                        triggered_rules.append(rule)

            rules = triggered_rules

        rule_filters = [rule.to_filter() for rule in rules if rule is not None]

        question_embedding = generate_embeddings(
            openai_api_key=self.openai_client.api_key,
            chunks=[question],
            model=embedding_model,
        )[0]

        matches = []  # Initialize matches outside the loop to collect matches from all queries
        match_texts = []

        # Check if there are any rule filters, and if not, proceed with a default query
        if not rule_filters:
            # Perform a default query
            query_response = index.query(
                namespace=namespace,
                top_k=top_k,
                vector=question_embedding,
                filter=None,  # No specific filter, or you can define a default filter as per your application's logic
                include_metadata=True,
            )
            matches = [
                PineconeMatch(**m.to_dict()) for m in query_response["matches"]
            ]
            match_texts = [m.metadata.text for m in matches]

        else:

            if process_rules_separately:
                for rule_filter in rule_filters:
                    if rule_filter:
                        query_response = index.query(
                            namespace=namespace,
                            top_k=top_k,
                            vector=question_embedding,
                            filter=rule_filter,
                            include_metadata=True,
                        )
                        matches.extend([
                            PineconeMatch(**m.to_dict()) for m in query_response["matches"]
                        ])
                        match_texts += [m.metadata.text for m in matches]
                match_texts = list(set(match_texts))  # Ensure unique match texts
            else:
                if rule_filters:
                    combined_filters = []
                    for rule_filter in rule_filters:
                        if rule_filter:
                            combined_filters.append(rule_filter)

                    rule_filter = {"$or": combined_filters} if combined_filters else None
                else:
                    rule_filter = None  # Fallback to a default query when no rules are provided or valid

                if rule_filter is not None:
                    query_response = index.query(
                        namespace=namespace,
                        top_k=top_k,
                        vector=question_embedding,
                        filter=rule_filter,
                        include_metadata=True,
                    )
                    matches = [
                        PineconeMatch(**m.to_dict()) for m in query_response["matches"]
                    ]
                    match_texts = [m.metadata.text for m in matches]

        # Proceed to create prompt, send it to OpenAI, and handle the response
        prompt = self.create_prompt(question, match_texts)
        response = self.openai_client.chat.completions.create(
            model=chat_model,
            seed=chat_seed,
            temperature=chat_temperature,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=chat_max_tokens,
        )

        output = self.process_response(response)

        return_dict: QueryReturnType = {
            "answer": output.answer,
            "matches": [m.model_dump() for m in matches],
            "used_contexts": output.contexts,
        }

        return return_dict

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
