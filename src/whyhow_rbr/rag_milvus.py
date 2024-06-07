"""Retrieval augmented generation logic."""

import logging
import os
import pathlib
import re
import uuid
from typing import Any, Dict, List, Optional, TypedDict, cast

from langchain_core.documents import Document
from openai import OpenAI
from pydantic import BaseModel, ValidationError, Field, field_validator
from pymilvus import DataType, MilvusClient, MilvusException

from whyhow_rbr.embedding import generate_embeddings
from whyhow_rbr.exceptions import (
    CollectionAlreadyExistsException,
    CollectionCreateFailureException,
    CollectionNotFoundException,
    OpenAIException,
)
from whyhow_rbr.processing import clean_chunks, parse_and_split

logger = logging.getLogger(__name__)


class MilvusMetadata(BaseModel, extra="forbid"):
    """The metadata to be stored in Milvus.

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
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class MilvusMatch(BaseModel, extra="ignore"):
    """The match returned from Milvus.

    Attributes
    ----------
    id : str
        The ID of the document.

    score : float
        The score of the match. Its meaning depends on the metric used for
        the index.

    metadata : MilvusMetadata
        The metadata of the document.

    """

    id: str
    score: float
    metadata: MilvusMetadata


class MilvusRule(BaseModel):
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

    def to_filter(self) -> str:
        """Convert rule to Milvus filter format."""
        if not any([self.filename, self.uuid, self.page_numbers]):
            return ''

        conditions: list = []
        if self.filename is not None:
            conditions.append(f'filename == "{self.filename}"')
        if self.uuid is not None:
            conditions.append(f'id == "{self.uuid}"')
        if self.page_numbers is not None:
            conditions.append(f"page_number in {self.page_numbers}")

        filter_ = " and ".join(conditions)
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

    matches : List[dict]
        The retrieved documents from the collection.

    used_contexts : list[int]
        The indices of the matches that were actually used to answer the question.
    """

    answer: str
    matches: List[Dict[Any, Any]]
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


"""Implementing RAG by Milvus"""


class ClientMilvus:
    """Synchronous client."""

    def __init__(
        self,
        milvus_uri_key: str,
        milvus_token: str = None,
        openai_api_key: str | None = None,
    ):
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key is None:
                raise ValueError(
                    "No OPENAI_API_KEY provided must be provided."
                )

        self.openai_client = OpenAI(api_key=openai_api_key)
        self.milvus_client = MilvusClient(
            uri=milvus_uri_key,
            token=milvus_token,
        )

    def create_collection(
            self,
            collection_name: str,
            dimension: int = 1536,
    ) -> None:
        """
        Initialize a collection.

        Parameters:
            dimension: int
                The dimension of the collection.

        Returns:
            None
        """
        self.collection_name = collection_name
        if self.milvus_client.has_collection(collection_name=self.collection_name):
            raise CollectionAlreadyExistsException(f"Collection {self.collection_name} already exists")

        try:
            # create schema, with dynamic field available
            schema = self.milvus_client.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )

            # add fields to schema
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=36,
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=dimension,
            )

            # prepare index parameters
            index_params = self.milvus_client.prepare_index_params()
            index_params.add_index(
                index_type="AUTOINDEX",
                field_name="vector",
                metric_type="COSINE",
            )

            # create a collection
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level=0,
            )

        except Exception as e:
            raise CollectionCreateFailureException(
                f"Error {e} occurred while attempting to creat collection {self.collection_name}."
            )

    def drop_collection(self) -> None:
        """Delete an existing collection.

        Raises
        ------
        CollectionNotFoundException
            If the collection does not exist.
        """
        try:
            self.milvus_client.drop_collection(collection_name=self.collection_name)
        except MilvusException as e:
            raise CollectionNotFoundException(
                f"Collection {self.collection_name} not found"
            ) from e

    def upload_documents(
        self,
        documents: List[str | pathlib.Path],
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        """Upload documents to the index.

        Parameters
        ----------
        documents : list[str | pathlib.Path]
            The documents to upload.

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

        datas = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            metadata = MilvusMetadata(
                text=chunk.page_content,
                page_number=chunk.metadata["page"],
                chunk_number=chunk.metadata["chunk"],
                filename=chunk.metadata["source"],
            )
            data = {
                "id": metadata.uuid,
                "vector": embedding,
                "text": metadata.text,
                "page_number": metadata.page_number,
                "chunk_number": metadata.chunk_number,
                "filename": metadata.filename,
            }

            datas.append(data)

        response = self.milvus_client.insert(
            collection_name=self.collection_name,
            data=datas,
        )

        insert_count = response["insert_count"]
        logger.info(f"Inserted {insert_count} documents")

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

    def create_search_params(
        self,
        metric_type: str = "COSINE",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create search parameters for the Milvus search."""
        if params is None:
            params = {}

        search_params = {"metric_type": metric_type, "params": params}

        return search_params

    def query(
        self,
        question: str,
        rules: list[MilvusRule] | None = None,
        limit: int = 5,
        chat_model: str = "gpt-4-1106-preview",
        chat_temperature: float = 0.0,
        chat_max_tokens: int = 1000,
        chat_seed: int = 2,
        embedding_model: str = "text-embedding-3-small",
        process_rules_separately: bool = True,
        keyword_trigger: bool = False,
        **kwargs: Dict[str, Any],
    ) -> QueryReturnType:
        """Query the index.

        Parameters
        ----------
        question : str
            The question to ask.

        limit : int
            The maximum number of answers to return.

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

        # size of 1024
        question_embedding = generate_embeddings(
            openai_api_key=self.openai_client.api_key,
            chunks=[question],
            model=embedding_model,
        )

        matches = []
        match_texts: List[str] = []

        # Check if there are any rule filters, and if not, proceed with a default query
        if not rule_filters:
            # Perform a default query
            query_response = self.milvus_client.search(
                collection_name=self.collection_name,
                limit=limit,
                data=question_embedding,
                output_fields=["*"],
            )
            matches = [
                MilvusMatch(
                    id=m['id'],
                    score=m['distance'],
                    metadata=MilvusMetadata(
                        text=m['entity']['text'],
                        page_number = m['entity']['page_number'],
                        chunk_number=m['entity']['chunk_number'],
                        filename=m['entity']['filename'],
                    )
                ) for m in query_response[0]
            ]
            match_texts = [m.metadata.text for m in matches]

        else:

            if process_rules_separately:
                for rule_filter in rule_filters:
                    query_response = self.milvus_client.search(
                        collection_name=self.collection_name,
                        data=question_embedding,
                        filter=rule_filter,
                        limit=limit,
                        output_fields=["*"],
                    )
                    matches = [
                        MilvusMatch(
                            id=m['id'],
                            score=m['distance'],
                            metadata=MilvusMetadata(
                                text=m['entity']['text'],
                                page_number=m['entity']['page_number'],
                                chunk_number=m['entity']['chunk_number'],
                                filename=m['entity']['filename'],
                            )
                        ) for m in query_response[0]
                    ]
                    match_texts = [m.metadata.text for m in matches]
                match_texts = list(set(match_texts))  # Ensure unique match texts
            else:
                if rule_filters:
                    rule_filter = " or ".join(rule_filters)

                    query_response = self.milvus_client.search(
                        collection_name=self.collection_name,
                        data=question_embedding,
                        filter=rule_filter,
                        limit=limit,
                        output_fields=["*"],
                    )
                    matches = [
                        MilvusMatch(
                            id=m['id'],
                            score=m['distance'],
                            metadata=MilvusMetadata(
                                text=m['entity']['text'],
                                page_number=m['entity']['page_number'],
                                chunk_number=m['entity']['chunk_number'],
                                filename=m['entity']['filename'],
                            )
                        ) for m in query_response[0]
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
