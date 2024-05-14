"""Retrieval augmented generation logic."""

import logging
import os
import pathlib
import re
from typing import Any, Dict, List, Optional, TypedDict, cast

from langchain_core.documents import Document
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from pymilvus import CollectionSchema, DataType, MilvusClient, MilvusException
from pymilvus.milvus_client import IndexParams

from whyhow_rbr.embedding import generate_embeddings
from whyhow_rbr.exceptions import (
    AddSchemaFieldFailureException,
    CollectionAlreadyExistsException,
    CollectionCreateFailureException,
    CollectionNotFoundException,
    OpenAIException,
    PartitionCreateFailureException,
    PartitionDropFailureException,
    PartitionListFailureException,
    SchemaCreateFailureException,
)
from whyhow_rbr.processing import clean_chunks, parse_and_split

logger = logging.getLogger(__name__)


class MilvusMetadata(BaseModel):
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
    vector: List[float]


"""Custom classes for constructing prompt, output and query result with examples"""


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
        milvus_uri: str,
        milvus_token: str = None,
        milvus_db_name: Optional[str] = None,
        timeout: float | None = None,
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
            uri=milvus_uri,
            token=milvus_token,
            db_name=milvus_db_name,
            timeout=timeout,
        )

    def get_collection_stats(
        self, collection_name: str, timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get an existing collection.

        Parameters
        ----------
        collection_name : str
            The name of the collection.

        timeout : Optional[float]
            The timeout duration for this operation.
            Setting this to None indicates that this operation timeouts when any response returns or error occurs.

        Returns
        -------
        Dict
            A dictionary that contains detailed information about the specified collection.

        Raises
        ------
        CollectionNotFoundException
            If the collection does not exist.
        """
        try:
            collection_stats = self.milvus_client.describe_collection(
                collection_name, timeout
            )
        except MilvusException as e:
            raise CollectionNotFoundException(
                f"Collection {collection_name} does not exist"
            ) from e

        return collection_stats

    def create_schema(
        self,
        auto_id: bool = False,
        enable_dynamic_field: bool = True,
        **kwargs: Any,
    ) -> CollectionSchema:
        """Create a schema to add in collection.

        Parameters
        ----------
        auto_id : bool
            Whether allows the primary field to automatically increment.

        enable_dynamic_field : bool
            Whether allows Milvus saves the values of undefined fields in a dynamic field
            if the data being inserted into the target collection includes fields that are not defined in the collection's schema.

        Returns
        -------
        CollectionSchema
            A Schema instance represents the schema of a collection.

        Raises
        ------
        SchemaCreateFailureException
            If schema create failure.
        """
        try:
            schema = MilvusClient.create_schema(
                auto_id=auto_id,
                enable_dynamic_field=enable_dynamic_field,
                **kwargs,
            )
        except MilvusException as e:
            raise SchemaCreateFailureException("Schema create failure.") from e

        return schema

    def add_field(
        self,
        schema: CollectionSchema,
        field_name: str,
        datatype: DataType,
        is_primary: bool = False,
        **kwargs: Any,
    ) -> CollectionSchema:
        """Add Field to current schema.

        Parameters
        ----------
        schema : CollectionSchema
            The exist schema object.

        field_name : str
            The name of the new field.

        datatype : DataType
            The data type of the field.
            You can choose from the following options when selecting a data type for different fields:

                Primary key field: Use DataType.INT64 or DataType.VARCHAR.

                Scalar fields: Choose from a variety of options, including:

                DataType.BOOL, DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64,
                DataType.FLOAT, DataType.DOUBLE, DataType.BINARY_VECTOR, DataType.FLOAT_VECTOR,
                DataType.FLOAT16_VECTOR, __DataType.BFLOAT16_VECTOR, DataType.VARCHAR,
                DataType.JSON, and DataType.ARRAY.

                Vector fields: Select DataType.BINARY_VECTOR or DataType.FLOAT_VECTOR.

        is_primary : bool
            Whether the current field is the primary field in a collection.
            **Each collection has only one primary field.

        **kwargs : Any

            max_length (int) -
            The maximum length of the field value.
            This is mandatory for a DataType.VARCHAR field.

            element_type (str) -
            The data type of the elements in the field value.
            This is mandatory for a DataType.Array field.

            max_capacity (int) -
            The number of elements in an Array field value.
            This is mandatory for a DataType.Array field.

            dim (int) -
            The dimension of the vector embeddings.
            This is mandatory for a DataType.FLOAT_VECTOR field or a DataType.BINARY_VECTOR field.

        Returns
        -------
        CollectionSchema
            A Schema instance represents the schema of a collection.

        Raises
        ------
        AddSchemaFieldFailureException
            If schema create failure.
        """
        try:
            schema.add_field(
                field_name=field_name,
                datatype=datatype,
                is_primary=is_primary,
                **kwargs,
            )
        except MilvusException as e:
            raise AddSchemaFieldFailureException(
                f"Fail to add {field_name} to current schema."
            ) from e

        return schema

    def prepare_index_params(self) -> IndexParams:
        """Prepare an index object."""
        index_params = self.milvus_client.prepare_index_params()

        return index_params

    def add_index(
        self,
        index_params: IndexParams,
        field_name: str,
        index_type: str = "AUTOINDEX",
        index_name: Optional[str] = None,
        metric_type: str = "COSINE",
        params: Optional[Dict[str, Any]] = None,
    ) -> IndexParams:
        """Add an index to IndexParams Object.

        Parameters
        ----------
        index_params : IndexParams
            index object

        field_name : str
            The name of the target file to apply this object applies.

        index_name : str
            The name of the index file generated after this object has been applied.

        index_type : str
            The name of the algorithm used to arrange data in the specific field.

        metric_type : str
            The algorithm that is used to measure similarity between vectors. Possible values are IP, L2, and COSINE.

        params : dict
            The fine-tuning parameters for the specified index type. For details on possible keys and value ranges, refer to In-memory Index.
        """
        index_params.add_index(
            field_name=field_name,
            index_type=index_type,
            index_name=index_name,
            metric_type=metric_type,
            params=params,
        )

        return index_params

    def create_index(
        self,
        collection_name: str,
        index_params: IndexParams,
        timeout: Optional[float] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Create an index.

        Parameters
        ----------
        index_params : IndexParams
            index object

        collection_name : str
            The name of the collection.

        timeout : Optional[float]
            The maximum duration to wait for the operation to complete before timing out.
        """
        self.milvus_client.create_index(
            collection_name=collection_name,
            index_params=index_params,
            timeout=timeout,
            **kwargs,
        )

    def create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        metric_type: str = "COSINE",
        timeout: Optional[float] = None,
        schema: Optional[CollectionSchema] = None,
        index_params: Optional[IndexParams] = None,
        enable_dynamic_field: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new collection.

        If the collection does not exist, it creates a new collection with the specified.

        Parameters
        ----------
        collection_name : str
            [REQUIRED]
            The name of the collection to create.

        dimension : int
            The dimension of the vector field in the collection.
            The reason choosing 1024 as default is that the model
                "text-embedding-3-small" we use generates a size of 1024 embeddings

        metric_type : str
            The metric used to measure similarities between vector embeddings in the collection.

        timeout : Optional[float]
            The maximum duration to wait for the operation to complete before timing out.

        schema : Optional[CollectionSchema]
            Defines the structure of the collection.

        enable_dynamic_field: bool:
            True can insert data without creating a schema first.

        Raises
        ------
        CollectionAlreadyExistsException
            If the collection already exists.
        """
        try:
            # Detect whether the collection exist or not
            self.get_collection_stats(collection_name, timeout)
        except CollectionNotFoundException:
            pass
        else:
            raise CollectionAlreadyExistsException(
                f"Collection {collection_name} already exists"
            )

        try:
            self.milvus_client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                metric_type=metric_type,
                schema=schema,
                index_params=index_params,
                timeout=timeout,
                enable_dynamic_field=enable_dynamic_field,
                **kwargs,
            )
        except MilvusException as e:
            raise CollectionCreateFailureException(
                f"Collection {collection_name} fail to create"
            ) from e

    def crate_partition(
        self,
        collection_name: str,
        partition_name: str,
        timeout: Optional[float] = None,
    ) -> None:
        """Create a partition in collection.

        Parameters
        ----------
        collection_name : str
            [REQUIRED]
            The name of the collection to add partition.

        partition_name : str
            [REQUIRED]
            The name of the partition to create.

        timeout : Optional[float]
            The timeout duration for this operation.
            Setting this to None indicates that this operation timeouts when any response arrives or any error occurs.

        Raises
        ------
        PartitionCreateFailureException
            If partition create failure.
        """
        try:
            self.milvus_client.create_partition(
                collection_name=collection_name,
                partition_name=partition_name,
                timeout=timeout,
            )
        except MilvusException as e:
            raise PartitionCreateFailureException(
                f"Partition {partition_name} fail to create"
            ) from e

    def drop_partition(
        self,
        collection_name: str,
        partition_name: str,
        timeout: Optional[float] = None,
    ) -> None:
        """Drop a partition in collection.

        Parameters
        ----------
        collection_name : str
            [REQUIRED]
            The name of the collection to drop partition.

        partition_name : str
            [REQUIRED]
            The name of the partition to drop.

        timeout : Optional[float]
            The timeout duration for this operation.
            Setting this to None indicates that this operation timeouts when any response arrives or any error occurs.

        Raises
        ------
        PartitionDropFailureException
            If partition drop failure.
        """
        try:
            self.milvus_client.drop_partition(
                collection_name=collection_name,
                partition_name=partition_name,
                timeout=timeout,
            )
        except MilvusException as e:
            raise PartitionDropFailureException(
                f"Partition {partition_name} fail to drop"
            ) from e

    def list_partition(
        self, collection_name: str, timeout: Optional[float] = None
    ) -> List[str]:
        """List all partitions in the specific collection.

        Parameters
        ----------
        collection_name : str
            [REQUIRED]
            The name of the collection to add partition.

        timeout : Optional[float]
            The timeout duration for this operation.
            Setting this to None indicates that this operation timeouts when any response arrives or any error occurs.

        Returns
        -------
        partitions : list[str]
            All the partitions in that specific collection.

        Raises
        ------
        PartitionListFailureException
            If partition listing failure.
        """
        try:
            partitions = self.milvus_client.list_partitions(
                collection_name=collection_name, timeout=timeout
            )
        except MilvusException as e:
            raise PartitionListFailureException(
                f"Partitions from {collection_name} fail to list"
            ) from e

        return partitions

    def drop_collection(self, collection_name: str) -> None:
        """Delete an existing collection.

        Parameters
        ----------
        collection_name : str
            The name of the collection.

        Raises
        ------
        CollectionNotFoundException
            If the collection does not exist.
        """
        try:
            self.milvus_client.drop_collection(collection_name=collection_name)
        except MilvusException as e:
            raise CollectionNotFoundException(
                f"Collection {collection_name} not found"
            ) from e

    def upload_documents(
        self,
        collection_name: str,
        documents: List[str | pathlib.Path],
        partition_name: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        """Upload documents to the index.

        Parameters
        ----------
        collection_name : str
            The name of the collection

        documents : list[str | pathlib.Path]
            The documents to upload.

        partition_name : str | None
            The name of the partition in that collection to insert the data

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

        data = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            rawdata = MilvusMetadata(
                text=chunk.page_content,
                page_number=chunk.metadata["page"],
                chunk_number=chunk.metadata["chunk"],
                filename=chunk.metadata["source"],
                vector=embedding,
            )
            metadata = {
                "text": rawdata.text,
                "page_number": str(rawdata.page_number),
                "chunk_number": str(rawdata.chunk_number),
                "filename": rawdata.filename,
                "embedding": list(rawdata.vector),
            }

            data.append(metadata)

        response = self.milvus_client.insert(
            collection_name=collection_name,
            partition_name=partition_name,
            data=data,
        )

        insert_count = response["insert_count"]
        logger.info(f"Inserted {insert_count} documents")

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

    def search(
        self,
        question: str,
        collection_name: str,
        anns_field: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        filter: str = "",
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
        chat_model: str = "gpt-4-1106-preview",
        chat_temperature: float = 0.0,
        chat_max_tokens: int = 1000,
        chat_seed: int = 2,
        embedding_model: str = "text-embedding-3-small",
        **kwargs: Dict[str, Any],
    ) -> QueryReturnType:
        """Query the index.

        Parameters
        ----------
        collection_name : str
            Name of the collection.

        anns_field : str
            Specific Field to search on.

        question : str
            The question to ask.

        limit : int
            The maximum number of answers to return.

        output_fields : str
            The field that should return.

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
        if output_fields is None:
            output_fields = ["text", "filename", "page_number"]

        if search_params is None:
            search_params = {}

        logger.info(f"Filter: {filter} and Search params: {search_params}")

        # size of 1024
        question_embedding = generate_embeddings(
            openai_api_key=self.openai_client.api_key,
            chunks=[question],
            model=embedding_model,
        )[0]

        match_texts: List[str] = []

        results: Optional[List[Any]] = []
        i = 0
        while results is not None and i < 5:
            results = self.milvus_client.search(
                collection_name=collection_name,
                anns_field=anns_field,
                partition_names=partition_names,
                filter=filter,
                data=[question_embedding],
                output_fields=[output_fields],
                limit=limit,
                search_params=search_params,
                **kwargs,
            )
            i += 1

        if results is not None:
            for result in results:
                text = result[0]["entity"]["text"]
                match_texts.append(text)

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
            "matches": [],
            "used_contexts": output.contexts,
        }

        if results is not None and len(results) > 0:
            return_dict["matches"] = results[0]

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
