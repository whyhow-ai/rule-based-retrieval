"""SDK."""

from src.whyhow_rbr.exceptions import (
    IndexAlreadyExistsException,
    IndexNotFoundException,
    OpenAIException,
    CollectionNotFoundException,
    CollectionAlreadyExistsException,
    CollectionCreateFailureException,
)
from whyhow_rbr.rag import Client, Rule
from whyhow_rbr.rag_milvus import ClientMilvus, MilvusRule

__version__ = "v0.1.4"
__all__ = [
    # Client
    "Client",
    "ClientMilvus",
    "Rule",
    "MilvusRule",

    # Error
    "IndexAlreadyExistsException",
    "IndexNotFoundException",
    "OpenAIException",
    "CollectionNotFoundException",
    "CollectionAlreadyExistsException",
    "CollectionCreateFailureException",
]
