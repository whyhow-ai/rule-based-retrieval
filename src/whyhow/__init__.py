"""SDK."""

from whyhow.exceptions import (
    IndexAlreadyExistsException,
    IndexNotFoundException,
)
from whyhow.rag import Client, Rule

__version__ = "v0.1.2"
__all__ = [
    "Client",
    "IndexAlreadyExistsException",
    "IndexNotFoundException",
    "Rule",
]
