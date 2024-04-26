"""SDK."""

from whyhow_rbr.exceptions import (
    IndexAlreadyExistsException,
    IndexNotFoundException,
)
from whyhow_rbr.rag import Client, Rule

__version__ = "v0.1.4"
__all__ = [
    "Client",
    "IndexAlreadyExistsException",
    "IndexNotFoundException",
    "Rule",
]
