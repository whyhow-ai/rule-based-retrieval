"""Collection of all custom exceptions for the package."""


class IndexAlreadyExistsException(Exception):
    """Raised when the index already exists."""

    pass


class IndexNotFoundException(Exception):
    """Raised when the index is not found."""

    pass


class OpenAIException(Exception):
    """Raised when the OpenAI API returns an error."""

    pass


class CollectionNotFoundException(Exception):
    """Raised when the Collection is not found."""

    pass


class CollectionAlreadyExistsException(Exception):
    """Raised when the collection already exists."""

    pass


class CollectionCreateFailureException(Exception):
    """Raised when fail to create a new collection."""

    pass
