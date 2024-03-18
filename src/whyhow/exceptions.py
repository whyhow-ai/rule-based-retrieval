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
