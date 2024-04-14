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


class SchemaCreateFailureException(Exception):
    """Raised when fail to create a new schema."""

    pass


class CollectionCreateFailureException(Exception):
    """Raised when fail to create a new collection."""

    pass


class AddSchemaFieldFailureException(Exception):
    """Raised when fail to add a field to schema."""

    pass


class PartitionCreateFailureException(Exception):
    """Raised when fail to create a partition."""

    pass


class PartitionDropFailureException(Exception):
    """Raised when fail to drop a partition."""

    pass


class PartitionListFailureException(Exception):
    """Raised when fail to list all partitions."""

    pass