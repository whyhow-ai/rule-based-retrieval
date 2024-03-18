"""Collection of utilities for working with embeddings."""

from langchain_openai import OpenAIEmbeddings


def generate_embeddings(
    openai_api_key: str,
    chunks: list[str],
    model: str = "text-embedding-3-small",
) -> list[list[float]]:
    """Generate embeddings for a list of chunks.

    Parameters
    ----------
    openai_api_key : str
        OpenAI API key.

    chunks : list[str]
        List of chunks to generate embeddings for.

    model : str
        OpenAI model to use for generating embeddings.

    Returns
    -------
    list[list[float]]
        List of embeddings for each chunk.

    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=model)  # type: ignore[call-arg]
    embeddings_array = embeddings.embed_documents(chunks)

    return embeddings_array
