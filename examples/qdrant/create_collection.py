"""Example of creating a Pinecone index and uploading documents to it."""

import logging

from openai import OpenAI
from qdrant_client import QdrantClient

from src.whyhow_rbr.rag_qdrant import Client

# Parameters
collection_name = "<collection_name>"  # Replace with your collection name
pdfs = (
    []
)  # Replace with the paths to your PDFs, e.g. ["path/to/pdf1.pdf", "path/to/pdf2.pdf
logging_level = logging.INFO

# Logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("create_index")
logger.setLevel(logging_level)


client = Client(
    OpenAI(),  # Set OPENAI_API_KEY environment variable
    QdrantClient(url="http://localhost:6333"),
)

client.create_collection(collection_name)
client.upload_documents(collection_name, documents=pdfs)
