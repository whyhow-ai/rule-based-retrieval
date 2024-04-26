"""Example of creating a Pinecone index and uploading documents to it."""

import logging

from pinecone import PodSpec

from whyhow_rbr import Client, IndexNotFoundException

# Parameters
index_name = ""  # Replace with your index name
namespace = ""  # Replace with your namespace name
pdfs = [
    "",
    "",
]  # Replace with the paths to your PDFs, e.g. ["path/to/pdf1.pdf", "path/to/pdf2.pdf
logging_level = logging.INFO

# Logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("create_index")
logger.setLevel(logging_level)

# Define OPENAI_API_KEY and PINECONE_API_KEY as environment variables
client = Client()

try:
    index = client.get_index(index_name)
    logger.info(f"Index {index_name} already exists, reusing it")
except IndexNotFoundException:
    spec = PodSpec(environment="gcp-starter")
    index = client.create_index(index_name, spec=spec)
    logger.info(f"Index {index_name} created")

client.upload_documents(index=index, documents=pdfs, namespace=namespace)
