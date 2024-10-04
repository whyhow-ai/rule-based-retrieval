"""Example demonostating how to perform RAG."""

import logging

from openai import OpenAI
from qdrant_client import QdrantClient

from src.whyhow_rbr.rag_qdrant import Client, Rule

# Parameters
collection_name = "<collection_name>"
question = ""  # Replace with your question
logging_level = logging.INFO  # Set to logging.DEBUG for more verbosity
top_k = 5

# Logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("querying")
logger.setLevel(logging_level)
logging.getLogger("whyhow_rbr").setLevel(logging_level)


client = Client(
    OpenAI(),  # Set OPENAI_API_KEY environment variable
    QdrantClient(url="http://localhost:6333"),
)

rules = [
    Rule(
        # Replace with your filename
        filename="name/of/pdf_1.pdf",
        page_numbers=[2],
        keywords=["keyword1", "keyword2"],
    ),
    Rule(
        # Replace with your filename
        filename="name/of/pdf_1.pdf",
        page_numbers=[1],
        keywords=[],
    ),
]

result = client.query(
    question=question,
    collection_name=collection_name,
    rules=rules,
    top_k=top_k,
    process_rules_separately=False,
    keyword_trigger=False,
)
answer = result["answer"]


logger.info(f"Answer: {answer}")
