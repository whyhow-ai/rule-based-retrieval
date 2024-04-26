"""Example demonostating how to perform RAG."""

import logging

from whyhow_rbr import Client, Rule

# Parameters
index_name = ""  # Replace with your index name
namespace = ""  # Replace with your namespace name
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


# Define OPENAI_API_KEY and PINECONE_API_KEY as environment variables
client = Client()

index = client.get_index(index_name)
logger.info(f"Index {index_name} exists")

rules = [
    Rule(
        # Replace with your filename
        filename="doc1.pdf",
        page_numbers=[26],
        keywords=["word", "test"],
    ),
    Rule(
        # Replace with your filename
        filename="doc2.pdf",
        page_numbers=[2],
        keywords=[],
    ),
]

result = client.query(
    question=question,
    index=index,
    namespace=namespace,
    rules=rules,
    top_k=top_k,
    process_rules_separately=False,
    keyword_trigger=False,
)
answer = result["answer"]


logger.info(f"Answer: {answer}")
