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
        filename="doc1.pdf",  # Replace with your filename
        page_numbers=[2],
        keywords=[],
    ),
    Rule(
        filename="doc2.pdf",  # Replace with your filename
        page_numbers=[26],
        keywords=['word', 'test'],
        author="John Doe",
        subject="Research Paper",
        creation_date="2022-01-01",
        modification_date="2022-03-15",
        pages=30,
        file_size=1024000,
        pdf_version="1.7",
    ),
]

result = client.query(
    question=question,
    index=index,
    namespace=namespace,
    rules=rules,
    top_k=top_k,
    process_rules_separately=False,
    keyword_trigger=False
)
answer = result["answer"]


logger.info(f"Answer: {answer}")
