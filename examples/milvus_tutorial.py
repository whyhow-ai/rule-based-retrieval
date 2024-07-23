"""Script that demonstrates how to use the RAG model with Milvus to implement rule-based retrieval."""

from whyhow_rbr.rag_milvus import ClientMilvus, MilvusRule

# Set up your Milvus Client information
YOUR_MILVUS_LITE_FILE_PATH = "milvus_demo.db"  # random name for milvus lite local db


# Define collection name
COLLECTION_NAME = "YOUR_COLLECTION_NAME"  # take your own collection name


# Initialize the ClientMilvus
milvus_client = ClientMilvus(
    milvus_uri=YOUR_MILVUS_LITE_FILE_PATH,
    openai_api_key="<YOUR_OPEN_AI_KEY>",
)


# Create Collection
milvus_client.create_collection(collection_name=COLLECTION_NAME)


# Uploading the PDF document
# get pdfs from data directory in current directory
pdfs = ["data/1.pdf", "data/2.pdf"]  # replace to your pdfs path


milvus_client.upload_documents(documents=pdfs)


# add your rules:
rules = [
    MilvusRule(
        # Replace with your filename
        filename="data/1.pdf",
        page_numbers=[],
    ),
    MilvusRule(
        # Replace with your filename
        filename="data/2.pdf",
        page_numbers=[],
    ),
]


# Search data and implement RAG!
res = milvus_client.query(
    question="YOUR_QUESTIONS",
    rules=rules,
    process_rules_separately=True,
    keyword_trigger=False,
)
print(res["answer"])
print(res["matches"])


# Clean up
milvus_client.drop_collection()
