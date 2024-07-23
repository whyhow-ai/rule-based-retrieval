# Tutorial of Rule-based Retrieval through Milvus

The `rule-based-retrieval` package helps create customized RAG pipelines. It is built on top
of the following technologies (and their respective Python SDKs)

- **Milvus** - vector database
- **OpenAI** - text generation

## Initialization

Install package
```shell
pip install rule-based-retrieval
```

Please import some essential package
```python
from whyhow_rbr import ClientMilvus, MilvusRule
```

## ClientMilvus

The central object is `ClientMilvus`. It manages all necessary resources
and provides a simple interface for all the RAG related tasks.

First of all, to instantiate it one needs to provide the following
credentials:

- `milvus_uri`
- `milvus_token` (optional)
- `openai_api_key`

You need to create a file with the format "xxx.db" in your current directory 
and use the file path as milvus_uri.

Initialize the ClientMilvus like this:

```python
# Set up your Milvus Client information
YOUR_MILVUS_LITE_FILE_PATH = "./milvus_demo.db" # random name for milvus lite local db
OPENAI_API_KEY="<YOUR_OPEN_AI_KEY>"

# Initialize the ClientMilvus
milvus_client = ClientMilvus(
    milvus_uri=YOUR_MILVUS_LITE_FILE_PATH,
    openai_api_key=OPENAI_API_KEY
)
```

## Vector database operations

This tutorial `whyhow_rbr` uses Milvus for everything related to vector databses.

### Create the collection

```python
# Define collection name
COLLECTION_NAME="YOUR_COLLECTION_NAME" # take your own collection name
# Define vector dimension size
DIMENSION=1536 # decide by the model you use

# Create Collection
milvus_client.create_collection(collection_name=COLLECTION_NAME, dimension=DIMENSION)
```

## Uploading documents

After creating a collection, we are ready to populate it with documents. In
`whyhow_rbr` this is done using the `upload_documents` method of the `ClientMilvus`.
It performs the following steps under the hood:

- **Preprocessing**: Reading and splitting the provided PDF files into chunks
- **Embedding**: Embedding all the chunks using an OpenAI model
- **Inserting**: Uploading both the embeddings and the metadata to a Milvus collection

See below an example of how to use it.

```python
# get pdfs
pdfs = ["harry-potter.pdf", "game-of-thrones.pdf"] # replace to your pdfs path

# Uploading the PDF document
milvus_client.upload_documents(
    documents=pdfs
)
```
## Question answering

Now we can finally move to retrieval augmented generation.

In `whyhow_rbr` with Milvus, it can be done via the `query` method.

1. Simple example without rules:

```python
# Search data and implement RAG!
result = milvus_client.query(
    question="What is Harry Potter's favorite food?",
    process_rules_separately=True,
    keyword_trigger=False,
)
print(result["answer"])
print(result["matches"])
```

The `result` is a dictionary that has the following keys

- `answer` - the the answer to the question
- `matches` - the `limit` most relevant documents from the index

Note that the number of matches will be in general equal to `limit` which
can be specified as a parameter. The default value is 5.

### Clean up

At last, after implemented all the instructuons, you can clean up the database
by calling `drop_collection()`.
```python
# Clean up
milvus_client.drop_collection()
```

### Rules

In the previous example, every single document in our collection was considered.
However, sometimes it might be beneficial to only retrieve documents satisfying some
predefined conditions (e.g. `filename=harry-potter.pdf`). In `whyhow_rbr` through Milvus, this
can be done via adjusting searching parameters.

A rule can control the following metadata attributes

- `filename` - name of the file
- `page_numbers` - list of integers corresponding to page numbers (0 indexing)
- `uuid` - unique identifier of a chunk (this is the most "extreme" filter)
- `keywords` - list of keywords to trigger the rule
- Other rules base on [Boolean Expressions](https://milvus.io/docs/boolean.md)

Rules Example:

```python
# RULES:
rules = [
    MilvusRule(
        # Replace with your rule
        filename="harry-potter.pdf",
        page_numbers=[120, 121, 150],
    ),
    MilvusRule(
        # Replace with your rule
        filename="harry-potter.pdf",
        page_numbers=[120, 121, 150],
        keywords=["food", "favorite", "likes to eat"]
    ),
]

# search with rules
res = milvus_client.query(
    question="What is Harry Potter's favorite food?",
    rules=rules,
    process_rules_separately=True,
    keyword_trigger=False,
)
print(res["answer"])
print(res["matches"])
```

In this example, the process_rules_separately parameter is set to True. This means that each rule will be processed independently, ensuring that both rules contribute to the final result set.

By default, all rules are run as one joined query, which means that one rule can dominate the others, and given the return limit, a lower priority rule might not return any results. However, by setting process_rules_separately to True, each rule will be processed independently, ensuring that every rule returns results, and the results will be combined at the end.

That's all for the Milvus implementation of Rule-based Retrieval.