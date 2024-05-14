# Tutorial of Rule-based Retrieval through Milvus

The `whyhow_rbr` package helps create customized RAG pipelines. It is built on top
of the following technologies (and their respective Python SDKs)

- **OpenAI** - text generation
- **Milvus** - vector database

## Initialization

Please import some essential package
```python
from pymilvus import DataType

from whyhow_rbr import ClientMilvus
```

## Client

The central object is a `ClientMilvus`. It manages all necessary resources
and provides a simple interface for all the RAG related tasks.

First of all, to instantiate it one needs to provide the following
credentials:

- `OPENAI_API_KEY`
- `Milvus_URI`

You need to create a file of format "xxx.db" in your current directory 
and use the file path as milvus_uri.

Initialize the ClientMilvus like this:

```python
# Set up your Milvus Client information
YOUR_MILVUS_LITE_FILE_PATH = "./milvus_demo.db" # random name for milvus lite local db

# Initialize the ClientMilvus
milvus_client = ClientMilvus(
    milvus_uri=YOUR_MILVUS_LITE_FILE_PATH,
)
```

## Vector database operations

This tutorial `whyhow_rbr` uses Milvus for everything related to vector databses.

### Defining necessary variables

```python
# Define collection name
COLLECTION_NAME="YOUR_COLLECTION_NAME" # take your own collection name

# Define vector dimension size
DIMENSION=1536 # decide by the model you use
```

### Add schema

Before inserting any data into Milvus database, we need to first define the data field, which is called schema in here. Through create object `CollectionSchema` and add data field through `addd_field()`, we can control our data type and their characteristics. This step is required.

```python
schema = milvus_client.create_schema(auto_id=True) # Enable id matching

schema = milvus_client.add_field(schema=schema, field_name="id", datatype=DataType.INT64, is_primary=True)
schema = milvus_client.add_field(schema=schema, field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
```
We only defined `id` and `embedding` here because we need to define a primary field for each collection. For embedding, we need to define the dimension. We allow `enable_dynamic_field` which support auto adding schema, but we still encourage you to add schema by yourself. This method is a thin wrapper around the official Milvus implementation ([official docs](https://milvus.io/api-reference/pymilvus/v2.4.x/MilvusClient/Collections/create_schema.md))

### Creating an index

For each schema, it is better to have an index so that the querying will be much more efficient. To create an index, we first need an index_params and later add more index data on this `IndexParams` object.
```python
# Start to indexing data field
index_params = milvus_client.prepare_index_params()
index_params = milvus_client.add_index(
    index_params=index_params,  # pass in index_params object
    field_name="embedding",
    index_type="AUTOINDEX",  # use autoindex instead of other complex indexing method
    metric_type="COSINE",  # L2, COSINE, or IP
)
```
This method  is a thin wrapper around the official Milvus implementation ([official docs](https://milvus.io/api-reference/pymilvus/v2.4.x/MilvusClient/Management/add_index.md)).

### Create Collection

After defining all the data field and indexing them, we now need to create our database collection so that we can access our data quick and precise. What's need to be mentioned is that we initialized the `enable_dynamic_field` to be true so that you can upload any data freely. The cost is the data querying might be inefficient.
```python
# Create Collection
milvus_client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params
)
```

## Uploading documents

After creating a collection, we are ready to populate it with documents. In
`whyhow_rbr` this is done using the `upload_documents` method of the `MilvusClient`.
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
    collection_name=COLLECTION_NAME,
    documents=pdfs
)
```
## Question answering

Now we can finally move to retrieval augmented generation.

In `whyhow_rbr` with Milvus, it can be done via the `search` method.

1. Simple example:

```python
# Search data and implement RAG!
res = milvus_client.search(
    question='What food does Harry Potter like to eat?',
    collection_name=COLLECTION_NAME,
    anns_field='embedding',
    output_fields='text'
)
print(res['answer'])
print(res['matches'])
```

The `result` is a dictionary that has the following keys

- `answer` - the the answer to the question
- `matches` - the `limit` most relevant documents from the index

Note that the number of matches will be in general equal to `limit` which
can be specified as a parameter.

### Clean up

At last, after implemented all the instructuons, you can clean up the database
by calling `drop_collection()`.
```python
# Clean up
milvus_client.drop_collection(
    collection_name=COLLECTION_NAME
)
```

### Rules

In the previous example, every single document in our index was considered.
However, sometimes it might be beneficial to only retrieve documents satisfying some
predefined conditions (e.g. `filename=harry-potter.pdf`). In `whyhow_rbr` through Milvus, this
can be done via adjusting searching parameters.

A rule can control the following metadata attributes

- `filename` - name of the file
- `page_numbers` - list of integers corresponding to page numbers (0 indexing)
- `id` - unique identifier of a chunk (this is the most "extreme" filter)
- Other rules base on [Boolean Expressions](https://milvus.io/docs/boolean.md)

Rules Example:

```python
# RULES(search on book harry-potter on page 8):
PARTITION_NAME='harry-potter' # search on books
page_number='page_number == 8'

# first create a partitions to store the book and later search on this specific partition:
milvus_client.crate_partition(
    collection_name=COLLECTION_NAME,
    partition_name=PARTITION_NAME # separate base on your pdfs type
)

# search with rules
res = milvus_client.search(
    question='Tell me about the greedy method',
    collection_name=COLLECTION_NAME,
    partition_names=PARTITION_NAME,
    filter=page_number, # append any rules follow the Boolean Expression Rule
    anns_field='embedding',
    output_fields='text'
)
print(res['answer'])
print(res['matches'])
```

In this example, we first create a partition that store harry-potter related pdfs, and through searching within this partition, we can get the most direct information. 
Also, we apply page number as a filter to specify the exact page we wish to search on.
Remember, the filer parameter need to follow the [boolean rule](https://milvus.io/docs/boolean.md).

That's all for the Milvus implementation of Rule-based Retrieval.