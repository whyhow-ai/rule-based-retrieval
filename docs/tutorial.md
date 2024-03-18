# Tutorial

The `whyhow` package helps create customized RAG pipelines. It is built on top
of the following technologies (and their respective Python SDKs)

- **OpenAI** - text generation
- **Pinecone** - vector database

## Client

The central object is a `Client`. It manages all necessary resources
and provides a simple interface for all the RAG related tasks.

First of all, to instantiate it one needs to provide the following
API keys:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`

One can either define the corresponding environment variables

```shell
export OPENAI_API_KEY=...
export PINECONE_API_KEY...
```

and then instantiate the client without any arguments.

```python title="getting_started.py"
from whyhow import Client

client = Client()

```

```shell
python getting_started.py
```

An alternative approach is to manually pass the keys when the client is
being constructed

```python title="getting_started.py"
from whyhow import Client

client = Client(
    openai_api_key="...",
    pinecone_api_key="..."

)
```

```shell
python getting_started.py
```

## Vector database operations

`whywow` uses Pinecone for everything related to vector databses.

### Creating an index

If you don't have a Pinecone index yet, you can create it using the
`create_index` method of the `Client`. This method
is a thin wrapper around the Pinecone SDK ([official docs](https://docs.pinecone.io/docs/create-an-index)).

First of all, you need to provide a specification. There are 2 types

- **Serverless**
- **Pod-based**

#### Serverless

To create a serverless index you can use

```python
# Code above omitted 👆

from pinecone import ServerlessSpec

spec = ServerlessSpec(
    cloud="aws",
    region="us-west-2"
)

index = client.create_index(
    name="great-index",  # the only required argument
    dimension=1536
    metric="cosine",
    spec=spec
)
```

??? note "Full code"

    ```python
    from pinecone import ServerlessSpec

    from whyhow import Client

    client = Client()

    spec = ServerlessSpec(
        cloud="aws",
        region="us-west-2"
    )

    index = client.create_index(
        name="great-index",  # the only required argument
        dimension=1536
        metric="cosine",
        spec=spec
    )

    ```

#### Pod-based

To create a pod-based index you can use

```python
# Code above omitted 👆

from pinecone import PodSpec

spec = PodSpec(
    environment="gcp-starter"
)

index = client.create_index(
    name="amazing-index",  # the only required argument
    dimension=1536
    metric="cosine",
    spec=spec
)
```

??? note "Full code"

    ```python
    from pinecone import PodSpec

    from whyhow import Client

    client = Client()

    spec = PodSpec(
        environment="gcp-starter"
    )

    index = client.create_index(
        name="amazing-index",  # the only required argument
        dimension=1536
        metric="cosine",
        spec=spec
    )

    ```

!!! info
For detailed information on what all of the parameters mean
please refer to [(Pinecone) Understanding indices](https://docs.pinecone.io/docs/indexes)

### Getting an existing index

If your exists already, you can use the `get_index` method to get it.

```python
# Code above omitted 👆

index = client.get_index("amazing-index")

```

??? note "Full code"

    ```python
    from pinecone import PodSpec

    from whyhow import Client

    client = Client()

    index = client.get_index("amazing-index")

    ```

### Index operations

Both `create_index` and `get_index` return an instance of `pinecone.Index`.
It offers multiple convenience methods. See below a few examples.

#### `describe_index_stats`

Shows useful information about the index.

```python
index.describe_index_stats()
```

Example output:

```python
{'dimension': 1536,
 'index_fullness': 0.00448,
 'namespaces': {'A': {'vector_count': 11},
                'B': {'vector_count': 11},
                'C': {'vector_count': 62},
                'D': {'vector_count': 82},
                'E': {'vector_count': 282}},
 'total_vector_count': 448}

```

#### `fetch`

[Fetch (Pinecone docs)](https://docs.pinecone.io/docs/fetch-data)

#### `upsert`

[Upsert (Pinecone docs)](https://docs.pinecone.io/docs/upsert-data)

#### `query`

[Query (Pinecone docs)](https://docs.pinecone.io/docs/query-data)

#### `delete`

[Delete (Pinecone docs)](https://docs.pinecone.io/docs/delete-data)

#### `update`

[Update (Pinecone docs)](https://docs.pinecone.io/docs/update-data)

## Uploading documents

After creating an index, we are ready to populate it with documents. In
`whyhow` this is done using the `upload_documents` method of the `Client`.
It performs the following steps under the hood:

- **Preprocessing**: Reading and splitting the provided PDF files into chunks
- **Embedding**: Embedding all the chunks using an OpenAI model
- **Upserting**: Uploading both the embeddings and the metadata to a Pinecone index

See below an example of how to use it.

```python
# Code above omitted 👆

namespace = "books"
pdfs = ["harry-potter.pdf", "game-of-thrones.pdf"]

client.upload_documents(
    index=index,
    documents=pdfs,
    namespace=namespace
)

```

??? note "Full code"

    ```python
    from whyhow import Client

    client = Client()

    index = client.get_index("amazing-index")

    namespace = "books"
    pdfs = ["harry-potter.pdf", "game-of-thrones.pdf"]

    client.upload_documents(
        index=index,
        documents=pdfs,
        namespace=namespace
    )

    ```

!!! warning

    The above example assumes you have two PDFs on your disk.

    * `harry-potter.pdf`
    * `game-of-thrones.pdf`

    However, feel free to provide different documents.

!!! info

    The `upload_documents` method does not return anything. If you want to
    get some information about what is going on you can activate logging.


    ```python
    import logging

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    ```
    Note that the above affects the root logger, however, you can also
    just customize the `whyhow` logger.

Navigate to [upload_documents (API docs)](./api.md#whyhow.rag.Client.upload_documents)
if you want to get more information on the parameters.

### Index schema

While Pinecone does not require each document in an index to have the same schema
all the document uploaded via the `upload_documents` will have a fixed schema.
This schema is defined in [PineconeDocument (API docs)](./api.md#whyhow.rag.PineconeDocument).
This is done in order to have a predictable set of attributes that
can be used to perform advanced filtering (via rules).

## Question answering

In previous sections we discussed how to to create an index and
populate it with documents. Now we can finally move to retrieval augmented generation.

In `whyhow`, it can be done via the `query` method.

1. Simple example:

```python

from whyhow import Client, Rule

client = Client()

index = client.get_index("amazing-index")
namespace = "books"

question = "What is Harry Potter's favorite food?"

rule = Rule(
    filename="harry-potter.pdf",
    page_numbers=[120, 121, 150]
)

result = client.query(
    question=question,
    index=index,
    namespace=namespace,
    rules=[rule]
)

print(result["answer"])
print(result["matches"])
print(result["used_contexts"])

```

The `result` is a dictionary that has the following three keys

- `answer` - the the answer to the question
- `matches` - the `top_k` most relevant documents from the index
- `used_contexts` - the matches (or more precisely just the texts/contexts) that
  the LLM used to answer the question.

```python
print(result["answer"])
```

```python
'Treacle tart'
```

```python
print(result["matches"])
```

```python
[{'id': 'harry-potter.pdf-120-5',
  'metadata': {'chunk_number': 5,
               'filename': 'harry-potter.pdf',
               'page_number': 120,
               'text': 'Harry loves the treacle tart.'
               'uuid': '86314e32-7d88-475c-b950-f8c156ebf259'},
  'score': 0.826438308},
 {'id': 'game-of-thrones.pdf-75-1',
  'metadata': {'chunk_number': 1,
               'filename': 'game-of-thrones.pdf',
               'page_number': 75,
               'text': 'Harry Strickland was the head of the exiled House Strickland.'
                       'He enjoys eating roasted beef.'
               'uuid': '684a978b-e6e7-45e2-8ba4-5c5019c7c676'},
  'score': 0.2052352},
  ...
  ]
```

Note that the number of matches will be in general equal to `top_k` which
can be specified as a parameter. Also, each match has a fixed schema -
it is a dump of [PineconeMatch (API docs)](./api.md#whyhow.rag.PineconeMatch).

```python
print(result["used_contexts"])
```

```python
[0]
```

The OpenAI model only used the context from the 1st match when answering the question.

??? note "Full code"

    ```python
    from whyhow import Client

    client = Client()

    index = client.get_index("amazing-index")

    namespace = "books"

    question = "What is Harry Potter's favourite food?"

    result = client.query(
        question=question
        index=index,
        namespace=namespace
    )

    print(result["answer"])
    print(result["matches"])
    print(result["used_contexts"])

    ```

Navigate to [query(API docs)](./api.md#whyhow.rag.Client.query)
if you want to get more information on the parameters.

### Rules

In the previous example, every single document in our index was considered.
However, sometimes it might be beneficial to only retrieve documents satisfying some
predefined conditions (e.g. `filename=harry-potter.pdf`). In `whyhow` this
can be done via the `Rule` class.

A rule can control the following metadata attributes

- `filename` - name of the file
- `page_numbers` - list of integers corresponding to page numbers (0 indexing)
- `uuid` - unique identifier of a chunk (this is the most "extreme" filter)
- `keywords` - list of keywords to trigger the rule

2. Keyword example:

```python
# Code above omitted 👆

from whyhow import Rule

question = "What is Harry Potter's favourite food?"

rule = Rule(
    filename="harry-potter.pdf",
    page_numbers=[120, 121, 150],
    keywords=["food", "favorite", "likes to eat"]
)
result = client.query(
    question=question
    index=index,
    namespace=namespace,
    rules=[rule],
    keyword_trigger=True
)

```

In this example, the keyword_trigger parameter is set to True, and the rule includes keywords. Only the rules whose keywords match the words in the question will be applied.

??? note "Full code"

    ```python
    from whyhow import Client, Rule

    client = Client()

    index = client.get_index("amazing-index")
    namespace = "books"

    question = "What does Harry Potter like to eat?"

    rule = Rule(
        filename="harry-potter.pdf",
        keywords=["food", "favorite", "likes to eat"]
    )

    result = client.query(
        question=question,
        index=index,
        namespace=namespace,
        rules=[rule],
        keyword_trigger=True
    )

    print(result["answer"])
    print(result["matches"])
    print(result["used_contexts"])
    ```

3. Process rules separately example:

Lastly, you can specify multiple rules at the same time. They
will be evaluated using the `OR` logical operator.

```python
# Code above omitted 👆

from whyhow import Rule

question = "What is Harry Potter's favorite food?"

rule_1 = Rule(
    filename="harry-potter.pdf",
    page_numbers=[120, 121, 150]
)

rule_2 = Rule(
    filename="harry-potter-volume-2.pdf",
    page_numbers=[80, 81, 82]
)

result = client.query(
    question=question,
    index=index,
    namespace=namespace,
    rules=[rule_1, rule_2],
    process_rules_separately=True
)

```

In this example, the process_rules_separately parameter is set to True. This means that each rule (rule_1 and rule_2) will be processed independently, ensuring that both rules contribute to the final result set.

By default, all rules are run as one joined query, which means that one rule can dominate the others, and given the limit by top_k, a lower priority rule might not return any results. However, by setting process_rules_separately to True, each rule will be processed independently, ensuring that every rule returns results, and the results will be combined at the end.

Depending on the number of rules you use in your query, you may return more chunks than your LLM’s context window can handle. Be mindful of your model’s token limits and adjust your top_k and rule count accordingly.

??? note "Full code"

    ```python
    from whyhow import Client, Rule

    client = Client()

    index = client.get_index("amazing-index")
    namespace = "books"

    question = "What is Harry Potter's favorite food?"

    rule_1 = Rule(
        filename="harry-potter.pdf",
        page_numbers=[120, 121, 150]
    )

    rule_2 = Rule(
        filename="harry-potter-volume-2.pdf",
        page_numbers=[80, 81, 82]
    )

    result = client.query(
        question=question,
        index=index,
        namespace=namespace,
        rules=[rule_1, rule_2],
        process_rules_separately=True
    )

    print(result["answer"])
    print(result["matches"])
    print(result["used_contexts"])
    ```

Navigate to [Rule (API docs)](./api.md#whyhow.rag.Rule)
if you want to get more information on the parameters.
