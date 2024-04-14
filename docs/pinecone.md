# How to do Rule-based Retrieval through Pinecone

Here is the brief introduction of the main functions of Rule-based Retrieval. For more specific tutorial of Pinecone, you can find it [here](tutorial.md).

## Set up the environment

```python
export OPENAI_API_KEY=<your open ai api key>
export PINECONE_API_KEY=<your pinecone api key>
```

## Create index & upload

```shell
from whyhow_rbr import Client

# Configure parameters
index_name = "whyhow-demo"
namespace = "demo"
pdfs = ["harry_potter_book_1.pdf"]

# Initialize client
client = Client()

# Create index
index = client.get_index(index_name)

# Upload, split, chunk, and vectorize documents in Pinecone
client.upload_documents(index=index, documents=pdfs, namespace=namespace)
```

## Query with rules

```shell
from whyhow_rbr import Client, Rule

# Configure query parameters
index_name = "whyhow-demo"
namespace = "demo"
question = "What does Harry wear?"
top_k = 5

# Initialize client
client = Client()

# Create rules
rules = [
    Rule(
        filename="harry_potter_book_1.pdf",
        page_numbers=[21, 22, 23]
    ),
    Rule(
        filename="harry_potter_book_1.pdf",
        page_numbers=[151, 152, 153, 154]
    )
]

# Run query
result = client.query(
    question=question,
    index=index,
    namespace=namespace,
    rules=rules,
    top_k=top_k,
)

answer = result["answer"]
used_contexts = [
    result["matches"][i]["metadata"]["text"] for i in result["used_contexts"]
]
print(f"Answer: {answer}")
print(
    f"The model used {len(used_contexts)} chunk(s) from the DB to answer the question"
)
```

## Query with keywords

```shell
from whyhow_rbr import Client, Rule

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

## Query each rule separately

```shell
from whyhow_rbr import Client, Rule

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