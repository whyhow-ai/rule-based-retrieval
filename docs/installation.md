# Installation

To install the Rule-based Retrieval package, follow these steps:

## Prerequisites

- Python 3.10 or higher
- OpenAI API key
- Pinecone API key

## Install from GitHub

Clone the repository:

```shell
git clone git@github.com:whyhow-ai/rule-based-retrieval.git
cd rule-based-retrieval
```

Install the packages:

```shell
pip install .
```

Set the required environment variables:

```shell
export OPENAI_API_KEY=<your open ai api key>
export PINECONE_API_KEY=<your pinecone api key>
```

## Developer Installation

For a developer installation, use an editable install and include the development dependencies:

```shell
pip install -e .[dev]
```

For ZSH:

```shell
pip install -e ".[dev]"
```

## Install Documentation Dependencies

To build and serve the documentation locally, install the documentation dependencies:

```shell
pip install -e .[docs]
```

For ZSH:

```shell
pip install -e ".[docs]"
```

Then, use mkdocs to serve the documentation:

```shell
mkdocs serve
```

Navigate to http://127.0.0.1:8000/ in your browser to view the documentation.

## Troubleshooting

If you encounter any issues during installation, please check the following:

- Ensure that you have Python 3.10 or higher installed. You can check your Python version by running `python --version` in your terminal.
- Make sure that you have correctly set the `OPENAI_API_KEY` and `PINECONE_API_KEY` environment variables with your respective API keys.
- If you are installing from GitHub, ensure that you have cloned the repository correctly and are in the right directory.
- If you are using a virtual environment, make sure that it is activated before running the installation commands.
- If you still face problems, please open an issue on the GitHub repository with detailed information about the error and your environment setup.
