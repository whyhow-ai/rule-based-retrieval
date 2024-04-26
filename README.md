# Rule-based Retrieval

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/pypi/v/rule-based-retrieval)](https://pypi.org/project/rule-based-retrieval/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)
[![Whyhow Discord](https://dcbadge.vercel.app/api/server/PAgGMxfhKd?compact=true&style=flat)](https://discord.gg/PAgGMxfhKd)

The Rule-based Retrieval package is a Python package that enables you to create and manage Retrieval Augmented Generation (RAG) applications with advanced filtering capabilities. It seamlessly integrates with OpenAI for text generation and Pinecone for efficient vector database management.

# Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- Pinecone or Milvus API key

### Install from PyPI

You can install the package directly from PyPI using pip:

```shell
pip install rule-based-retrieval
```

### Install from GitHub

Alternatively, you can clone the repo and install the package:

```shell
git clone git@github.com:whyhow-ai/rule-based-retrieval.git
cd rule-based-retrieval
pip install .
```

### Developer Install

For a developer installation, use an editable install and include the development dependencies:

```shell
pip install -e .[dev]
```

For ZSH:

```shell
pip install -e ".[dev]"
```

If you want to install the package directly without explicitly cloning yourself
run

```shell
pip install git+ssh://git@github.com/whyhow-ai/rule-based-retrieval
```

# Documentation

Documentation can be found [here](https://whyhow-ai.github.io/rule-based-retrieval/).

To serve the docs locally run

```shell
pip install -e .[docs]
mkdocs serve
```

For ZSH:

```shell
pip install -e ".[docs]"
mkdocs serve
```

Navigate to http://127.0.0.1:8000/ in your browser to view the documentation.

# Examples

Check out the `examples/` directory for sample scripts demonstrating how to use the Rule-based Retrieval package.

# How to

### [Demo](https://www.loom.com/share/089101b455b34701875b9f362ba16b89)
`whyhow_rbr` offers different ways to implement Rule-based Retrieval through two databases and down below are the documentations(tutorial and example) for each implementation:

- [Milvus](docs/milvus.md) 
- [Pinecone](docs/pinecone.md)

# Contributing

We welcome contributions to improve the Rule-based Retrieval package! If you have any ideas, bug reports, or feature requests, please open an issue on the GitHub repository.

If you'd like to contribute code, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with descriptive messages
4. Push your changes to your forked repository
5. Open a pull request to the main repository

### License

This project is licensed under the MIT License.

### Support

WhyHow.AI is building tools to help developers bring more determinism and control to their RAG pipelines using graph structures. If you're thinking about, in the process of, or have already incorporated knowledge graphs in RAG, we’d love to chat at team@whyhow.ai, or follow our newsletter at [WhyHow.AI](https://www.whyhow.ai/). Join our discussions about rules, determinism and knowledge graphs in RAG on our newly-created [Discord](https://discord.com/invite/9bWqrsxgHr).
