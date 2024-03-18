# Welcome to Rule-based Retrieval Documentation

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)

The Rule-based Retrieval package is a Python package for creating Retrieval Augmented Generation (RAG) applications with filtering capabilities. It leverages OpenAI for text generation and Pinecone for vector database management.

## Key Features

- Easy-to-use API for creating and managing Pinecone indexes
- Uploading and processing documents (currently supports PDF files)
- Generating embeddings using OpenAI models
- Querying the index with custom filtering rules
- Retrieval Augmented Generation for question answering
- Querying the index with custom filtering rules, including processing rules separately and triggering rules based on keywords

## Getting Started

1. Install the package by following the [Installation Guide](installation.md)
2. Set up your OpenAI and Pinecone API keys as environment variables
3. Create an index and upload your documents using the `Client` class
4. Query the index with custom rules to retrieve relevant documents, optionally processing rules separately or triggering rules based on keywords
5. Use the retrieved documents to generate answers to your questions

For a detailed walkthrough and code examples, check out the [Tutorial](tutorial.md).

## Architecture Overview

The Rule-based Retrieval package consists of the following main components:

- `Client`: The central class for managing resources and performing RAG-related tasks
- `Rule`: Allows defining custom filtering rules for retrieving documents
- `PineconeMetadata` and `PineconeDocument`: Classes for representing and storing document metadata and embeddings in Pinecone
- `embedding`, `processing`, and `exceptions` modules: Utility functions and custom exceptions