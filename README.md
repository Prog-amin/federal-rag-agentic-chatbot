---
title: Federal Rag Agentic Chatbot
emoji: 🐠
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
short_description: RAG Agentic System for US Federal Registry
---

# Federal Registry RAG Agent

An AI-powered Retrieval-Augmented Generation (RAG) system that provides intelligent access to US Federal Registry documents including executive orders, regulations, notices, and other government publications.

## Features

- **Real-time Data Pipeline**: Automatically fetches and processes federal documents from the official Federal Register API
- **Intelligent Search**: AI agent with tool-calling capabilities to query and analyze federal documents
- **Interactive Chat Interface**: Natural language interface to ask questions about federal documents
- **SQLite Database**: Efficient local storage optimized for Hugging Face Spaces

## Quick Start

1. **Initialize System**: Click "Initialize System" to set up the database
2. **Run Data Pipeline**: Click "Run Data Pipeline" to download recent federal documents
3. **Start Chatting**: Ask questions about federal documents, executive orders, regulations, etc.

## Example Queries

- "What are the recent executive orders from the last 7 days?"
- "Find documents about artificial intelligence regulations"
- "Show me documents from the Department of Defense"
- "Search for climate change related documents" 
- "What new regulations were published this month?"

## Technical Architecture

- **Agent System**: OpenAI-compatible LLM with function calling capabilities
- **Data Pipeline**: Asynchronous processing of Federal Register API data
- **Database**: SQLite for document storage and retrieval

## Configuration

The system requires an external LLM API key (OpenAI, Groq, etc.). Set your API credentials in the Hugging Face Spaces secrets:

- `LLM_API_KEY`: Your LLM provider API key
- `LLM_BASE_URL`: API endpoint URL (default: OpenAI)
- `LLM_MODEL`: Model name (default: gpt-3.5-turbo)

## Data Source

Documents are sourced from the official [Federal Register API](https://www.federalregister.gov/developers/documentation/api/v1), ensuring access to authoritative and up-to-date government publications.

## Limitations

- Free tier limitations apply to external LLM API usage
- Data pipeline processes recent documents (last 7 days by default)
- Storage limited to SQLite database on HF Spaces

## Development

This system demonstrates:
- Asynchronous Python programming
- RAG system architecture  
- LLM agent design with tool calling
- API integration and data processing
- Real-time web interfaces

## Example Flow: Behind the Scenes of Federal RAG Agent
Data Ingestion: Integrated the official Federal Register API to stream and parse new
documents automatically.
Vectorization: Each document is chunked and converted into embeddings using an LLM-based
encoder.
Retrieval: Embeddings are stored in a SQLite vector database (now migrating to Pinecone for
scalability). At query time, the system performs a semantic search to fetch the most relevant
passages.
Answer Generation: Retrieved passages are combined with the user’s question and passed to
an LLM (Meta-Llama-3.2-3B-Instruct) with strict grounding instructions to ensure accuracy.


## Live Demo

The Federal RAG Agentic Chatbot is live and accessible on [Hugging Face Spaces](https://huggingface.co/spaces/Prog-amin/federal-rag-agentic-chatbot)

Try it out directly in your browser to interact with federal documents in real time. Perfect for learning modern AI application development patterns and government data integration.
