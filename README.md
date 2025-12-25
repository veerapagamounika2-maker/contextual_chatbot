# Contextual Customer Support Chatbot (RAG)

This project implements a contextual customer support chatbot using Retrieval-Augmented Generation (RAG).
The chatbot answers user queries accurately from company documents while maintaining conversation context.

## Features

- Document-based question answering
- Context-aware multi-turn conversations
- Vector-based semantic search
- Streamlit-based chat interface

## Tech Stack

- Python
- LangChain
- OpenAI / LLM
- FAISS (Vector Database)
- Streamlit

## Project Workflow

1. Collect customer support documents
2. Load and preprocess data
3. Split documents into chunks
4. Generate embeddings and store in vector database
5. Retrieve relevant context
6. Generate responses using LLM
7. Display responses via UI

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
