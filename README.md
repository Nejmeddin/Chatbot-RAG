# Chatbot RAG with PostgreSQL

This repository contains a **Retrieval-Augmented Generation (RAG)** chatbot built as a student project.  
The goal is to implement the architecture given in the assignment while adapting it to **real constraints**:

- Use a **free, local Hugging Face sentence-transformer** instead of paid embedding APIs.
- Store embeddings in **PostgreSQL arrays (`DOUBLE PRECISION[]`)** instead of `pgvector`.
- Support **both PDF and TXT files** in the ingestion pipeline.
- Keep everything runnable on a **single laptop**.

---

## 🔍 Overview

RAG combines:

1. **Retrieval** – find the most relevant pieces of information in a local knowledge base.
2. **Generation** – use a Large Language Model (LLM) to answer the user’s question using these retrieved documents.

In this project:

- Documents are stored on disk (`data/` folder).
- Embeddings are computed with a Hugging Face model (via `sentence-transformers`).
- Chunks + embeddings + metadata are stored in a PostgreSQL database.
- At query time, the chatbot:
  - embeds the user question,
  - finds similar chunks in PostgreSQL,
  - sends the question + retrieved context to an LLM (e.g. Groq / Llama model),
  - returns a grounded answer.

---

## 🗂 Project Structure

> Exact file names may evolve, but the project is organized roughly like this:

```text
.
├── data/                     
│   ├── ...                   
│
├── notebook/
│   └── prototypage.ipynb    
│
├── src/                     
│   ├── .env        
│
├── requirements.txt
├── .venv 
├── .gitignore                      
└── README.md

**Nejmeddine Ben Maatoug -ENIS**