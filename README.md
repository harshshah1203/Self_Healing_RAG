# Self-Healing RAG System

A local Retrieval-Augmented Generation (RAG) project that stores text documents in ChromaDB, retrieves relevant chunks for a user question, generates an answer with Groq, grades whether the answer is grounded in the retrieved context, and retries with a rewritten query when needed.

## Features

- Ingests `.txt` files from the `docs/` folder.
- Splits source documents into overlapping chunks.
- Persists embeddings and document chunks in a local ChromaDB database.
- Retrieves the top matching chunks for a question.
- Uses `llama-3.3-70b-versatile` through Groq for generation, grading, and query rewriting.
- Uses LangGraph to coordinate the retrieve, generate, grade, and retry workflow.

## Project Structure

```text
.
|-- docs/                 # Source .txt documents for ingestion
|-- chroma_db/            # Local persistent ChromaDB database
|-- ingest.py             # Loads docs and stores chunks in ChromaDB
|-- rag_agent.py          # Main self-healing RAG agent
|-- rag2.py               # Earlier/experimental RAG agent draft
|-- pyproject.toml        # Project metadata and dependencies
|-- uv.lock               # Locked dependency versions
`-- README.md
```

## Requirements

- Python 3.10 or newer
- A Groq API key
- `uv` for dependency management, or another Python environment manager

## Setup

1. Create and activate a virtual environment.

   ```powershell
   uv venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies.

   ```powershell
   uv sync
   ```

3. Create a `.env` file in the project root.

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Add Documents

Put source files in the `docs/` directory as plain text files:

```text
docs/
|-- python_basics.txt
`-- ml.txt
```

Only `.txt` files are loaded by the current ingestion script.

## Ingest Documents

Run the ingestion script to load and split files from `docs/`, then store the chunks in ChromaDB:

```powershell
uv run python ingest.py
```

This creates or updates the local `chroma_db/` directory and the `knowledge_base` collection.

## Ask Questions

Run the main agent:

```powershell
uv run python rag_agent.py
```

Then enter a question when prompted:

```text
Ask a question: What is overfitting?
```

The agent will:

1. Retrieve relevant chunks from ChromaDB.
2. Generate an answer using only the retrieved context.
3. Grade whether the answer is grounded.
4. Rewrite the question and retry if the answer is not grounded.

## How It Works

The workflow in `rag_agent.py` is built with LangGraph:

```text
retrieve -> generate -> grade_answer
                       |-- grounded -> end
                       `-- not_grounded -> rewrite_question -> retrieve
```

The retry loop is limited by `MAX_RETRIES`, currently set to `2`.

## Troubleshooting

- `Collection knowledge_base does not exist`: run `uv run python ingest.py` before running the agent.
- `GROQ_API_KEY` errors: confirm that `.env` exists and contains a valid `GROQ_API_KEY`.
- No useful answer: add more relevant `.txt` files to `docs/`, rerun ingestion, and ask again.
- Duplicate chunks after repeated ingestion: clear or recreate `chroma_db/` before ingesting the same documents again.

## Notes

- `rag_agent.py` is the recommended entry point.
- `rag2.py` appears to be an earlier draft and is not required for normal usage.
- The current vector store is local, so deleting `chroma_db/` removes the ingested knowledge base.
