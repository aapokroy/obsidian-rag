# Obsidian RAG Chat

A local RAG chat for an Obsidian knowledge base with hybrid search, reranking, and streaming answers.

> Note: this is a vibe-coded project. Most of the code was built through AI-assisted iteration.

## Features

- **Chat over an Obsidian vault**: ask questions and get answers grounded in your notes
- **Hybrid search**: BM25 + vector search + Reciprocal Rank Fusion
- **Reranking**: separate Nemotron 1B microservice with MPS acceleration
- **Streaming answers**: tokens are streamed to the UI as they are generated
- **Chat history**: persisted in SQLite and available after restarts
- **Incremental indexing**: only changed files are reprocessed
- **Markdown support**: formatted answers with code blocks and tables
- **Local-first setup**: LLM, embeddings, and reranker can all run locally

## Architecture

```text
┌─────────────────────────────────────────────┐
│  Mac host                                   │
│                                             │
│  ┌──────────────┐  ┌─────────────────────┐  │
│  │  LM Studio   │  │  rerank-server      │  │
│  │  LLM + Emb.  │  │  Nemotron + MPS     │  │
│  │  :1234       │  │  :8001              │  │
│  └──────┬───────┘  └──────────┬──────────┘  │
│         │                     │             │
│  ┌──────┴─────────────────────┴──────────┐  │
│  │         Docker                         │  │
│  │  ┌──────────────────────────────────┐  │  │
│  │  │  FastAPI server                  │  │  │
│  │  │  SQLite + sqlite-vec + FTS5      │  │  │
│  │  │  :8000                           │  │  │
│  │  └──────────────────────────────────┘  │  │
│  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Stack

| Component | Technology | Model |
|-----------|------------|-------|
| LLM | LM Studio OpenAI-compatible API | `t-lite-it-2.1` |
| Embeddings | LM Studio API | `text-embedding-user-bge-m3` |
| Reranker | FastAPI + Transformers + MPS | `nvidia/Llama-Nemotron-Rerank-1B-v2` |
| Storage | SQLite + sqlite-vec + FTS5 | - |
| Hybrid search | BM25 + vector search + RRF | - |
| Chunking | LangChain MarkdownTextSplitter | - |
| Server | FastAPI + uvicorn | - |
| Dependencies | Poetry | - |

## Quick Start

### 1. Install LM Studio

Download it from [lmstudio.ai](https://lmstudio.ai) and install it normally.

### 2. Download models in LM Studio

- **LLM**: find `t-lite-it-2.1` and choose a suitable quantization such as `Q5_K_M`
- **Embeddings**: find `text-embedding-user-bge-m3`

### 3. Start LM Studio servers

In the **Developer** tab:

- Select the LLM model, set GPU offload/context length as needed, and start the server
- The embeddings model will be loaded automatically when the app requests embeddings

### 4. Start the rerank server

```bash
cd rerank-server
source venv/bin/activate
pip install -r requirements.txt
python rerank_server.py
```

### 5. Configure `.env`

```bash
echo "VAULT_PATH=/Users/you/ObsidianVault" > .env
```

### 6. Start the RAG server

```bash
docker compose up -d
```

### 7. Open the UI

```text
http://localhost:8000
```

## Configuration

The main configuration lives in `config.yaml` and is grouped by section:

- `paths`: vault path, data directory, and SQLite database path
- `urls`: LM Studio and rerank-server endpoints
- `models`: LLM, embedding, and reranker model names
- `indexing`: chunk size, overlap, and batch size
- `search`: top-k values, relevance threshold, BM25 weight, and RRF constant
- `chat` and `llm`: history limit, temperature, and max tokens

## Project Structure

```text
obsidian-rag/
├── server.py                    # Minimal uvicorn entry point
├── lib/
│   ├── app.py                   # FastAPI app, routes, and service wiring
│   ├── config.py                # Pydantic configuration
│   ├── prompting.py             # Prompt and source assembly
│   ├── embedder.py              # LM Studio embeddings client
│   ├── reranker.py              # Rerank server client
│   ├── retriever.py             # Hybrid search
│   ├── indexer.py               # Obsidian vault indexing
│   └── db/                      # SQLite schema and repositories
├── rerank-server/               # Reranker microservice
│   └── rerank_server.py
├── config.yaml
├── chat_ui.html
├── docker-compose.yaml
├── Dockerfile
└── pyproject.toml
```

## Model Options

### LLM

| Model | Size | Speed | Russian |
|-------|------|-------|---------|
| `t-lite-it-2.1` | ~5.9 GB | fast | excellent |
| `qwen2.5:14b` | ~8.9 GB | medium | excellent |
| `gemma4:e4b` | ~5 GB | fast | good |

### Embeddings

| Model | Size | Context |
|-------|------|---------|
| `text-embedding-user-bge-m3` | ~2.2 GB | 8192 tokens |
| `multilingual-e5-large` | ~2.1 GB | 514 tokens |
| `enbeddrus` | ~0.4 GB | 512 tokens |

### Reranker

| Model | Size | MPS speed |
|-------|------|-----------|
| `nvidia/Llama-Nemotron-Rerank-1B-v2` | ~2.2 GB | ~0.1 sec/chunk |
| `BAAI/bge-reranker-v2-m3` | ~1.2 GB | ~0.05 sec/chunk |

## Development

```bash
# Install dependencies
poetry install

# Regenerate the lock file
poetry lock

# Build Docker image
docker compose build

# Start services
docker compose up -d
```
