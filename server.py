#!/usr/bin/env python3
"""FastAPI сервер Obsidian RAG."""

import sys
import json
import time
import uuid
import threading
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

import chromadb
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from lib.config import Config
from lib.chat_store import (
    get_chat_history, add_message_to_chat,
    get_all_chats, delete_chat_from_disk,
    save_chat_to_disk, load_chat_from_disk
)
from lib.retriever import retrieve_context_hybrid
from lib.reranker import rerank_chunks
from lib.prompt import build_prompt_with_history
from lib.indexer import run_indexer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*telemetry.*")

# Конфиг
config = Config.from_yaml()

# Состояние индексации
indexing_state = {
    "running": False,
    "stage": "",
    "progress": 0.0,
    "message": "",
    "total_chunks": 0,
    "total_documents": 0,
    "error": None,
    "full_rebuild": False
}

print("=" * 50)
print("🚀 Запуск Obsidian RAG сервера...")
print("=" * 50)
print(f"📋 Конфиг загружен: vault={config.vault_path}, llm={config.llm_model}")

# ChromaDB
print("\n💾 Подключение к ChromaDB...")
try:
    chroma_client = chromadb.PersistentClient(path=config.db_path)
    collection = chroma_client.get_collection("obsidian_vault")
    print(f"   ✅ Найдено чанков: {collection.count()}")
except:
    print("   ⚠️  Коллекция не найдена")
    collection = None

# BM25
bm25 = None
bm25_chunks = None
bm25_path = Path(config.db_path) / "bm25_index.pkl"
if bm25_path.exists():
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    bm25_chunks = data["chunks"]
    print(f"📇 BM25 загружен ({len(bm25_chunks)} чанков)")

# LLM клиент
print("🦙 Подключение к LLM API...")
llm_client = OpenAI(base_url=config.llm_url, api_key="lm-studio")
models = llm_client.models.list()
print(f"   Доступные модели: {[m.id for m in models.data]}")

# Чат-хранилище
CHATS_DIR = Path("/app/chats")
CHATS_DIR.mkdir(exist_ok=True)
chat_histories = {}
for chat_file in CHATS_DIR.glob("*.json"):
    chat_histories[chat_file.stem] = load_chat_from_disk(chat_file.stem)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ Сервер готов!")
    yield

app = FastAPI(title="Obsidian RAG", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# --- Модели ---

class QueryRequest(BaseModel):
    question: str
    chat_id: str = "default"

class Source(BaseModel):
    source: str
    title: str
    snippet: str
    relevance: float

class ChatInfo(BaseModel):
    id: str
    title: str
    message_count: int

class ChatHistoryResponse(BaseModel):
    chat_id: str
    messages: list[dict]


# --- Эндпоинты ---

@app.get("/")
async def root():
    return FileResponse("chat_ui.html")

@app.get("/chats", response_model=list[ChatInfo])
async def list_chats():
    return get_all_chats()

@app.post("/chat/new", response_model=ChatInfo)
async def create_chat():
    chat_id = str(uuid.uuid4())[:8]
    chat_histories[chat_id] = []
    save_chat_to_disk(chat_id, [])
    return ChatInfo(id=chat_id, title="Новый чат", message_count=0)

@app.get("/chat/{chat_id}", response_model=ChatHistoryResponse)
async def get_chat(chat_id: str):
    return ChatHistoryResponse(chat_id=chat_id, messages=get_chat_history(chat_id))

@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    delete_chat_from_disk(chat_id)
    if chat_id in chat_histories:
        del chat_histories[chat_id]
    return {"status": "ok"}

@app.post("/stream")
async def stream_query(request: QueryRequest):
    question = request.question.strip()
    chat_id = request.chat_id
    
    if not question:
        raise HTTPException(400, "Вопрос не может быть пустым")
    
    history = get_chat_history(chat_id)
    
    search_query = question
    for msg in reversed(history):
        if msg["role"] == "assistant":
            search_query = f"{msg['content'][:500]}\n{question}"
            break
    
    print(f"\n📝 [{chat_id}]: {question}")
    
    chunks = retrieve_context_hybrid(
        search_query, collection, bm25, bm25_chunks,
        config.embeddings_url, config.embeddings_model,
        config.top_k_retrieval, config.bm25_weight, config.rrf_k
    )
    
    if not chunks:
        async def empty():
            yield f"data: {json.dumps({'type': 'error', 'content': 'Нет информации в базе знаний.'})}\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")
    
    top_chunks = rerank_chunks(config.reranker_url, question, chunks, config.top_k_final)
    
    sources = [
        Source(source=c["source"], title=c["title"],
               snippet=c["text"][:200] + "...", relevance=c.get("relevance", 0))
        for c in top_chunks if c.get("relevance", 0) >= config.min_relevance
    ]
    
    messages = build_prompt_with_history(question, top_chunks, history)
    add_message_to_chat(chat_id, "user", question)
    
    return StreamingResponse(
        generate_stream(messages, config.llm_model, sources, chat_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


def generate_stream(messages, model, sources, chat_id):
    full = ""
    try:
        stream = llm_client.chat.completions.create(
            model=model, messages=messages, temperature=0.1, stream=True,
            extra_body={"max_tokens": 4096}
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        
        add_message_to_chat(chat_id, "assistant", full)
        
        if sources:
            src_data = [{"source": s.source, "title": s.title, "snippet": s.snippet, "relevance": s.relevance} for s in sources]
            yield f"data: {json.dumps({'type': 'sources', 'content': src_data})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


# --- Индексация ---

@app.post("/reindex")
async def reindex():
    return _start_reindex(False)

@app.post("/reindex/full")
async def reindex_full():
    return _start_reindex(True)

def _start_reindex(full: bool):
    global collection, bm25, bm25_chunks
    if indexing_state["running"]:
        return {"status": "already_running"}
    
    indexing_state["full_rebuild"] = full
    indexing_state["running"] = True
    
    def job():
        global collection, bm25, bm25_chunks
        result = run_indexer(config, collection, bm25, bm25_chunks, indexing_state)
        if result and result[0]:
            collection, bm25, bm25_chunks = result
    
    threading.Thread(target=job).start()
    return {"status": "started"}

@app.get("/reindex/status")
async def reindex_status():
    try:
        count = collection.count() if collection else 0
    except:
        count = 0
    return {"running": indexing_state["running"], "chunks_count": count,
            "stage": indexing_state["stage"], "progress": indexing_state["progress"]}

@app.get("/reindex/stream")
async def reindex_stream():
    async def event_stream():
        last_progress = -1
        while True:
            current = dict(indexing_state)
            if current["progress"] != last_progress:
                yield f"data: {json.dumps(current)}\n\n"
                last_progress = current["progress"]
            if not current["running"]:
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)