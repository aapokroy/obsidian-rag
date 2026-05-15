#!/usr/bin/env python3
"""FastAPI сервер с RAG-пайплайном, стримингом, гибридным поиском и инкрементальной индексацией."""

import sys
import yaml
import chromadb
import torch
import json
import time
import uuid
import re
import pickle
import numpy as np
import threading
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import MarkdownTextSplitter
from rank_bm25 import BM25Okapi
from ollama import Client

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*telemetry.*")

# --- Глобальное состояние индексации ---
indexing_state = {
    "running": False,
    "stage": "",
    "progress": 0.0,
    "message": "",
    "total_chunks": 0,
    "total_documents": 0,
    "error": None
}

# --- Файл с метаданными индексации (mtime файлов) ---
INDEX_META_FILE = "index_metadata.json"

print("=" * 50)
print("🚀 Запуск Obsidian RAG сервера...")
print("=" * 50)

# Загрузка конфига
print("📋 Загрузка конфигурации...")
config = yaml.safe_load(open("config.yaml"))
print(f"   vault_path: {config['vault_path']}")
print(f"   db_path: {config['db_path']}")
print(f"   embeddings: {config['embeddings_model']}")
print(f"   reranker: {config['reranker_model']}")
print(f"   llm: {config['llm_model']}")
print(f"   ollama_url: {config['ollama_url']}")

# Определение устройства
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\n💻 Устройство: {device}")

# Подключение к ChromaDB
print("\n💾 Подключение к ChromaDB...")
try:
    chroma_client = chromadb.PersistentClient(path=config["db_path"])
    collection = chroma_client.get_collection("obsidian_vault")
    print(f"   ✅ Найдено чанков: {collection.count()}")
except Exception as e:
    print(f"   ⚠️  Коллекция не найдена: {e}")
    print("   Будет создана при первой индексации")
    collection = None

# Загрузка BM25 индекса
bm25 = None
bm25_chunks = None

print("\n📇 Загрузка BM25 индекса...")
bm25_path = Path(config["db_path"]) / "bm25_index.pkl"
if bm25_path.exists():
    try:
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        bm25 = bm25_data["bm25"]
        bm25_chunks = bm25_data["chunks"]
        print(f"   ✅ BM25 загружен ({len(bm25_chunks)} чанков)")
    except Exception as e:
        print(f"   ⚠️  Ошибка загрузки BM25: {e}")
else:
    print("   ⚠️  BM25 индекс не найден.")

# Загрузка эмбеддера
print("\n🧠 Загрузка модели эмбеддингов...")
print(f"   Модель: {config['embeddings_model']}")
print("   ⏳ Загрузка...")
t0 = time.time()
try:
    embedder = SentenceTransformer(config["embeddings_model"], device=device)
    print(f"   ✅ Модель загружена за {time.time() - t0:.1f} сек")
except Exception as e:
    print(f"   ❌ Ошибка загрузки: {e}")
    sys.exit(1)

# Проверка Ollama
print("\n🦙 Проверка подключения к Ollama...")
try:
    ollama_client = Client(host=config["ollama_url"])
    models_response = ollama_client.list()
    
    model_names = []
    if hasattr(models_response, 'models'):
        for model in models_response.models:
            if hasattr(model, 'model'):
                model_names.append(model.model)
            elif isinstance(model, dict):
                model_names.append(model.get("model", model.get("name", str(model))))
            else:
                model_names.append(str(model))
    
    if model_names:
        print(f"   ✅ Доступные модели: {model_names}")
        target_model = config["llm_model"]
        found = any(target_model in name or name in target_model for name in model_names)
        if found:
            print(f"   ✅ Целевая модель '{target_model}' доступна")
        else:
            print(f"   ⚠️  Модель '{target_model}' не найдена")
    else:
        print(f"   ⚠️  Не удалось извлечь имена моделей")
        
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    print("   Убедитесь, что Ollama запущен на хосте")
    sys.exit(1)

# Кэш для реранкера
reranker = None
reranker_name = None

# Персистентное хранилище чатов
CHATS_DIR = Path("/app/chats")
CHATS_DIR.mkdir(exist_ok=True)

chat_histories: dict[str, list[dict]] = {}

MAX_HISTORY = 40


def load_chat_from_disk(chat_id: str) -> list[dict]:
    chat_file = CHATS_DIR / f"{chat_id}.json"
    if chat_file.exists():
        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


def save_chat_to_disk(chat_id: str, messages: list[dict]):
    chat_file = CHATS_DIR / f"{chat_id}.json"
    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def get_chat_history(chat_id: str) -> list[dict]:
    if chat_id not in chat_histories:
        chat_histories[chat_id] = load_chat_from_disk(chat_id)
    return chat_histories[chat_id]


def add_message_to_chat(chat_id: str, role: str, content: str):
    history = get_chat_history(chat_id)
    history.append({"role": role, "content": content})
    
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    
    chat_histories[chat_id] = history
    save_chat_to_disk(chat_id, history)


def get_all_chats() -> list[dict]:
    chats = []
    for chat_file in sorted(CHATS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        chat_id = chat_file.stem
        history = load_chat_from_disk(chat_id)
        title = "Пустой чат"
        for msg in history:
            if msg["role"] == "user":
                title = msg["content"][:50]
                if len(msg["content"]) > 50:
                    title += "..."
                break
        chats.append({
            "id": chat_id,
            "title": title,
            "message_count": len(history)
        })
    return chats


def delete_chat_from_disk(chat_id: str):
    chat_file = CHATS_DIR / f"{chat_id}.json"
    if chat_file.exists():
        chat_file.unlink()
    if chat_id in chat_histories:
        del chat_histories[chat_id]


def get_reranker():
    global reranker, reranker_name
    model_name = config["reranker_model"]
    
    if reranker is not None and reranker_name == model_name:
        return reranker
    
    print(f"\n🎯 Загрузка BGE реранкера...")
    print(f"   Модель: {model_name}")
    t0 = time.time()
    
    try:
        reranker = CrossEncoder(model_name, device=device, max_length=512)
        print(f"   ✅ Загружен за {time.time() - t0:.1f} сек")
        reranker_name = model_name
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        reranker = None
        reranker_name = None
    
    return reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n⚡️ Предзагрузка реранкера...")
    get_reranker()
    
    print("📂 Загрузка истории чатов...")
    for chat_file in CHATS_DIR.glob("*.json"):
        chat_id = chat_file.stem
        chat_histories[chat_id] = load_chat_from_disk(chat_id)
    print(f"   Загружено чатов: {len(chat_histories)}")
    
    print("✅ Готово!")
    yield

app = FastAPI(title="Obsidian RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n" + "=" * 50)
print("✅ Сервер готов к работе!")
print(f"🌐 Откройте: http://127.0.0.1:8000")
print("=" * 50 + "\n")


# --- Модели данных ---

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


# --- RAG функции ---

def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def retrieve_context_hybrid(question: str, top_k: int = 30) -> list[dict]:
    if collection is None:
        return []
        
    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,
        include=["documents", "metadatas", "distances"]
    )
    
    bm25_scores = []
    if bm25 is not None:
        tokenized_query = tokenize(question)
        bm25_scores = list(bm25.get_scores(tokenized_query))
    
    rrf_k = config.get("rrf_k", 60)
    vector_weight = 1.0 - config.get("bm25_weight", 0.3)
    bm25_weight = config.get("bm25_weight", 0.3)
    
    chunk_scores = {}
    
    for rank, (doc, meta, dist) in enumerate(zip(
        vector_results["documents"][0],
        vector_results["metadatas"][0],
        vector_results["distances"][0]
    )):
        chunk_key = f"{meta.get('source', '')}_{meta.get('chunk_index', 0)}"
        rrf_score = vector_weight / (rrf_k + rank + 1)
        
        if chunk_key not in chunk_scores:
            chunk_scores[chunk_key] = {
                "score": rrf_score,
                "text": doc,
                "source": meta.get("source", "unknown"),
                "title": meta.get("title", "unknown"),
                "distance": dist
            }
        else:
            chunk_scores[chunk_key]["score"] += rrf_score
    
    if bm25 is not None and bm25_scores:
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
        
        for rank, (idx, _) in enumerate(bm25_ranked):
            chunk = bm25_chunks[idx]
            chunk_key = f"{chunk['source']}_{chunk['chunk_index']}"
            rrf_score = bm25_weight / (rrf_k + rank + 1)
            
            if chunk_key in chunk_scores:
                chunk_scores[chunk_key]["score"] += rrf_score
            else:
                chunk_scores[chunk_key] = {
                    "score": rrf_score,
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "title": chunk["title"],
                    "distance": 1.0
                }
    
    sorted_chunks = sorted(chunk_scores.values(), key=lambda x: x["score"], reverse=True)
    
    chunks = []
    for item in sorted_chunks[:top_k]:
        chunks.append({
            "text": item["text"],
            "source": item["source"],
            "title": item["title"],
            "distance": item.get("distance", 1.0),
            "hybrid_score": round(item["score"], 4)
        })
    
    return chunks


def rerank_chunks(question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks or len(chunks) <= top_k:
        chunks.sort(key=lambda x: x["distance"])
        for c in chunks:
            c["relevance"] = round(1.0 - c["distance"], 4)
        return chunks[:top_k]

    r = get_reranker()
    
    if r is None:
        chunks.sort(key=lambda x: x["distance"])
        for c in chunks:
            c["relevance"] = round(1.0 - c["distance"], 4)
        return chunks[:top_k]
    
    pairs = [[question, chunk["text"]] for chunk in chunks]
    scores = r.predict(pairs, batch_size=8, show_progress_bar=False)
    scores = np.array(scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    normalized = exp_scores / np.sum(exp_scores)
    
    scored = list(zip(chunks, normalized))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    result = []
    for chunk, score in scored[:top_k]:
        chunk["relevance"] = round(float(score), 4)
        result.append(chunk)
    
    return result


def build_prompt_with_history(question: str, chunks: list[dict], history: list[dict]):
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Источник {i}: {chunk['title']} ({chunk['source']})]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    system_prompt = (
        "Ты — ассистент, который отвечает строго на основе фрагментов базы знаний. "
        "Не ссылайся на источники в формате 'В источнике [1: ...]', просто пиши текст без ссылок. "
        "Ответ оформляй в формате markdown. "
        "Всегда отвечай на русском языке. "
        "Тебе будет предоставлено несколько источников, информация в них не всегда полностью релевантна запросу, "
        "не включай в ответ лишнюю информацию."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    user_message = (
        f"Вопрос: {question}\n\n"
        f"Фрагменты базы знаний:\n{context}\n\n"
        f"Ответь строго по фрагментам."
    )
    messages.append({"role": "user", "content": user_message})
    
    return messages


# --- Функции индексации ---

def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def load_index_metadata() -> dict:
    """Загрузка сохранённых mtime файлов."""
    meta_path = Path(config["db_path"]) / INDEX_META_FILE
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except:
            pass
    return {}


def save_index_metadata(meta: dict):
    """Сохранение mtime файлов."""
    meta_path = Path(config["db_path"]) / INDEX_META_FILE
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_markdown_files(vault_path: str, old_meta: dict = None, full_rebuild: bool = False) -> tuple[list[dict], dict]:
    """
    Загрузка .md файлов.
    Если full_rebuild=False и old_meta передан — возвращает только изменённые/новые файлы.
    Возвращает (documents, new_meta).
    """
    documents = []
    new_meta = {}
    vault = Path(vault_path)
    
    for md_file in vault.rglob("*.md"):
        if any(part.startswith(".") for part in md_file.parts):
            continue
        
        relative_path = str(md_file.relative_to(vault))
        current_mtime = md_file.stat().st_mtime
        
        new_meta[relative_path] = current_mtime
        
        # Проверяем, нужно ли индексировать
        if not full_rebuild and old_meta is not None:
            old_mtime = old_meta.get(relative_path, 0)
            if current_mtime <= old_mtime:
                continue  # Файл не изменился — пропускаем
        
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                documents.append({
                    "content": content,
                    "source": relative_path,
                    "title": md_file.stem
                })
        except Exception as e:
            print(f"⚠️  Ошибка чтения {md_file}: {e}")
    
    return documents, new_meta


def chunk_documents(documents: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    
    for doc in documents:
        doc_chunks = splitter.split_text(doc["content"])
        for i, chunk in enumerate(doc_chunks):
            chunks.append({
                "text": chunk,
                "source": doc["source"],
                "title": doc["title"],
                "chunk_index": i,
                "tokens": tokenize_for_bm25(chunk)
            })
    
    return chunks


def run_indexer(full_rebuild: bool = False):
    """
    Запуск индексации.
    full_rebuild=False — инкрементальная (только изменённые файлы).
    full_rebuild=True — полная переиндексация.
    """
    global collection, bm25, bm25_chunks, indexing_state
    
    indexing_state["running"] = True
    indexing_state["error"] = None
    
    try:
        vault_path = config["vault_path"]
        db_path = config["db_path"]
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        embeddings_model = config["embeddings_model"]
        
        # Загружаем старые метаданные
        old_meta = load_index_metadata() if not full_rebuild else {}
        
        # Этап 1: Загрузка файлов (5%)
        indexing_state["stage"] = "loading"
        indexing_state["progress"] = 0.05
        mode_text = "полная" if full_rebuild else "инкрементальная"
        indexing_state["message"] = f"Загрузка файлов ({mode_text})..."
        
        print(f"🔍 Загрузка файлов из vault ({mode_text} индексация)...")
        documents, new_meta = load_markdown_files(vault_path, old_meta, full_rebuild)
        indexing_state["total_documents"] = len(documents)
        
        if full_rebuild:
            print(f"   Найдено документов: {len(documents)} (полный пересчёт)")
        else:
            print(f"   Изменённых/новых документов: {len(documents)}")
        
        if not documents:
            indexing_state["stage"] = "done"
            indexing_state["progress"] = 1.0
            indexing_state["message"] = "Нет новых или изменённых документов"
            indexing_state["running"] = False
            print("   Нет файлов для индексации")
            return
        
        # Этап 2: Чанкинг (15%)
        indexing_state["stage"] = "chunking"
        indexing_state["progress"] = 0.15
        indexing_state["message"] = "Разбивка на чанки..."
        
        print("✂️  Разбивка на чанки...")
        new_chunks = chunk_documents(documents, chunk_size, chunk_overlap)
        indexing_state["total_chunks"] = len(new_chunks)
        print(f"   Создано чанков: {len(new_chunks)}")
        
        # Этап 3: BM25 (25%)
        indexing_state["stage"] = "bm25"
        indexing_state["progress"] = 0.25
        indexing_state["message"] = "Создание BM25 индекса..."
        
        print("📇 Обновление BM25 индекса...")
        
        if full_rebuild or bm25 is None:
            # Полный пересчёт BM25
            tokenized_chunks = [c["tokens"] for c in new_chunks]
            bm25_new = BM25Okapi(tokenized_chunks)
            all_chunks = new_chunks
        else:
            # Инкрементальное обновление: удаляем старые чанки этих файлов, добавляем новые
            changed_sources = set(doc["source"] for doc in documents)
            old_chunks = [c for c in (bm25_chunks or []) if c["source"] not in changed_sources]
            all_chunks = old_chunks + new_chunks
            tokenized_chunks = [c["tokens"] for c in all_chunks]
            bm25_new = BM25Okapi(tokenized_chunks)
        
        bm25_path = Path(db_path) / "bm25_index.pkl"
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": bm25_new, "chunks": all_chunks}, f)
        
        # Этап 4: Эмбеддинги (30% - 75%)
        indexing_state["stage"] = "embeddings"
        indexing_state["progress"] = 0.30
        indexing_state["message"] = "Загрузка модели эмбеддингов..."
        
        print(f"🧠 Загрузка модели: {embeddings_model}")
        embed_model = SentenceTransformer(embeddings_model, device=device)
        
        indexing_state["message"] = "Создание эмбеддингов..."
        texts = [c["text"] for c in new_chunks]
        
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_emb = embed_model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            all_embeddings.append(batch_emb)
            
            progress = 0.30 + 0.45 * (i + len(batch_texts)) / len(texts)
            indexing_state["progress"] = min(progress, 0.75)
            indexing_state["message"] = f"Эмбеддинги: {min(i + batch_size, len(texts))}/{len(texts)}"
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Этап 5: Сохранение в ChromaDB (75% - 95%)
        indexing_state["stage"] = "saving"
        indexing_state["progress"] = 0.75
        indexing_state["message"] = "Сохранение в ChromaDB..."
        
        print("💾 Сохранение в ChromaDB...")
        chroma_client_local = chromadb.PersistentClient(path=db_path)
        
        if full_rebuild:
            try:
                chroma_client_local.delete_collection("obsidian_vault")
            except:
                pass
            collection_new = chroma_client_local.create_collection(
                name="obsidian_vault",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Все чанки с эмбеддингами
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                end_idx = min(i + batch_size, len(embeddings))
                batch_emb = embeddings[i:end_idx]
                
                if len(batch_emb) > 0 and len(batch) > 0:
                    # Обрезаем batch до размера batch_emb (если последний батч меньше)
                    batch = batch[:len(batch_emb)]
                    
                    collection_new.add(
                        ids=[f"chunk_{i + j}" for j in range(len(batch))],
                        embeddings=batch_emb.tolist(),
                        documents=[c["text"] for c in batch],
                        metadatas=[{
                            "source": c["source"],
                            "title": c["title"],
                            "chunk_index": c["chunk_index"]
                        } for c in batch]
                    )
                
                progress = 0.75 + 0.20 * min(i + batch_size, len(all_chunks)) / len(all_chunks)
                indexing_state["progress"] = min(progress, 0.95)
                indexing_state["message"] = f"Сохранение: {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}"
        else:
            # Инкрементальное обновление
            collection_new = chroma_client_local.get_or_create_collection(
                name="obsidian_vault",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Удаляем старые чанки изменённых файлов
            changed_sources = set(doc["source"] for doc in documents)
            if collection is not None and changed_sources:
                try:
                    existing = collection.get(include=["metadatas"])
                    ids_to_delete = []
                    for id_, meta in zip(existing["ids"], existing["metadatas"]):
                        if meta.get("source") in changed_sources:
                            ids_to_delete.append(id_)
                    if ids_to_delete:
                        collection.delete(ids=ids_to_delete)
                except:
                    pass
            
            # Добавляем новые чанки
            offset = collection_new.count()
            for i in range(0, len(new_chunks), batch_size):
                batch = new_chunks[i:i + batch_size]
                batch_emb = embeddings[i:i + batch_size]
                
                collection_new.add(
                    ids=[f"chunk_{offset + j}" for j in range(len(batch))],
                    embeddings=batch_emb.tolist(),
                    documents=[c["text"] for c in batch],
                    metadatas=[{
                        "source": c["source"],
                        "title": c["title"],
                        "chunk_index": c["chunk_index"]
                    } for c in batch]
                )
                offset += len(batch)
                
                progress = 0.75 + 0.20 * (i + len(batch)) / len(new_chunks)
                indexing_state["progress"] = min(progress, 0.95)
                indexing_state["message"] = f"Сохранение: {min(i + batch_size, len(new_chunks))}/{len(new_chunks)}"
        
        # Этап 6: Завершение (100%)
        collection = collection_new
        bm25 = bm25_new
        bm25_chunks = all_chunks
        
        # Сохраняем метаданные файлов
        if full_rebuild:
            save_index_metadata(new_meta)
        else:
            merged_meta = {**old_meta, **new_meta}
            save_index_metadata(merged_meta)
        
        indexing_state["stage"] = "done"
        indexing_state["progress"] = 1.0
        indexing_state["message"] = f"Готово: {collection.count()} чанков из {len(all_chunks)} всего"
        
        print(f"✅ Индексация завершена! Чанков в коллекции: {collection.count()}")
        
    except Exception as e:
        indexing_state["error"] = str(e)
        indexing_state["stage"] = "error"
        indexing_state["message"] = f"Ошибка: {str(e)[:100]}"
        print(f"❌ Ошибка индексации: {e}")
    
    finally:
        indexing_state["running"] = False


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
    print(f"🆕 Новый чат: {chat_id}")
    return ChatInfo(id=chat_id, title="Новый чат", message_count=0)


@app.get("/chat/{chat_id}", response_model=ChatHistoryResponse)
async def get_chat(chat_id: str):
    history = get_chat_history(chat_id)
    return ChatHistoryResponse(chat_id=chat_id, messages=history)


@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    delete_chat_from_disk(chat_id)
    print(f"🗑️ Чат удалён: {chat_id}")
    return {"status": "ok"}


@app.post("/stream")
async def stream_query(request: QueryRequest):
    question = request.question.strip()
    chat_id = request.chat_id
    
    if not question:
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    print(f"\n📝 Вопрос (stream) [{chat_id}]: {question}")

    history = get_chat_history(chat_id)

    t0 = time.time()
    print("🔍 Поиск релевантных чанков...")
    
    search_query = question
    if history:
        for msg in reversed(history):
            if msg["role"] == "assistant":
                search_query = f"{msg['content'][:500]}\n{question}"
                break
    
    chunks = retrieve_context_hybrid(search_query, config["top_k_retrieval"])
    print(f"   Найдено: {len(chunks)} за {time.time() - t0:.1f} сек")

    if not chunks:
        async def empty_response():
            yield f"data: {json.dumps({'type': 'error', 'content': 'В базе знаний нет информации по этому вопросу.'})}\n\n"
        return StreamingResponse(empty_response(), media_type="text/event-stream")

    t0 = time.time()
    print("🎯 Реранкинг...")
    top_chunks = rerank_chunks(question, chunks, config["top_k_final"])
    print(f"   Отобрано лучших: {len(top_chunks)} за {time.time() - t0:.1f} сек")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"   {i}. {chunk['title']} (r={chunk['relevance']})")

    min_relevance = config.get("min_relevance", 0.0)
    sources = []
    for chunk in top_chunks:
        if chunk["relevance"] >= min_relevance:
            sources.append(Source(
                source=chunk["source"],
                title=chunk["title"],
                snippet=chunk["text"][:200] + "...",
                relevance=chunk["relevance"]
            ))

    print("🤖 Генерация ответа (stream)...")
    messages = build_prompt_with_history(question, top_chunks, history)

    add_message_to_chat(chat_id, "user", question)

    return StreamingResponse(
        generate_stream_with_history(messages, config["llm_model"], sources, chat_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


def generate_stream_with_history(messages: list[dict], model: str, sources: list[Source], chat_id: str):
    full_response = ""
    
    try:
        stream = ollama_client.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.1},
            stream=True
        )
        
        for chunk in stream:
            if chunk["message"]["content"]:
                token = chunk["message"]["content"]
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        
        add_message_to_chat(chat_id, "assistant", full_response)
        
        sources_data = []
        for s in sources:
            sources_data.append({
                "source": s.source,
                "title": s.title,
                "snippet": s.snippet,
                "relevance": s.relevance
            })
        
        if sources_data:
            yield f"data: {json.dumps({'type': 'sources', 'content': sources_data})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


# --- Эндпоинты переиндексации ---

@app.post("/reindex")
async def reindex():
    """Запуск инкрементальной индексации (только изменённые файлы)."""
    return _start_reindex(full_rebuild=False)


@app.post("/reindex/full")
async def reindex_full():
    """Запуск полной переиндексации (все файлы)."""
    return _start_reindex(full_rebuild=True)


def _start_reindex(full_rebuild: bool):
    if indexing_state["running"]:
        return {"status": "already_running", "message": "Индексация уже выполняется"}
    
    thread = threading.Thread(target=run_indexer, args=(full_rebuild,))
    thread.start()
    
    mode = "полная" if full_rebuild else "инкрементальная"
    return {"status": "started", "message": f"Запущена {mode} индексация"}


@app.get("/reindex/status")
async def reindex_status():
    try:
        count = collection.count() if collection else 0
    except:
        count = 0
    return {
        "running": indexing_state["running"],
        "chunks_count": count,
        "stage": indexing_state["stage"],
        "progress": indexing_state["progress"]
    }


@app.get("/reindex/stream")
async def reindex_stream():
    """SSE-поток с прогрессом индексации."""
    
    async def event_stream():
        last_progress = -1
        while True:
            current = dict(indexing_state)
            
            if current["progress"] != last_progress or current["stage"] != getattr(event_stream, "last_stage", ""):
                yield f"data: {json.dumps(current)}\n\n"
                last_progress = current["progress"]
                event_stream.last_stage = current["stage"]
            
            if not current["running"]:
                yield f"data: {json.dumps(current)}\n\n"
                break
            
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)