"""Индексация vault."""

import json
import pickle
import re
import numpy as np
from pathlib import Path

import chromadb
from langchain_text_splitters import MarkdownTextSplitter
from rank_bm25 import BM25Okapi

from lib.embeddings import get_embeddings

INDEX_META_FILE = "index_metadata.json"


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def load_index_metadata(db_path: str) -> dict:
    meta_path = Path(db_path) / INDEX_META_FILE
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except:
            pass
    return {}


def save_index_metadata(db_path: str, meta: dict):
    meta_path = Path(db_path) / INDEX_META_FILE
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_markdown_files(vault_path: str, old_meta: dict = None, full_rebuild: bool = False) -> tuple[list[dict], dict]:
    documents = []
    new_meta = {}
    vault = Path(vault_path)
    
    for md_file in vault.rglob("*.md"):
        if any(part.startswith(".") for part in md_file.parts):
            continue
        
        relative_path = str(md_file.relative_to(vault))
        current_mtime = md_file.stat().st_mtime
        
        new_meta[relative_path] = current_mtime
        
        if not full_rebuild and old_meta is not None:
            if current_mtime <= old_meta.get(relative_path, 0):
                continue
        
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
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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


def run_indexer(config, collection, bm25, bm25_chunks, indexing_state):
    """Запуск индексации vault."""
    full_rebuild = indexing_state.get("full_rebuild", False)
    
    indexing_state["running"] = True
    indexing_state["error"] = None
    
    try:
        vault_path = config.vault_path
        db_path = config.db_path
        chunk_size = config.chunk_size
        chunk_overlap = config.chunk_overlap
        
        old_meta = load_index_metadata(db_path) if not full_rebuild else {}
        
        indexing_state["stage"] = "loading"
        indexing_state["progress"] = 0.05
        mode_text = "полная" if full_rebuild else "инкрементальная"
        indexing_state["message"] = f"Загрузка файлов ({mode_text})..."
        
        documents, new_meta = load_markdown_files(vault_path, old_meta, full_rebuild)
        indexing_state["total_documents"] = len(documents)
        
        if not documents:
            indexing_state["stage"] = "done"
            indexing_state["progress"] = 1.0
            indexing_state["message"] = "Нет новых или изменённых документов"
            indexing_state["running"] = False
            return None, bm25, bm25_chunks
        
        indexing_state["stage"] = "chunking"
        indexing_state["progress"] = 0.15
        indexing_state["message"] = "Разбивка на чанки..."
        
        new_chunks = chunk_documents(documents, chunk_size, chunk_overlap)
        indexing_state["total_chunks"] = len(new_chunks)
        
        indexing_state["stage"] = "bm25"
        indexing_state["progress"] = 0.25
        indexing_state["message"] = "Создание BM25 индекса..."
        
        if full_rebuild or bm25 is None:
            tokenized_chunks = [c["tokens"] for c in new_chunks]
            bm25_new = BM25Okapi(tokenized_chunks)
            all_chunks = new_chunks
        else:
            changed_sources = set(doc["source"] for doc in documents)
            old_chunks = [c for c in (bm25_chunks or []) if c["source"] not in changed_sources]
            all_chunks = old_chunks + new_chunks
            tokenized_chunks = [c["tokens"] for c in all_chunks]
            bm25_new = BM25Okapi(tokenized_chunks)
        
        bm25_path = Path(db_path) / "bm25_index.pkl"
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": bm25_new, "chunks": all_chunks}, f)
        
        indexing_state["stage"] = "embeddings"
        indexing_state["progress"] = 0.30
        indexing_state["message"] = "Создание эмбеддингов..."
        
        texts = [c["text"] for c in new_chunks]
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_emb = get_embeddings(config.embeddings_url, config.embeddings_model, batch_texts)
            all_embeddings.append(batch_emb)
            
            progress = 0.30 + 0.45 * (i + len(batch_texts)) / len(texts)
            indexing_state["progress"] = min(progress, 0.75)
            indexing_state["message"] = f"Эмбеддинги: {min(i + batch_size, len(texts))}/{len(texts)}"
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        indexing_state["stage"] = "saving"
        indexing_state["progress"] = 0.75
        indexing_state["message"] = "Сохранение в ChromaDB..."
        
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
            
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                end_idx = min(i + batch_size, len(embeddings))
                batch_emb = embeddings[i:end_idx]
                
                if len(batch_emb) > 0 and len(batch) > 0:
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
            collection_new = chroma_client_local.get_or_create_collection(
                name="obsidian_vault",
                metadata={"hnsw:space": "cosine"}
            )
            
            changed_sources = set(doc["source"] for doc in documents)
            if collection is not None and changed_sources:
                try:
                    existing = collection.get(include=["metadatas"])
                    ids_to_delete = [id_ for id_, meta in zip(existing["ids"], existing["metadatas"])
                                     if meta.get("source") in changed_sources]
                    if ids_to_delete:
                        collection.delete(ids=ids_to_delete)
                except:
                    pass
            
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
        
        if full_rebuild:
            save_index_metadata(db_path, new_meta)
        else:
            save_index_metadata(db_path, {**old_meta, **new_meta})
        
        indexing_state["stage"] = "done"
        indexing_state["progress"] = 1.0
        indexing_state["message"] = f"Готово: {collection_new.count()} чанков"
        
        return collection_new, bm25_new, all_chunks
        
    except Exception as e:
        indexing_state["error"] = str(e)
        indexing_state["stage"] = "error"
        indexing_state["message"] = f"Ошибка: {str(e)[:100]}"
        return None, bm25, bm25_chunks
    
    finally:
        indexing_state["running"] = False