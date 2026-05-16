"""Гибридный поиск: BM25 + векторный + RRF."""

import re
import numpy as np


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def retrieve_context_hybrid(
    question: str,
    collection,
    bm25,
    bm25_chunks,
    embeddings_url: str,
    embeddings_model: str,
    top_k: int = 30,
    bm25_weight: float = 0.3,
    rrf_k: int = 60
) -> list[dict]:
    """Гибридный поиск."""
    from lib.embeddings import get_embeddings
    
    if collection is None:
        return []
    
    query_embedding = get_embeddings(embeddings_url, embeddings_model, [question])[0].tolist()

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,
        include=["documents", "metadatas", "distances"]
    )
    
    bm25_scores = []
    if bm25 is not None:
        tokenized_query = tokenize(question)
        bm25_scores = list(bm25.get_scores(tokenized_query))
    
    vector_weight = 1.0 - bm25_weight
    
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