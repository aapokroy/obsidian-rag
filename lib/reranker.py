"""Реранкинг через внешний API."""

import numpy as np
import requests


def rerank_chunks(
    reranker_url: str,
    question: str,
    chunks: list[dict],
    top_k: int = 5,
    fallback: bool = True
) -> list[dict]:
    """Переранжирование чанков через API реранкера."""
    if not chunks or len(chunks) <= top_k:
        chunks.sort(key=lambda x: x["distance"])
        for c in chunks:
            c["relevance"] = round(1.0 - c["distance"], 4)
        return chunks[:top_k]
    
    try:
        response = requests.post(
            reranker_url,
            json={
                "query": question,
                "documents": [c["text"][:1024] for c in chunks]
            },
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Статус {response.status_code}")
        
        data = response.json()
        results = data.get("results", [])
        scores = [r.get("relevance_score", 1.0) for r in results]
        
        scores = np.array(scores, dtype=np.float64)
        scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        normalized = exp_scores / np.sum(exp_scores)
        
        result = []
        for item, score in zip(results, normalized):
            idx = item.get("index", 0)
            if idx < len(chunks):
                chunk = chunks[idx]
                chunk["relevance"] = round(float(score), 4)
                result.append(chunk)
        
        return result[:top_k]
        
    except Exception as e:
        print(f"⚠️ Ошибка реранкера: {e}, fallback на distance")
        if fallback:
            chunks.sort(key=lambda x: x["distance"])
            for c in chunks:
                c["relevance"] = round(1.0 - c["distance"], 4)
            return chunks[:top_k]
        raise