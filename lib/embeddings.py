"""Эмбеддинги через внешний API (LM Studio)."""

import numpy as np
import requests


def get_embeddings(embeddings_url: str, embeddings_model: str, texts: list[str]) -> np.ndarray:
    """Получение эмбеддингов через API."""
    response = requests.post(
        embeddings_url,
        json={
            "model": embeddings_model,
            "input": texts
        },
        timeout=60
    )
    response.raise_for_status()
    data = response.json()
    embeddings = np.array([item["embedding"] for item in data["data"]])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms