#!/usr/bin/env python3
"""Микро-сервер реранкера Nemotron с MPS-ускорением."""

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_MODEL_NAME = "nvidia/Llama-Nemotron-Rerank-1B-v2"
_MAX_LENGTH = 1024
_HOST = "0.0.0.0"
_PORT = 8001

app = FastAPI(title="Nemotron Rerank Server")

# --- Устройство ---
_device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Загрузка модели ---
print("=" * 50)
print("🚀 Запуск Nemotron Rerank Server...")
print("=" * 50)
print(f"💻 Устройство: {_device}")
print(f"⏳ Загрузка модели {_MODEL_NAME}...")

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, trust_remote_code=True)
_model = AutoModelForSequenceClassification.from_pretrained(
    _MODEL_NAME,
    trust_remote_code=True,
)
_model.to(_device)
_model.eval()

print("✅ Модель загружена\n")


# --- Модели запросов ---

class RerankRequest(BaseModel):
    """Запрос на реранкинг."""

    query: str
    documents: list[str]


# --- Эндпоинты ---

@app.post("/rerank")
async def rerank(request: RerankRequest) -> dict:
    """Ранжирует документы по релевантности запросу.

    Args:
        request: Запрос с полем query и списком documents.

    Returns:
        Словарь с ключом results — список {"index": int, "relevance_score": float},
        отсортированный по убыванию релевантности.
    """
    scores: list[float] = []

    with torch.no_grad():
        for doc in request.documents:
            text = f"Query: {request.query}\nDocument: {doc}"
            inputs = _tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=_MAX_LENGTH,
            ).to(_device)

            outputs = _model(**inputs)
            scores.append(outputs.logits[0][0].item())

    results = sorted(
        (
            {"index": i, "relevance_score": score}
            for i, score in enumerate(scores)
        ),
        key=lambda r: r["relevance_score"],
        reverse=True,
    )

    return {"results": results}


@app.get("/health")
async def health() -> dict:
    """Эндпоинт проверки здоровья сервера."""
    return {"status": "ok", "device": _device}


# --- Точка входа ---

if __name__ == "__main__":
    uvicorn.run(app, host=_HOST, port=_PORT)
