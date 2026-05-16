#!/usr/bin/env python3
"""Микро-сервер реранкера Nemotron с MPS-ускорением."""

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

app = FastAPI(title="Nemotron Rerank Server")

print("=" * 50)
print("🚀 Запуск Nemotron Rerank Server...")
print("=" * 50)

# Устройство
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"💻 Устройство: {device}")

# Загрузка модели
print("⏳ Загрузка модели nvidia/Llama-Nemotron-Rerank-1B-v2...")
tokenizer = AutoTokenizer.from_pretrained(
    "nvidia/Llama-Nemotron-Rerank-1B-v2",
    trust_remote_code=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    "nvidia/Llama-Nemotron-Rerank-1B-v2",
    trust_remote_code=True
)
model.to(device)
model.eval()
print("✅ Модель загружена\n")


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


@app.post("/rerank")
async def rerank(request: RerankRequest):
    """Реранкинг документов."""
    scores = []
    with torch.no_grad():
        for doc in request.documents:
            text = f"Query: {request.query}\nDocument: {doc}"
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(device)
            
            outputs = model(**inputs)
            score = outputs.logits[0][0].item()
            scores.append(score)
    
    results = [
        {"index": i, "relevance_score": score}
        for i, score in enumerate(scores)
    ]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return {"results": results}


@app.get("/health")
async def health():
    return {"status": "ok", "device": device}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
