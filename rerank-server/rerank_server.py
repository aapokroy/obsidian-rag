#!/usr/bin/env python3
"""FastAPI server for local document reranking."""

import logging

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_MODEL_NAME = "nvidia/Llama-Nemotron-Rerank-1B-v2"
_MAX_LENGTH = 1024
_HOST = "0.0.0.0"
_PORT = 8001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rerank-server")

app = FastAPI(title="Nemotron Rerank Server")
_device = "mps" if torch.backends.mps.is_available() else "cpu"

logger.info("=" * 50)
logger.info("Starting Nemotron Rerank Server")
logger.info("=" * 50)
logger.info("Device: %s", _device)
logger.info("Loading model %s...", _MODEL_NAME)

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, trust_remote_code=True)
_model = AutoModelForSequenceClassification.from_pretrained(
    _MODEL_NAME,
    trust_remote_code=True,
)
_model.to(_device)
_model.eval()

logger.info("Model loaded")


class RerankRequest(BaseModel):
    """Request payload for reranking a list of documents."""

    query: str
    documents: list[str]


@app.post("/rerank")
async def rerank(request: RerankRequest) -> dict:
    """Ranks documents by relevance to the query."""
    logger.info("Rerank request: documents=%s", len(request.documents))
    scores: list[float] = []

    with torch.no_grad():
        for document in request.documents:
            text = f"Query: {request.query}\nDocument: {document}"
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
            {"index": index, "relevance_score": score}
            for index, score in enumerate(scores)
        ),
        key=lambda result: result["relevance_score"],
        reverse=True,
    )
    logger.debug("Rerank response prepared: results=%s", len(results))
    return {"results": results}


@app.get("/health")
async def health() -> dict:
    """Returns server status and active device."""
    return {"status": "ok", "device": _device}


if __name__ == "__main__":
    uvicorn.run(app, host=_HOST, port=_PORT)
