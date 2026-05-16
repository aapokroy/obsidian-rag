"""Pydantic схемы для веб-приложения."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


# ---------- Документы ----------

class DocumentCreate(BaseModel):
    path: str
    text: str


class DocumentResponse(BaseModel):
    document_id: str
    path: str
    text: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------- Чанки ----------

class ChunkCreate(BaseModel):
    document_id: str
    text: str
    embedding: list[float]


class ChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    text: str


# ---------- Чаты ----------

class ChatResponse(BaseModel):
    chat_id: str
    title: str
    updated_at: datetime
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------- Сообщения ----------

class MessageCreate(BaseModel):
    text: str
    role: Literal['user', 'assistant', 'system']


class MessageResponse(BaseModel):
    message_id: str
    chat_id: str
    text: str
    role: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------- Поиск ----------


class RetrieveResultItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    document_path: str
    rrf_score: float


class RerankResultItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    document_path: str
    relevance: float


# ---------- Статус UI ----------

class IndexingState(BaseModel):
    status: Literal['idle', 'running', 'done', 'error'] = "idle"
    message: str = ""
    started_at: datetime | None = None


class QueryRequest(BaseModel):
    query: str
    chat_id: str
