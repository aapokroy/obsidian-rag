"""Pydantic schemas for requests, responses, and internal DTOs."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

MessageRole = Literal["user", "assistant", "system"]
IndexingStatus = Literal["idle", "running", "done", "error"]


class DocumentCreate(BaseModel):
    """Data required to create an indexed document."""

    path: str
    text: str


class DocumentResponse(BaseModel):
    """Document returned by the storage layer."""

    document_id: str
    path: str
    text: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChunkCreate(BaseModel):
    """Data required to create a chunk and its embedding."""

    document_id: str
    text: str
    embedding: list[float]


class ChunkResponse(BaseModel):
    """Chunk returned by the storage layer."""

    chunk_id: str
    document_id: str
    text: str


class ChatResponse(BaseModel):
    """Chat metadata."""

    chat_id: str
    title: str
    updated_at: datetime
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageCreate(BaseModel):
    """Data required to create a chat message."""

    text: str
    role: MessageRole


class MessageResponse(BaseModel):
    """Chat message returned by the API."""

    message_id: str
    chat_id: str
    text: str
    role: MessageRole
    created_at: datetime

    model_config = {"from_attributes": True}


class RetrieveResultItem(BaseModel):
    """Result item from the initial hybrid search."""

    chunk_id: str
    document_id: str
    text: str
    document_path: str
    rrf_score: float


class RerankResultItem(BaseModel):
    """Result item after reranking."""

    chunk_id: str
    document_id: str
    text: str
    document_path: str
    relevance: float


class SourceResponse(BaseModel):
    """Source sent to the client with the answer."""

    document_path: str
    text: str
    relevance: float


class IndexingState(BaseModel):
    """Current state of background indexing."""

    status: IndexingStatus = "idle"
    message: str = ""
    started_at: datetime | None = None


class QueryRequest(BaseModel):
    """User request for the streaming endpoint."""

    query: str
    chat_id: str
