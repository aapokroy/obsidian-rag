"""Internal dataclass models for the storage layer."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class Document:
    """Obsidian document stored in SQLite."""

    document_id: str
    path: str
    text: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class Chunk:
    """Text fragment of a document."""

    chunk_id: str
    document_id: str
    text: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class Chat:
    """User chat metadata."""

    chat_id: str
    title: str = "Без названия"
    updated_at: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class Message:
    """Message inside a chat."""

    message_id: str
    chat_id: str
    text: str
    role: str
    created_at: datetime = field(default_factory=datetime.now)
