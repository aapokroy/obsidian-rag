"""DTO для работы с базой данных."""
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class Document:
    document_id: str
    path: str
    text: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class Chat:
    chat_id: str
    title: str = "Без названия"
    updated_at: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class Message:
    message_id: str
    chat_id: str
    text: str
    role: str
    created_at: datetime = field(default_factory=datetime.now)
