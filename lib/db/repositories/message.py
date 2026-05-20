"""Repository for chat messages."""

import logging
import sqlite3
import uuid
from datetime import datetime

from lib.config import config
from lib.db.repositories.base import BaseRepository
from lib.db.repositories.chat import ChatRepository
from lib.schema import MessageCreate, MessageResponse

logger = logging.getLogger("obsidian-rag")


class MessageRepository(BaseRepository):
    """Repository for chat messages."""

    def __init__(self, conn: sqlite3.Connection, chat_repository: ChatRepository) -> None:
        """Initializes the repository and ChatRepository dependency."""
        super().__init__(conn)
        self.chat_repository = chat_repository

    def create(
        self,
        chat_id: str,
        data: MessageCreate,
    ) -> MessageResponse:
        """Creates a message and atomically updates chat metadata."""
        message_id = str(uuid.uuid4())
        now = datetime.now()
        is_first_message = self._is_first_message(chat_id)

        with self.conn:
            self.execute(
                """
                INSERT INTO messages (message_id, chat_id, text, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    chat_id,
                    data.text,
                    data.role,
                    now,
                ),
            )

            self.chat_repository.touch(chat_id, commit=False)

            if is_first_message:
                title = self._make_title(data.text)
                self.chat_repository.update_title(chat_id, title, commit=False)

        logger.debug("Message created: message_id=%s, chat_id=%s, role=%s", message_id, chat_id, data.role)
        return MessageResponse(
            message_id=message_id,
            chat_id=chat_id,
            text=data.text,
            role=data.role,
            created_at=now,
        )

    def get_by_chat(self, chat_id: str) -> list[MessageResponse]:
        """Returns chat messages in chronological order."""
        rows = self.fetch_all(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
            (chat_id,),
        )
        return [self._to_response(row) for row in rows]

    def _is_first_message(self, chat_id: str) -> bool:
        """Checks whether the next message is the first in the chat."""
        row = self.fetch_one(
            "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ?",
            (chat_id,),
        )
        return row["cnt"] == 0

    @staticmethod
    def _make_title(text: str) -> str:
        """Builds a short chat title from the first message."""
        title = text.strip()
        max_length = config.chat.title_max_length
        if len(title) <= max_length:
            return title
        return title[:max_length].rstrip() + "..."

    @staticmethod
    def _to_response(row: sqlite3.Row) -> MessageResponse:
        """Converts a SQLite row into MessageResponse."""
        return MessageResponse(
            message_id=row["message_id"],
            chat_id=row["chat_id"],
            text=row["text"],
            role=row["role"],
            created_at=row["created_at"],
        )
