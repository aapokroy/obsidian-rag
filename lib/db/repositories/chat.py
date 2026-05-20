"""Repository for chat metadata."""

import logging
import sqlite3
import uuid
from datetime import datetime

from lib.config import config
from lib.db.repositories.base import BaseRepository
from lib.schema import ChatResponse

logger = logging.getLogger("obsidian-rag")


class ChatRepository(BaseRepository):
    """Repository for chat metadata."""

    def create(self) -> ChatResponse:
        """Creates a new chat with the default title."""
        chat_id = str(uuid.uuid4())
        now = datetime.now()
        title = config.chat.default_title

        with self.conn:
            self.execute(
                """
                INSERT INTO chats (chat_id, title, updated_at, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    chat_id,
                    title,
                    now,
                    now,
                ),
            )

        logger.debug("Chat created: chat_id=%s", chat_id)
        return ChatResponse(
            chat_id=chat_id,
            title=title,
            updated_at=now,
            created_at=now,
        )

    def get_by_id(self, chat_id: str) -> ChatResponse | None:
        """Returns a chat by id or None."""
        row = self.fetch_one(
            "SELECT * FROM chats WHERE chat_id = ?",
            (chat_id,),
        )
        if row is None:
            return None
        return self._to_response(row)

    def get_all(self) -> list[ChatResponse]:
        """Returns all chats from newest to oldest."""
        rows = self.fetch_all(
            """
            SELECT *
            FROM chats
            ORDER BY updated_at DESC
            """,
        )
        return [self._to_response(row) for row in rows]

    def exists(self, chat_id: str) -> bool:
        """Checks whether a chat exists."""
        row = self.fetch_one(
            "SELECT 1 FROM chats WHERE chat_id = ?",
            (chat_id,),
        )
        return row is not None

    def delete(self, chat_id: str) -> None:
        """Deletes a chat and related messages through cascade."""
        with self.conn:
            self.execute(
                "DELETE FROM chats WHERE chat_id = ?",
                (chat_id,),
            )
        logger.debug("Chat deleted: chat_id=%s", chat_id)

    def update_title(
        self,
        chat_id: str,
        title: str,
        *,
        commit: bool = True,
    ) -> None:
        """Updates a chat title."""
        self.execute(
            "UPDATE chats SET title = ? WHERE chat_id = ?",
            (
                title,
                chat_id,
            ),
        )
        if commit:
            self.conn.commit()
        logger.debug("Chat title updated: chat_id=%s", chat_id)

    def touch(
        self,
        chat_id: str,
        *,
        commit: bool = True,
    ) -> None:
        """Updates the chat last-activity timestamp."""
        self.execute(
            "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE chat_id = ?",
            (chat_id,),
        )
        if commit:
            self.conn.commit()
        logger.debug("Chat touched: chat_id=%s", chat_id)

    @staticmethod
    def _to_response(row: sqlite3.Row) -> ChatResponse:
        """Converts a SQLite row into ChatResponse."""
        return ChatResponse(
            chat_id=row["chat_id"],
            title=row["title"],
            updated_at=row["updated_at"],
            created_at=row["created_at"],
        )
