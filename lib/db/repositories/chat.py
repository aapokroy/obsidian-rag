"""Репозиторий для работы с чатами."""
import uuid
from datetime import datetime

import sqlite3

from lib.schema import ChatResponse


class ChatRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self) -> ChatResponse:
        chat_id = str(uuid.uuid4())
        now = datetime.now()

        self.conn.execute(
            """
            INSERT INTO chats (chat_id, title, updated_at, created_at)
            VALUES (?, 'Без названия', ?, ?)
            """,
            (
                chat_id,
                now,
                now,
            ),
        )
        self.conn.commit()

        return ChatResponse(
            chat_id=chat_id,
            title="Без названия",
            updated_at=now,
            created_at=now,
        )

    def get_all(self) -> list[ChatResponse]:
        rows = self.conn.execute(
            """
            SELECT c.*
            FROM chats c
            LEFT JOIN messages m ON c.chat_id = m.chat_id
            GROUP BY c.chat_id
            ORDER BY c.updated_at DESC
            """,
        ).fetchall()
        chats = [
            ChatResponse(
                chat_id=row["chat_id"],
                title=row["title"],
                updated_at=row["updated_at"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
        return chats

    def delete(self, chat_id: str) -> None:
        self.conn.execute(
            "DELETE FROM chats WHERE chat_id = ?",
            (chat_id,),
        )
        self.conn.commit()

    def update_title(
        self,
        chat_id: str,
        title: str,
    ) -> None:
        self.conn.execute(
            "UPDATE chats SET title = ? WHERE chat_id = ?",
            (
                title,
                chat_id,
            ),
        )
        self.conn.commit()

    def touch(self, chat_id: str) -> None:
        self.conn.execute(
            "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE chat_id = ?",
            (chat_id,),
        )
        self.conn.commit()
