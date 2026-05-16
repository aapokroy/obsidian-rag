"""Репозиторий для работы с сообщениями."""
import uuid
from datetime import datetime

import sqlite3

from lib.db.repositories.chat import ChatRepository
from lib.schema import MessageCreate, MessageResponse


class MessageRepository:
    def __init__(self, conn: sqlite3.Connection, chat_repository: ChatRepository) -> None:
        self.conn = conn
        self.chat_repository = chat_repository

    def create(
        self,
        chat_id: str,
        data: MessageCreate,
    ) -> MessageResponse:
        message_id = str(uuid.uuid4())
        now = datetime.now()

        count = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()["cnt"]

        self.conn.execute(
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

        self.chat_repository.touch(chat_id)

        if count == 0:
            title = data.text[:50] + ("..." if len(data.text) > 50 else "")
            self.chat_repository.update_title(chat_id, title)

        self.conn.commit()

        return MessageResponse(
            message_id=message_id,
            chat_id=chat_id,
            text=data.text,
            role=data.role,
            created_at=now,
        )

    def get_by_chat(self, chat_id: str) -> list[MessageResponse]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
            (chat_id,),
        ).fetchall()
        messages = [
            MessageResponse(
                message_id=row["message_id"],
                chat_id=row["chat_id"],
                text=row["text"],
                role=row["role"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
        return messages
