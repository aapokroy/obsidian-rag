"""Репозиторий для работы с документами."""
import uuid

import sqlite3

from lib.db.models import Document
from lib.schema import DocumentCreate, DocumentResponse


class DocumentRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create_one(self, schema: DocumentCreate) -> DocumentResponse:
        document = Document(
            document_id=str(uuid.uuid4()),
            path=schema.path,
            text=schema.text,
        )
        self.conn.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?)",
            (
                document.document_id,
                document.path,
                document.text,
                document.created_at,
            ),
        )
        self.conn.commit()
        return DocumentResponse(
            document_id=document.document_id,
            path=document.path,
            text=document.text,
            created_at=document.created_at,
        )

    def create_many(
        self,
        schemas: list[DocumentCreate],
    ) -> list[DocumentResponse]:
        """Создать множество документов и вернуть их представления."""
        documents = [
            Document(
                document_id=str(uuid.uuid4()),
                path=schema.path,
                text=schema.text,
            )
            for schema in schemas
        ]

        self.conn.executemany(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?)",
            [
                (
                    d.document_id,
                    d.path,
                    d.text,
                    d.created_at,
                )
                for d in documents
            ],
        )
        self.conn.commit()

        return [
            DocumentResponse(
                document_id=d.document_id,
                path=d.path,
                text=d.text,
                created_at=d.created_at,
            )
            for d in documents
        ]

    def get_all(self) -> list[DocumentResponse]:
        rows = self.conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC",
        ).fetchall()
        documents = [
            DocumentResponse(
                document_id=row["document_id"],
                path=row["path"],
                text=row["text"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
        return documents

    def delete_many(self, document_ids: list[str]) -> int:
        self.conn.executemany(
            "DELETE FROM documents WHERE document_id = ?",
            [(doc_id,) for doc_id in document_ids],
        )
        self.conn.commit()
        return len(document_ids)

    def delete_all(self) -> None:
        self.conn.execute("DELETE FROM documents")
        self.conn.commit()
