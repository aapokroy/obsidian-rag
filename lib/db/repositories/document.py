"""Repository for indexed documents."""

import logging
import sqlite3
import uuid

from lib.db.models import Document
from lib.db.repositories.base import BaseRepository
from lib.schema import DocumentCreate, DocumentResponse

logger = logging.getLogger("obsidian-rag")


class DocumentRepository(BaseRepository):
    """Repository for Obsidian documents."""

    def create_one(self, schema: DocumentCreate) -> DocumentResponse:
        """Creates one document."""
        return self.create_many([schema])[0]

    def create_many(
        self,
        schemas: list[DocumentCreate],
    ) -> list[DocumentResponse]:
        """Creates multiple documents and returns their DTOs."""
        documents = [
            Document(
                document_id=str(uuid.uuid4()),
                path=schema.path,
                text=schema.text,
            )
            for schema in schemas
        ]

        if not documents:
            return []

        with self.conn:
            self.executemany(
                """
                INSERT OR REPLACE INTO documents (document_id, path, text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        document.document_id,
                        document.path,
                        document.text,
                        document.created_at,
                    )
                    for document in documents
                ],
            )

        logger.debug("Documents created: count=%s", len(documents))
        return [self._to_response(document) for document in documents]

    def get_all(self) -> list[DocumentResponse]:
        """Returns all documents from newest to oldest."""
        rows = self.fetch_all(
            "SELECT * FROM documents ORDER BY created_at DESC",
        )
        return [self._row_to_response(row) for row in rows]

    def delete_many(self, document_ids: list[str]) -> int:
        """Deletes documents by id and returns the requested deletion count."""
        if not document_ids:
            return 0

        with self.conn:
            self.executemany(
                "DELETE FROM documents WHERE document_id = ?",
                [(document_id,) for document_id in document_ids],
            )
        logger.debug("Documents deleted: count=%s", len(document_ids))
        return len(document_ids)

    def delete_all(self) -> None:
        """Deletes all documents and related chunks."""
        with self.conn:
            self.execute("DELETE FROM documents")
        logger.debug("All documents deleted")

    @staticmethod
    def _to_response(document: Document) -> DocumentResponse:
        """Converts a Document dataclass into DocumentResponse."""
        return DocumentResponse(
            document_id=document.document_id,
            path=document.path,
            text=document.text,
            created_at=document.created_at,
        )

    @staticmethod
    def _row_to_response(row: sqlite3.Row) -> DocumentResponse:
        """Converts a SQLite row into DocumentResponse."""
        return DocumentResponse(
            document_id=row["document_id"],
            path=row["path"],
            text=row["text"],
            created_at=row["created_at"],
        )
