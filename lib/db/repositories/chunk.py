"""Репозиторий для работы с чанками."""
import struct
import uuid

import numpy as np
import sqlite3

from lib.db.models import Chunk
from lib.schema import ChunkCreate, ChunkResponse


class ChunkRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create_many(self, schemas: list[ChunkCreate]) -> list[ChunkResponse]:
        chunks = [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=schema.document_id,
                text=schema.text,
            )
            for schema in schemas
        ]
        embeddings = [
            np.array(schema.embedding, dtype=np.float32)
            for schema in schemas
        ]

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")

        self.conn.execute("BEGIN TRANSACTION")
        try:
            self.conn.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        c.text,
                        c.created_at,
                    )
                    for c in chunks
                ],
            )

            for chunk, embedding in zip(chunks, embeddings):
                emb_bytes = struct.pack(
                    f"{len(embedding)}f",
                    *embedding,
                )
                self.conn.execute(
                    "INSERT INTO chunks_vec VALUES (?, ?)",
                    (
                        chunk.chunk_id,
                        emb_bytes,
                    ),
                )

            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

        return [
            ChunkResponse(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
            )
            for chunk in chunks
        ]
