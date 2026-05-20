"""Repository for chunks and vector rows."""

import logging
import struct
import uuid

import numpy as np

from lib.db.models import Chunk
from lib.db.repositories.base import BaseRepository
from lib.schema import ChunkCreate, ChunkResponse

logger = logging.getLogger("obsidian-rag")


class ChunkRepository(BaseRepository):
    """Repository for text chunks and their vectors."""

    def create_many(self, schemas: list[ChunkCreate]) -> list[ChunkResponse]:
        """Creates chunks and matching sqlite-vec rows."""
        chunks = [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=schema.document_id,
                text=schema.text,
            )
            for schema in schemas
        ]

        if not chunks:
            return []

        with self.conn:
            self.executemany(
                """
                INSERT INTO chunks (chunk_id, document_id, text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.text,
                        chunk.created_at,
                    )
                    for chunk in chunks
                ],
            )
            self.executemany(
                """
                INSERT INTO chunks_vec (chunk_id, embedding)
                VALUES (?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        self._pack_embedding(schema.embedding),
                    )
                    for chunk, schema in zip(chunks, schemas)
                ],
            )

        logger.debug("Chunks created: count=%s", len(chunks))
        return [self._to_response(chunk) for chunk in chunks]

    @staticmethod
    def _pack_embedding(embedding: list[float]) -> bytes:
        """Packs an embedding into sqlite-vec binary format."""
        vector = np.array(embedding, dtype=np.float64)
        return struct.pack(f"{len(vector)}f", *vector)

    @staticmethod
    def _to_response(chunk: Chunk) -> ChunkResponse:
        """Converts a Chunk dataclass into ChunkResponse."""
        return ChunkResponse(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=chunk.text,
        )
