"""Гибридный поиск: BM25 + векторный с RRF-фьюжном."""

import struct
import sqlite3

import numpy as np

from lib.config import config
from lib.db.database import Database
from lib.embedder import Embedder
from lib.schema import RetrieveResultItem


class Retriever:
    """Гибридный ретривер: BM25 (FTS5) + векторный (sqlite-vec) с RRF."""

    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        top_k: int | None = None,
        bm25_weight: float | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.top_k = top_k or config.search.top_k_retrieval
        self.bm25_weight = bm25_weight or config.search.bm25_weight
        self.rrf_k = rrf_k or config.search.rrf_k

    def hybrid_search(self, query: str) -> list[RetrieveResultItem]:
        """Выполняет гибридный поиск с BM25 и векторным поиском через RRF."""
        if not query.strip():
            return []

        bm25_rows = self._search_bm25(
            query=query,
            limit=self.top_k * 2,
        )

        query_embedding = self.embedder.embed(query)
        vector_rows = self._search_vector(
            embedding=query_embedding,
            limit=self.top_k * 2,
        )

        rrf_scores: dict[str, float] = {}
        vector_weight = 1.0 - self.bm25_weight

        for rank, row in enumerate(bm25_rows, start=1):
            rrf_scores[row["chunk_id"]] = self.bm25_weight / (self.rrf_k + rank)

        for rank, row in enumerate(vector_rows, start=1):
            cid = row["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + vector_weight / (self.rrf_k + rank)

        sorted_ids = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: self.top_k]

        results: list[RetrieveResultItem] = []
        for chunk_id, rrf_score in sorted_ids:
            row = self.db.conn.execute(
                """
                SELECT c.chunk_id, c.document_id, c.text, d.path
                FROM chunks c
                JOIN documents d ON c.document_id = d.document_id
                WHERE c.chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()

            if row:
                results.append(
                    RetrieveResultItem(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        text=row["text"],
                        document_path=row["path"],
                        rrf_score=round(rrf_score, 5),
                    ),
                )

        return results

    def _search_bm25(
        self,
        query: str,
        limit: int,
    ) -> list[sqlite3.Row]:
        """Поиск через BM25 (FTS5)."""
        safe_query = query.replace('"', '""')
        return self.db.conn.execute(
            """
            SELECT c.chunk_id, bm25(chunks_fts) as score
            FROM chunks c
            JOIN chunks_fts fts ON c.rowid = fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (
                f'"{safe_query}"',
                limit,
            ),
        ).fetchall()

    def _search_vector(
        self,
        embedding: np.ndarray,
        limit: int,
    ) -> list[sqlite3.Row]:
        """Векторный поиск через sqlite-vec (HNSW)."""
        emb_bytes = struct.pack(
            f"{len(embedding)}f",
            *embedding,
        )
        return self.db.conn.execute(
            """
            SELECT chunk_id, distance
            FROM chunks_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (
                emb_bytes,
                limit,
            ),
        ).fetchall()
