"""Hybrid BM25 + sqlite-vec retrieval with RRF fusion."""

import logging
import sqlite3
import struct

import numpy as np
import numpy.typing as npt

from lib.config import config
from lib.db.database import Database
from lib.embedder import Embedder
from lib.schema import RetrieveResultItem

logger = logging.getLogger("obsidian-rag")


class Retriever:
    """Hybrid retriever: FTS5 BM25 + sqlite-vec + Reciprocal Rank Fusion."""

    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        top_k: int | None = None,
        bm25_weight: float | None = None,
        rrf_k: int | None = None,
    ) -> None:
        """Initializes the retriever and RRF fusion parameters."""
        self.db = db
        self.embedder = embedder
        self.top_k = top_k or config.search.top_k_retrieval
        self.bm25_weight = bm25_weight or config.search.bm25_weight
        self.rrf_k = rrf_k or config.search.rrf_k
        logger.debug(
            "Retriever initialized: top_k=%s, bm25_weight=%s, rrf_k=%s",
            self.top_k,
            self.bm25_weight,
            self.rrf_k,
        )

    def hybrid_search(self, query: str) -> list[RetrieveResultItem]:
        """Runs hybrid search and returns results in RRF order."""
        query = query.strip()
        if not query:
            logger.debug("Retrieval skipped: empty query")
            return []

        logger.debug("Retrieval started: query=%r", query[:80])
        limit = self.top_k * 2
        bm25_rows = self._search_bm25(query=query, limit=limit)
        vector_rows = self._search_vector(
            embedding=self.embedder.embed(query),
            limit=limit,
        )
        ranked_scores = self._fuse_scores(bm25_rows, vector_rows)
        results = self._fetch_results(ranked_scores[: self.top_k])
        logger.debug(
            "Retrieval finished: bm25=%s, vector=%s, returned=%s",
            len(bm25_rows),
            len(vector_rows),
            len(results),
        )
        return results

    def _fuse_scores(
        self,
        bm25_rows: list[sqlite3.Row],
        vector_rows: list[sqlite3.Row],
    ) -> list[tuple[str, float]]:
        """Combines BM25 and vector-search ranks through RRF."""
        scores: dict[str, float] = {}
        vector_weight = 1.0 - self.bm25_weight

        self._add_rrf_scores(scores, bm25_rows, weight=self.bm25_weight)
        self._add_rrf_scores(scores, vector_rows, weight=vector_weight)

        return sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

    def _add_rrf_scores(
        self,
        scores: dict[str, float],
        rows: list[sqlite3.Row],
        *,
        weight: float,
    ) -> None:
        """Adds one search channel contribution to the total RRF score."""
        for rank, row in enumerate(rows, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (self.rrf_k + rank)

    def _fetch_results(
        self,
        ranked_scores: list[tuple[str, float]],
    ) -> list[RetrieveResultItem]:
        """Loads chunk data while preserving RRF ranking order."""
        if not ranked_scores:
            return []

        chunk_ids = [chunk_id for chunk_id, _score in ranked_scores]
        placeholders = ", ".join("?" for _ in chunk_ids)
        rows = self.db.conn.execute(
            f"""
            SELECT c.chunk_id, c.document_id, c.text, d.path
            FROM chunks c
            JOIN documents d ON c.document_id = d.document_id
            WHERE c.chunk_id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
        row_by_id = {row["chunk_id"]: row for row in rows}

        results: list[RetrieveResultItem] = []
        for chunk_id, rrf_score in ranked_scores:
            row = row_by_id.get(chunk_id)
            if row is None:
                logger.warning("Chunk referenced by search result was not found: %s", chunk_id)
                continue
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
        """Searches candidates through FTS5 BM25."""
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
        embedding: npt.NDArray[np.float64],
        limit: int,
    ) -> list[sqlite3.Row]:
        """Searches candidates through sqlite-vec KNN."""
        return self.db.conn.execute(
            """
            SELECT chunk_id, distance
            FROM chunks_vec
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (
                self._pack_embedding(embedding),
                limit,
            ),
        ).fetchall()

    @staticmethod
    def _pack_embedding(embedding: npt.NDArray[np.float64]) -> bytes:
        """Packs an embedding into sqlite-vec binary format."""
        return struct.pack(f"{len(embedding)}f", *embedding)
