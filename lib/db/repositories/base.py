"""Shared utilities for SQLite repositories."""

import sqlite3
from collections.abc import Iterable, Sequence


class BaseRepository:
    """Shared thin wrapper around sqlite3.Connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Stores the SQLite connection for a concrete repository."""
        self.conn = conn

    def execute(
        self,
        query: str,
        params: Sequence[object] = (),
    ) -> sqlite3.Cursor:
        """Executes one SQL statement and returns a cursor."""
        return self.conn.execute(query, params)

    def executemany(
        self,
        query: str,
        params: Iterable[Sequence[object]],
    ) -> sqlite3.Cursor:
        """Executes one SQL statement for a sequence of parameters."""
        return self.conn.executemany(query, params)

    def fetch_one(
        self,
        query: str,
        params: Sequence[object] = (),
    ) -> sqlite3.Row | None:
        """Returns one result row or None."""
        return self.execute(query, params).fetchone()

    def fetch_all(
        self,
        query: str,
        params: Sequence[object] = (),
    ) -> list[sqlite3.Row]:
        """Returns all result rows."""
        return self.execute(query, params).fetchall()
