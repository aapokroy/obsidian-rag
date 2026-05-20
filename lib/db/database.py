"""Database connection management and schema initialization."""

import logging
import sqlite3
from pathlib import Path

import sqlite_vec

from lib.config import config

logger = logging.getLogger("obsidian-rag")


class Database:
    """Manages SQLite connections and schema initialization."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        """Opens the main connection and applies the database schema."""
        self.db_path = Path(db_path or config.paths.db_path)
        self.embedding_dim = embedding_dim or config.embeddings.dimension

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = self._create_connection()
        self._init_schema()

    def _create_connection(self) -> sqlite3.Connection:
        """Creates a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return conn

    def new_connection(self) -> sqlite3.Connection:
        """Creates a new connection for another thread."""
        return self._create_connection()

    def _init_schema(self) -> None:
        """Applies the SQL schema and creates the vector table."""
        schema_path = Path(__file__).parent / "migrations" / "schema.sql"
        self.conn.executescript(schema_path.read_text(encoding="utf-8"))
        self._init_vector_table()
        self.conn.commit()

    def _init_vector_table(self) -> None:
        """Creates the sqlite-vec virtual table for embeddings."""
        try:
            self.conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                    chunk_id TEXT PRIMARY KEY,
                    embedding FLOAT[{self.embedding_dim}]
                )
                """,
            )
        except sqlite3.Error as exc:
            logger.warning("Failed to create chunks_vec: %s", exc)

    def close(self) -> None:
        """Closes the main database connection."""
        self.conn.close()

    def __enter__(self) -> "Database":
        """Returns self for Database context-manager usage."""
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Closes the connection when leaving the context manager."""
        self.close()
