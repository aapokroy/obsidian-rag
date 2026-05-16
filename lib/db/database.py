# lib/db/database.py
"""Управление подключением к БД и инициализация схемы."""
import sqlite3
from pathlib import Path

import sqlite_vec

from lib.config import config


class Database:
    """Управляет подключением к SQLite и инициализацией схемы."""

    def __init__(
        self,
        db_path: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self.db_path = db_path or config.paths.db_path
        self.embedding_dim = embedding_dim or config.embeddings.dimension

        # Создаём директорию для БД, если её нет
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Основное соединение
        self.conn = self._create_connection()
        self._init_schema()

    def _create_connection(self) -> sqlite3.Connection:
        """Создаёт новое соединение с БД."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return conn

    def new_connection(self) -> sqlite3.Connection:
        """Создаёт новое соединение для использования в другом потоке."""
        return self._create_connection()

    def _init_schema(self) -> None:
        """Применяет SQL-схему и создаёт векторную таблицу."""
        schema_path = Path(__file__).parent / "migrations" / "schema.sql"

        with open(schema_path, "r", encoding="utf-8") as f:
            self.conn.executescript(f.read())

        try:
            self.conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                    chunk_id TEXT PRIMARY KEY,
                    embedding FLOAT[{self.embedding_dim}]
                )
                """,
            )
        except Exception:
            pass

    def close(self) -> None:
        """Закрывает соединение с БД."""
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        self.close()
