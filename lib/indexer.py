"""Индексация Obsidian vault: чанки, BM25, векторный индекс."""

import logging
from datetime import datetime

from langchain_text_splitters import MarkdownTextSplitter
from more_itertools import chunked

from lib.config import config
from lib.db.database import Database
from lib.db.repositories import ChunkRepository, DocumentRepository
from lib.embedder import Embedder
from lib.schema import (
    ChunkCreate,
    DocumentCreate,
    DocumentResponse,
    IndexingState,
)

logger = logging.getLogger("obsidian-rag")


class Indexer:
    """Индексатор Obsidian vault.

    Процесс:
    1. Сканирует vault на .md-файлы
    2. Сравнивает с документами в БД по path и mtime
    3. Удаляет устаревшие, создаёт/обновляет актуальные
    4. Разбивает на чанки
    5. Получает эмбеддинги и сохраняет чанки в БД
    """

    def __init__(
        self,
        embedder: Embedder,
        database: Database,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.embedder = embedder
        self.database = database
        self.chunk_size = chunk_size or config.indexing.chunk_size
        self.chunk_overlap = chunk_overlap or config.indexing.chunk_overlap
        self.batch_size = batch_size or config.indexing.batch_size

    def run(
        self,
        reindex: bool,
        state: IndexingState,
    ) -> None:
        """Запускает индексацию vault в текущем потоке."""
        conn = self.database.new_connection()
        document_repository = DocumentRepository(conn)
        chunk_repository = ChunkRepository(conn)

        try:
            state.status = "running"
            state.started_at = datetime.now()
            state.message = ""

            if reindex:
                logger.info("Полная переиндексация...")
                state.message = "Полная переиндексация..."
                document_repository.delete_all()

            state.message = "Загрузка документов..."
            stale_document_ids, new_document_schemas = self._load_documents(
                document_repository,
            )

            if stale_document_ids:
                logger.info(f"Удаление {len(stale_document_ids)} устаревших документов...")
                state.message = (
                    f"Удаление {len(stale_document_ids)} устаревших документов..."
                )
                document_repository.delete_many(stale_document_ids)

            if not new_document_schemas:
                logger.info("Нет новых документов для индексации")
                state.status = "done"
                state.message = "Нет новых документов для индексации"
                return

            logger.info(f"Создание {len(new_document_schemas)} новых документов...")
            state.message = (
                f"Создание {len(new_document_schemas)} новых документов..."
            )
            new_documents = document_repository.create_many(new_document_schemas)

            document_ids, chunk_texts = self._split_documents(new_documents)

            total_chunks = len(chunk_texts)
            logger.info(f"Эмбеддинг {total_chunks} чанков батчами по {self.batch_size}...")
            state.message = f"Эмбеддинг {total_chunks} чанков..."
            chunk_embeddings = self._encode_chunks(chunk_texts, state)

            logger.info(f"Сохранение {total_chunks} чанков...")
            state.message = f"Сохранение {total_chunks} чанков..."
            new_chunks = [
                ChunkCreate(
                    document_id=document_id,
                    text=text,
                    embedding=embedding,
                )
                for document_id, text, embedding in zip(
                    document_ids,
                    chunk_texts,
                    chunk_embeddings,
                )
            ]
            chunk_repository.create_many(new_chunks)

            state.status = "done"
            state.message = (
                f"Индексация завершена: {len(new_documents)} документов, "
                f"{total_chunks} чанков"
            )
            logger.info(state.message)

        except Exception as e:
            logger.error(f"Ошибка индексации: {e}", exc_info=True)
            state.status = "error"
            state.message = f"Ошибка индексации: {e}"
        finally:
            conn.close()
            logger.debug("Соединение индексатора закрыто")

    def _load_documents(
        self,
        document_repository: DocumentRepository,
    ) -> tuple[list[str], list[DocumentCreate]]:
        """Сканирует vault и возвращает списки устаревших и новых документов."""
        stale_document_ids: list[str] = []
        new_document_schemas: list[DocumentCreate] = []

        existing_docs = document_repository.get_all()
        document_mapping = {
            doc.path: doc
            for doc in existing_docs
        }

        logger.info(f"Сканирование vault: {config.paths.vault_path}")
        md_files = list(config.paths.vault_path.rglob("*.md"))
        logger.info(f"Найдено .md файлов: {len(md_files)}")

        for md_file in md_files:
            if any(part.startswith(".") for part in md_file.parts):
                continue

            path = str(md_file.relative_to(config.paths.vault_path))
            modified_dt = datetime.fromtimestamp(md_file.stat().st_mtime)
            current = document_mapping.get(path)

            if current:
                if current.created_at >= modified_dt:
                    continue
                stale_document_ids.append(current.document_id)

            text: str | None = None
            try:
                text = md_file.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(f"Ошибка чтения {md_file}: {e}")

            if not text:
                continue

            new_document_schemas.append(DocumentCreate(path=path, text=text))

        logger.info(
            f"Устаревших: {len(stale_document_ids)}, "
            f"новых: {len(new_document_schemas)}"
        )
        return stale_document_ids, new_document_schemas

    def _split_documents(
        self,
        documents: list[DocumentResponse],
    ) -> tuple[list[str], list[str]]:
        """Разбивает документы на чанки."""
        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        document_ids: list[str] = []
        chunk_texts: list[str] = []

        for document in documents:
            chunks = splitter.split_text(document.text)
            for chunk_text in chunks:
                document_ids.append(document.document_id)
                chunk_texts.append(chunk_text)

        logger.info(
            f"Разбито {len(documents)} документов на {len(chunk_texts)} чанков"
        )
        return document_ids, chunk_texts

    def _encode_chunks(
        self,
        chunk_texts: list[str],
        state: IndexingState,
    ) -> list[list[float]]:
        """Получает эмбеддинги для чанков батчами."""
        total = len(chunk_texts)
        chunk_embeddings: list[list[float]] = []

        for i, texts in enumerate(chunked(chunk_texts, self.batch_size)):
            embeddings = self.embedder.embed_many(list(texts))
            chunk_embeddings.extend(embeddings.tolist())

            processed = min((i + 1) * self.batch_size, total)
            state.message = f"Эмбеддинг чанков: {processed}/{total}"

            if i % 10 == 0:
                logger.debug(f"Прогресс эмбеддинга: {processed}/{total}")

        logger.info(f"Эмбеддинг завершён: {len(chunk_embeddings)} векторов")
        return chunk_embeddings
