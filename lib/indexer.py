"""Obsidian vault indexing: documents, chunks, and vector index."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_text_splitters import MarkdownTextSplitter
from more_itertools import chunked

from lib.config import config
from lib.db.database import Database
from lib.db.repositories import ChunkRepository, DocumentRepository
from lib.embedder import Embedder
from lib.schema import ChunkCreate, DocumentCreate, DocumentResponse, IndexingState

logger = logging.getLogger("obsidian-rag")


@dataclass(slots=True)
class VaultChanges:
    """Set of vault changes detected before indexing."""

    stale_document_ids: list[str]
    new_documents: list[DocumentCreate]


class Indexer:
    """Indexer for an Obsidian vault."""

    def __init__(
        self,
        embedder: Embedder,
        database: Database,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        batch_size: int | None = None,
        vault_path: Path | None = None,
    ) -> None:
        """Initializes the indexer and chunking parameters."""
        self.embedder = embedder
        self.database = database
        self.chunk_size = chunk_size or config.indexing.chunk_size
        self.chunk_overlap = chunk_overlap or config.indexing.chunk_overlap
        self.batch_size = batch_size or config.indexing.batch_size
        self.vault_path = vault_path or config.paths.vault_path

    def run(
        self,
        reindex: bool,
        state: IndexingState,
    ) -> None:
        """Runs vault indexing in the current thread."""
        conn = self.database.new_connection()
        document_repository = DocumentRepository(conn)
        chunk_repository = ChunkRepository(conn)

        try:
            self._mark_running(state)

            if reindex:
                self._set_message(state, "Full reindexing...")
                document_repository.delete_all()

            changes = self._load_changes(document_repository, state)
            self._delete_stale_documents(document_repository, changes, state)

            if not changes.new_documents:
                self._mark_done(state, "No new documents to index")
                return

            documents = self._create_documents(document_repository, changes, state)
            chunks = self._build_chunks(documents, state)
            chunk_repository.create_many(chunks)

            self._mark_done(
                state,
                (
                    f"Indexing finished: {len(documents)} documents, "
                    f"{len(chunks)} chunks"
                ),
            )

        except Exception as exc:
            logger.error("Indexing error: %s", exc, exc_info=True)
            state.status = "error"
            state.message = f"Indexing error: {exc}"
        finally:
            conn.close()
            logger.debug("Indexer connection closed")

    def _load_changes(
        self,
        document_repository: DocumentRepository,
        state: IndexingState,
    ) -> VaultChanges:
        """Scans the vault and returns stale and new documents."""
        self._set_message(state, "Loading documents...")

        existing_documents = {
            document.path: document
            for document in document_repository.get_all()
        }
        stale_document_ids: list[str] = []
        new_documents: list[DocumentCreate] = []

        markdown_files = list(self._iter_markdown_files())
        logger.info("Scanning vault: %s", self.vault_path)
        logger.info("Markdown files found: %s", len(markdown_files))

        for markdown_file in markdown_files:
            relative_path = str(markdown_file.relative_to(self.vault_path))
            existing = existing_documents.get(relative_path)

            if existing and not self._is_modified(markdown_file, existing.created_at):
                continue
            if existing:
                stale_document_ids.append(existing.document_id)

            text = self._read_markdown(markdown_file)
            if text:
                new_documents.append(DocumentCreate(path=relative_path, text=text))

        logger.info(
            "Stale documents: %s, new documents: %s",
            len(stale_document_ids),
            len(new_documents),
        )
        return VaultChanges(
            stale_document_ids=stale_document_ids,
            new_documents=new_documents,
        )

    def _iter_markdown_files(self):
        """Iterates vault markdown files while skipping hidden directories."""
        for markdown_file in self.vault_path.rglob("*.md"):
            if any(part.startswith(".") for part in markdown_file.parts):
                continue
            yield markdown_file

    @staticmethod
    def _is_modified(path: Path, indexed_at: datetime) -> bool:
        """Checks whether a file changed after it was indexed."""
        modified_at = datetime.fromtimestamp(path.stat().st_mtime)
        return indexed_at < modified_at

    @staticmethod
    def _read_markdown(path: Path) -> str:
        """Reads a markdown file and returns an empty string on failure."""
        try:
            return path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return ""

    def _delete_stale_documents(
        self,
        document_repository: DocumentRepository,
        changes: VaultChanges,
        state: IndexingState,
    ) -> None:
        """Deletes documents that will be replaced by a newer version."""
        count = len(changes.stale_document_ids)
        if count == 0:
            return

        self._set_message(state, f"Deleting {count} stale documents...")
        document_repository.delete_many(changes.stale_document_ids)

    def _create_documents(
        self,
        document_repository: DocumentRepository,
        changes: VaultChanges,
        state: IndexingState,
    ) -> list[DocumentResponse]:
        """Creates new documents in the database and returns their DTOs."""
        count = len(changes.new_documents)
        self._set_message(state, f"Creating {count} new documents...")
        return document_repository.create_many(changes.new_documents)

    def _build_chunks(
        self,
        documents: list[DocumentResponse],
        state: IndexingState,
    ) -> list[ChunkCreate]:
        """Splits documents, encodes chunks, and builds DTOs for persistence."""
        document_ids, chunk_texts = self._split_documents(documents)
        total_chunks = len(chunk_texts)

        logger.info("Embedding %s chunks in batches of %s...", total_chunks, self.batch_size)
        self._set_message(state, f"Embedding {total_chunks} chunks...")
        embeddings = self._encode_chunks(chunk_texts, state)

        self._set_message(state, f"Saving {total_chunks} chunks...")
        return [
            ChunkCreate(
                document_id=document_id,
                text=text,
                embedding=embedding,
            )
            for document_id, text, embedding in zip(document_ids, chunk_texts, embeddings)
        ]

    def _split_documents(
        self,
        documents: list[DocumentResponse],
    ) -> tuple[list[str], list[str]]:
        """Splits documents into chunks."""
        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        document_ids: list[str] = []
        chunk_texts: list[str] = []

        for document in documents:
            for chunk_text in splitter.split_text(document.text):
                document_ids.append(document.document_id)
                chunk_texts.append(chunk_text)

        logger.info(
            "Split %s documents into %s chunks",
            len(documents),
            len(chunk_texts),
        )
        return document_ids, chunk_texts

    def _encode_chunks(
        self,
        chunk_texts: list[str],
        state: IndexingState,
    ) -> list[list[float]]:
        """Gets embeddings for chunks in batches."""
        total = len(chunk_texts)
        chunk_embeddings: list[list[float]] = []

        for batch_index, texts in enumerate(chunked(chunk_texts, self.batch_size)):
            embeddings = self.embedder.embed_many(list(texts))
            chunk_embeddings.extend(embeddings.tolist())

            processed = min((batch_index + 1) * self.batch_size, total)
            state.message = f"Embedding chunks: {processed}/{total}"

            if batch_index % 10 == 0:
                logger.debug("Embedding progress: %s/%s", processed, total)

        logger.info("Embedding finished: %s vectors", len(chunk_embeddings))
        return chunk_embeddings

    @staticmethod
    def _mark_running(state: IndexingState) -> None:
        """Moves indexing state to running."""
        state.status = "running"
        state.started_at = datetime.now()
        state.message = ""

    @staticmethod
    def _mark_done(state: IndexingState, message: str) -> None:
        """Marks indexing as successfully finished."""
        state.status = "done"
        state.message = message
        logger.info(message)

    @staticmethod
    def _set_message(state: IndexingState, message: str) -> None:
        """Updates the user-facing progress message and logs it."""
        state.message = message
        logger.info(message)
