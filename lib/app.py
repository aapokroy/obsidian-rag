"""FastAPI application for Obsidian RAG."""

import asyncio
import json
import threading
import warnings
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from lib.config import config
from lib.db import (
    ChatRepository,
    ChunkRepository,
    Database,
    DocumentRepository,
    MessageRepository,
)
from lib.embedder import Embedder
from lib.indexer import Indexer
from lib.logging_config import configure_logging
from lib.prompting import build_context, filter_sources
from lib.reranker import Reranker
from lib.retriever import Retriever
from lib.schema import (
    ChatResponse,
    IndexingState,
    MessageCreate,
    MessageResponse,
    QueryRequest,
    RerankResultItem,
    RetrieveResultItem,
    SourceResponse,
)

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*telemetry.*")

logger = configure_logging()


@dataclass(slots=True)
class AppServices:
    """Container for application dependencies."""

    database: Database
    document_repository: DocumentRepository
    chunk_repository: ChunkRepository
    chat_repository: ChatRepository
    message_repository: MessageRepository
    embedder: Embedder
    indexer: Indexer
    retriever: Retriever
    reranker: Reranker
    llm_client: OpenAI
    indexing_state: IndexingState


def create_services() -> AppServices:
    """Initializes external services and repositories."""
    logger.info("=" * 50)
    logger.info("Starting Obsidian RAG server...")
    logger.info("=" * 50)
    logger.info("vault_path: %s", config.paths.vault_path)
    logger.info("db_path: %s", config.paths.db_path)
    logger.info("embeddings_dim: %s", config.embeddings.dimension)
    logger.info("llm_url: %s", config.urls.llm_url)
    logger.info("embeddings_url: %s", config.urls.embeddings_url)
    logger.info("reranker_url: %s", config.urls.reranker_url)

    database = Database()
    chat_repository = ChatRepository(database.conn)
    message_repository = MessageRepository(database.conn, chat_repository)
    document_repository = DocumentRepository(database.conn)
    chunk_repository = ChunkRepository(database.conn)

    embedder = Embedder()
    indexer = Indexer(database=database, embedder=embedder)
    retriever = Retriever(db=database, embedder=embedder)
    reranker = Reranker()

    llm_client = OpenAI(
        base_url=config.urls.llm_url,
        api_key="lm-studio",
    )
    _log_llm_models(llm_client)
    _log_chunk_count(database)

    return AppServices(
        database=database,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        chat_repository=chat_repository,
        message_repository=message_repository,
        embedder=embedder,
        indexer=indexer,
        retriever=retriever,
        reranker=reranker,
        llm_client=llm_client,
        indexing_state=IndexingState(),
    )


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    services = create_services()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manages application lifespan and database shutdown."""
        logger.info("Server is ready")
        yield
        logger.info("Closing database connection...")
        services.database.close()

    app = FastAPI(
        title="Obsidian RAG",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        """Serves the web UI."""
        return FileResponse("chat_ui.html")

    @app.get("/chats", response_model=list[ChatResponse])
    async def list_chats():
        """Returns all chats."""
        return services.chat_repository.get_all()

    @app.post("/chat/new", response_model=ChatResponse)
    async def create_chat():
        """Creates a new chat."""
        chat = services.chat_repository.create()
        logger.info("New chat: %s", chat.chat_id)
        return chat

    @app.get("/chat/{chat_id}", response_model=ChatResponse)
    async def get_chat(chat_id: str):
        """Returns chat metadata."""
        chat = services.chat_repository.get_by_id(chat_id)
        if chat is None:
            logger.warning("Chat not found: %s", chat_id)
            raise HTTPException(status_code=404, detail="Chat not found")
        return chat

    @app.get("/chat/{chat_id}/messages", response_model=list[MessageResponse])
    async def get_chat_messages(chat_id: str):
        """Returns chat messages."""
        _ensure_chat_exists(services, chat_id)
        return services.message_repository.get_by_chat(chat_id)

    @app.delete("/chat/{chat_id}")
    async def delete_chat(chat_id: str):
        """Deletes a chat and all of its messages."""
        _ensure_chat_exists(services, chat_id)
        services.chat_repository.delete(chat_id)
        logger.info("Chat deleted: %s", chat_id)
        return {"status": "ok"}

    @app.post("/stream")
    async def stream_query(request: QueryRequest):
        """Generates an answer as a token stream."""
        return _stream_query(request, services)

    @app.post("/reindex")
    async def reindex():
        """Starts incremental indexing."""
        logger.info("Starting incremental indexing")
        return _start_reindex(services, reindex=False)

    @app.post("/reindex/full")
    async def reindex_full():
        """Starts full reindexing."""
        logger.info("Starting full reindexing")
        return _start_reindex(services, reindex=True)

    @app.get("/reindex/status")
    async def reindex_status():
        """Returns current indexing status."""
        state = services.indexing_state
        return {
            "status": state.status,
            "message": state.message,
            "started_at": state.started_at.isoformat() if state.started_at else None,
        }

    @app.get("/reindex/stream")
    async def reindex_stream():
        """SSE stream with indexing progress."""
        return StreamingResponse(
            _reindex_events(services.indexing_state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return app


def _stream_query(
    request: QueryRequest,
    services: AppServices,
) -> StreamingResponse:
    """Handles a user query and returns the answer SSE stream."""
    query = request.query.strip()
    chat_id = request.chat_id

    logger.info("Query [%s]: %s...", chat_id, query[:100])

    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    _ensure_chat_exists(services, chat_id)
    messages = services.message_repository.get_by_chat(chat_id)

    try:
        retrieve_results = services.retriever.hybrid_search(query)
    except Exception as exc:
        logger.error("Search error: %s", exc, exc_info=True)
        return _error_stream(f"Search error: {exc}")

    if not retrieve_results:
        logger.warning("No information found in the knowledge base")
        return _error_stream("No information in the knowledge base.")

    logger.info("Retrieved results: %s", len(retrieve_results))
    _log_retrieval_stats(retrieve_results)

    try:
        reranked = services.reranker.rerank(
            query=query,
            retrieve_results=retrieve_results,
        )
    except Exception as exc:
        logger.error("Reranking error: %s", exc, exc_info=True)
        return _error_stream(f"Reranking error: {exc}")

    logger.info("After reranking: %s results", len(reranked))
    _log_rerank_stats(reranked)

    sources = filter_sources(reranked)
    logger.info(
        "Sources after filtering (min_relevance=%s): %s",
        config.search.min_relevance,
        len(sources),
    )

    services.message_repository.create(
        chat_id=chat_id,
        data=MessageCreate(text=query, role="user"),
    )
    context = build_context(
        messages=messages,
        query=query,
        reranked=reranked,
    )

    return StreamingResponse(
        _generate_stream(
            services=services,
            messages=context,
            model=config.models.llm_model,
            sources=sources,
            chat_id=chat_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _start_reindex(
    services: AppServices,
    reindex: bool,
) -> dict[str, str]:
    """Starts indexing in a background thread."""
    state = services.indexing_state
    if state.status == "running":
        logger.warning("Indexing is already running")
        return {"status": "already_running"}

    def _job() -> None:
        """Runs indexing inside a daemon thread."""
        logger.info("Background indexing started (reindex=%s)", reindex)
        services.indexer.run(reindex=reindex, state=state)
        logger.info("Background indexing finished: %s", state.status)

    threading.Thread(target=_job, daemon=True).start()
    return {"status": "started"}


async def _reindex_events(state: IndexingState):
    """Generates SSE events with indexing status."""
    last_message = ""
    while True:
        current_message = state.message
        current_status = state.status

        if current_message != last_message:
            yield _sse(
                {
                    "status": current_status,
                    "message": current_message,
                },
            )
            last_message = current_message

        if current_status in ("done", "error"):
            break

        await asyncio.sleep(0.5)


def _generate_stream(
    services: AppServices,
    messages: list[ChatCompletionMessageParam],
    model: str,
    sources: list[SourceResponse],
    chat_id: str,
) -> Iterator[str]:
    """Generator for streaming the LLM answer."""
    full_response = ""

    try:
        logger.info(
            "Sending request to LLM (model: %s, messages: %s)",
            model,
            len(messages),
        )
        stream = services.llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.llm.temperature,
            stream=True,
            max_tokens=config.llm.max_tokens,
        )

        token_count = 0
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                full_response += token
                token_count += 1
                yield _sse({"type": "token", "content": token})

        logger.info(
            "LLM answer received: %s tokens, %s characters",
            token_count,
            len(full_response),
        )

        if full_response:
            _save_assistant_message(services, chat_id, full_response)

        if sources:
            yield _sse(
                {
                    "type": "sources",
                    "content": [source.model_dump() for source in sources],
                },
            )

        yield _sse({"type": "done", "content": ""})

    except Exception as exc:
        logger.error("Answer generation error: %s", exc, exc_info=True)
        yield _sse({"type": "error", "content": str(exc)})


def _save_assistant_message(
    services: AppServices,
    chat_id: str,
    text: str,
) -> None:
    """Saves the assistant answer through a separate connection."""
    conn = services.database.new_connection()
    try:
        chat_repository = ChatRepository(conn)
        message_repository = MessageRepository(conn, chat_repository)
        message_repository.create(
            chat_id=chat_id,
            data=MessageCreate(text=text, role="assistant"),
        )
        logger.debug("Assistant answer saved")
    except Exception as exc:
        logger.error("Failed to save assistant answer: %s", exc, exc_info=True)
    finally:
        conn.close()


def _ensure_chat_exists(
    services: AppServices,
    chat_id: str,
) -> None:
    """Checks chat existence and raises HTTP 404 if it is missing."""
    if not services.chat_repository.exists(chat_id):
        logger.warning("Chat not found: %s", chat_id)
        raise HTTPException(status_code=404, detail="Chat not found")


def _error_stream(message: str) -> StreamingResponse:
    """Creates an SSE response with a single error event."""
    async def _events():
        """Sends a single SSE error event."""
        yield _sse({"type": "error", "content": message})

    return StreamingResponse(_events(), media_type="text/event-stream")


def _sse(payload: dict) -> str:
    """Serializes a payload into Server-Sent Events format."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _log_llm_models(llm_client: OpenAI) -> None:
    """Logs the list of models available through the LLM API."""
    try:
        models = llm_client.models.list()
        logger.info("Available models: %s", [model.id for model in models.data])
    except Exception as exc:
        logger.warning("Failed to fetch LLM model list: %s", exc)


def _log_chunk_count(database: Database) -> None:
    """Logs the current number of chunks in the database."""
    try:
        row = database.conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        logger.info("Chunks in database: %s", row["cnt"])
    except Exception as exc:
        logger.warning("Failed to check chunk count: %s", exc)


def _log_retrieval_stats(results: list[RetrieveResultItem]) -> None:
    """Logs aggregate RRF-score statistics for initial retrieval."""
    if not results:
        return

    scores = [result.rrf_score for result in results]
    logger.info(
        "Retriever RRF scores: min=%.5f, max=%.5f, mean=%.5f",
        min(scores),
        max(scores),
        sum(scores) / len(scores),
    )


def _log_rerank_stats(results: list[RerankResultItem]) -> None:
    """Logs aggregate relevance-score statistics after reranking."""
    if not results:
        return

    scores = [result.relevance for result in results]
    logger.info(
        "Reranker scores: min=%.4f, max=%.4f, mean=%.4f",
        min(scores),
        max(scores),
        sum(scores) / len(scores),
    )
