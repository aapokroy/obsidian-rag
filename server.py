#!/usr/bin/env python3
"""FastAPI сервер Obsidian RAG."""

import asyncio
import json
import logging
import threading
import warnings
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from lib.config import config
from lib.db import (
    Database,
    ChunkRepository,
    DocumentRepository,
    ChatRepository,
    MessageRepository,
)
from lib.embedder import Embedder
from lib.indexer import Indexer
from lib.reranker import Reranker
from lib.retriever import Retriever
from lib.schema import (
    ChatResponse,
    IndexingState,
    MessageCreate,
    MessageResponse,
    RerankResultItem,
    QueryRequest,
)

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*telemetry.*")

# ============================================================
# Логирование
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("obsidian-rag")

# ============================================================
# Инициализация компонентов
# ============================================================

indexing_state = IndexingState()

logger.info("=" * 50)
logger.info("🚀 Запуск Obsidian RAG сервера...")
logger.info("=" * 50)
logger.info(f"📋 vault_path: {config.paths.vault_path}")
logger.info(f"📋 db_path: {config.paths.db_path}")
logger.info(f"📋 embeddings_dim: {config.embeddings.dimension}")
logger.info(f"📋 llm_url: {config.urls.llm_url}")
logger.info(f"📋 embeddings_url: {config.urls.embeddings_url}")
logger.info(f"📋 reranker_url: {config.urls.reranker_url}")

logger.info("Подключение к БД...")
database = Database()
document_repository = DocumentRepository(database.conn)
chunk_repository = ChunkRepository(database.conn)
chat_repository = ChatRepository(database.conn)
message_repository = MessageRepository(database.conn, chat_repository)
logger.info("БД готова")

embedder = Embedder()
indexer = Indexer(
    database=database,
    embedder=embedder,
)
retriever = Retriever(db=database, embedder=embedder)
reranker = Reranker()

logger.info("Подключение к LLM API...")
llm_client = OpenAI(
    base_url=config.urls.llm_url,
    api_key="lm-studio",
)
try:
    models = llm_client.models.list()
    logger.info(f"Доступные модели: {[m.id for m in models.data]}")
except Exception as e:
    logger.warning(f"Не удалось получить список моделей LLM: {e}")

# Проверка количества чанков в БД
try:
    count_row = database.conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
    logger.info(f"Чанков в БД: {count_row['cnt']}")
except Exception as e:
    logger.warning(f"Не удалось проверить количество чанков: {e}")


class Source:
    """Источник информации для ответа."""
    def __init__(
        self,
        document_path: str,
        text: str,
        relevance: float,
    ) -> None:
        self.document_path = document_path
        self.text = text
        self.relevance = relevance

    def dict(self) -> dict:
        return {
            "document_path": self.document_path,
            "text": self.text,
            "relevance": self.relevance,
        }


# ============================================================
# FastAPI приложение
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("✅ Сервер готов!")
    yield
    logger.info("Закрытие соединения с БД...")
    database.close()


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


# ============================================================
# Эндпоинты чатов
# ============================================================

@app.get("/")
async def root():
    """Отдаёт веб-интерфейс."""
    return FileResponse("chat_ui.html")


@app.get("/chats", response_model=list[ChatResponse])
async def list_chats():
    """Возвращает список всех чатов."""
    chat_list = chat_repository.get_all()
    logger.debug(f"Чатов в списке: {len(chat_list)}")
    return chat_list


@app.post("/chat/new", response_model=ChatResponse)
async def create_chat():
    """Создаёт новый чат."""
    chat = chat_repository.create()
    logger.info(f"Новый чат: {chat.chat_id}")
    return chat


@app.get("/chat/{chat_id}", response_model=ChatResponse)
async def get_chat(chat_id: str):
    """Возвращает информацию о чате."""
    chat_list = chat_repository.get_all()
    for chat in chat_list:
        if chat.chat_id == chat_id:
            return chat
    logger.warning(f"Чат не найден: {chat_id}")
    raise HTTPException(status_code=404, detail="Чат не найден")


@app.get("/chat/{chat_id}/messages", response_model=list[MessageResponse])
async def get_chat_messages(chat_id: str):
    """Возвращает сообщения чата."""
    messages = message_repository.get_by_chat(chat_id)
    logger.debug(f"Сообщений в чате {chat_id}: {len(messages)}")
    return messages


@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    """Удаляет чат со всеми сообщениями."""
    chat_repository.delete(chat_id)
    logger.info(f"Чат удалён: {chat_id}")
    return {"status": "ok"}


# ============================================================
# Эндпоинт стриминга
# ============================================================

# server.py — фрагмент эндпоинта /stream

@app.post("/stream")
async def stream_query(request: QueryRequest):
    """Генерирует ответ с потоковой передачей токенов."""
    query = request.query.strip()
    chat_id = request.chat_id

    logger.info(f"Запрос [{chat_id}]: {query[:100]}...")

    if not query:
        logger.warning("Пустой запрос")
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    # Проверяем существование чата
    chat_list = chat_repository.get_all()
    chat_exists = any(c.chat_id == chat_id for c in chat_list)

    if not chat_exists:
        logger.warning(f"Чат не найден: {chat_id}")
        raise HTTPException(status_code=404, detail="Чат не найден")

    # Загружаем историю (может быть пустой)
    messages = message_repository.get_by_chat(chat_id)
    logger.debug(f"История чата: {len(messages)} сообщений")

    # Получаем эмбеддинг запроса
    logger.debug("Получение эмбеддинга запроса...")
    try:
        query_embedding = embedder.embed(query)
        logger.debug(f"Эмбеддинг получен, размерность: {query_embedding.shape}")
    except Exception as e:
        logger.error(f"Ошибка получения эмбеддинга: {e}")
        async def _embedding_error():
            yield f"data: {json.dumps({'type': 'error', 'content': f'Ошибка эмбеддинга: {e}'})}\n\n"
        return StreamingResponse(_embedding_error(), media_type="text/event-stream")

    # Гибридный поиск
    logger.debug("Гибридный поиск...")
    try:
        retrieve_results = retriever.hybrid_search(query)
        logger.info(f"Найдено результатов: {len(retrieve_results)}")
        if retrieve_results:
            rrf_scores = [r.rrf_score for r in retrieve_results]
            logger.info(
                f"Retriever RRF scores: min={min(rrf_scores):.5f}, "
                f"max={max(rrf_scores):.5f}, "
                f"mean={sum(rrf_scores)/len(rrf_scores):.5f}",
            )
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        async def _search_error():
            yield f"data: {json.dumps({'type': 'error', 'content': f'Ошибка поиска: {e}'})}\n\n"
        return StreamingResponse(_search_error(), media_type="text/event-stream")

    if not retrieve_results:
        logger.warning("Ничего не найдено в базе знаний")
        async def _empty():
            yield f"data: {json.dumps({'type': 'error', 'content': 'Нет информации в базе знаний.'})}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    # Реранкинг
    logger.debug("Реранкинг...")
    reranked = reranker.rerank(
        query=query,
        retrieve_results=retrieve_results,
    )
    logger.info(f"После реранкинга: {len(reranked)} результатов")
    if reranked:
        relevance_scores = [r.relevance for r in reranked]
        logger.info(
            f"Reranker scores: min={min(relevance_scores):.4f}, "
            f"max={max(relevance_scores):.4f}, "
            f"mean={sum(relevance_scores)/len(relevance_scores):.4f}",
        )
        logger.debug(f"Топ-3 релевантность: {[f'{s:.4f}' for s in relevance_scores[:3]]}")

    # Формируем источники с учётом порога релевантности
    sources = [
        Source(
            document_path=r.document_path,
            text=r.text,
            relevance=r.relevance,
        )
        for r in reranked
        if r.relevance >= config.search.min_relevance
    ]
    logger.info(
        f"Источников после фильтрации "
        f"(min_relevance={config.search.min_relevance}): {len(sources)}",
    )

    # Сохраняем сообщение пользователя
    logger.debug("Сохранение сообщения пользователя...")
    message_repository.create(
        chat_id=chat_id,
        data=MessageCreate(
            text=query,
            role="user",
        ),
    )

    # Формируем контекст для LLM
    context = build_context(
        messages=messages,
        query=query,
        reranked=reranked,
    )
    logger.debug(f"Контекст сформирован: {len(context)} сообщений для LLM")

    return StreamingResponse(
        _generate_stream(
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
# ============================================================
# Эндпоинты индексации
# ============================================================

@app.post("/reindex")
async def reindex():
    """Запускает инкрементальную индексацию."""
    logger.info("Запуск инкрементальной индексации")
    return _start_reindex(reindex=False)


@app.post("/reindex/full")
async def reindex_full():
    """Запускает полную переиндексацию."""
    logger.info("Запуск ПОЛНОЙ переиндексации")
    return _start_reindex(reindex=True)


def _start_reindex(reindex: bool) -> dict:
    """Запускает индексацию в фоновом потоке."""
    if indexing_state.status == "running":
        logger.warning("Индексация уже выполняется")
        return {"status": "already_running"}

    def _job() -> None:
        logger.info(f"Фоновая индексация запущена (reindex={reindex})")
        indexer.run(reindex=reindex, state=indexing_state)
        logger.info(f"Фоновая индексация завершена: {indexing_state.status}")

    threading.Thread(target=_job, daemon=True).start()
    return {"status": "started"}


@app.get("/reindex/status")
async def reindex_status():
    """Возвращает текущий статус индексации."""
    return {
        "status": indexing_state.status,
        "message": indexing_state.message,
        "started_at": indexing_state.started_at.isoformat() if indexing_state.started_at else None,
    }


@app.get("/reindex/stream")
async def reindex_stream():
    """SSE-поток с прогрессом индексации."""
    async def _event_stream():
        last_message = ""
        while True:
            current_message = indexing_state.message
            current_status = indexing_state.status

            if current_message != last_message:
                yield f"data: {json.dumps({'status': current_status, 'message': current_message})}\n\n"
                last_message = current_message

            if current_status in ("done", "error"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================================
# Вспомогательные функции
# ============================================================

def build_context(
    messages: list[MessageResponse],
    query: str,
    reranked: list[RerankResultItem],
) -> list[ChatCompletionMessageParam]:
    """Формирует сообщения для LLM с контекстом из найденных чанков."""
    context_text = "\n\n".join(
        f"Источник ({r.document_path}):\n{r.text}"
        for r in reranked
    )

    system_prompt = (
        "Ты — ассистент, который отвечает строго на основе фрагментов базы знаний. "
        "Не ссылайся на источники в формате 'В источнике [1: ...]', просто пиши текст без ссылок. "
        "Ответ оформляй в формате markdown. "
        "Всегда отвечай на русском языке. "
        "Тебе будет предоставлено несколько источников, информация в них не всегда полностью релевантна запросу, "
        "не включай в ответ лишнюю информацию."
    )

    system_message: ChatCompletionMessageParam = {
        "role": "system",
        "content": system_prompt,
    }

    chat_messages: list[ChatCompletionMessageParam] = [system_message]

    # Добавляем историю (последние N сообщений)
    recent = messages[-config.chat.context_limit:]
    for msg in recent:
        chat_messages.append({
            "role": msg.role,
            "content": msg.text,
        })

    # Добавляем текущий запрос с контекстом
    user_message = (
        f"Контекст из базы знаний:\n{context_text}\n\n"
        f"Запрос пользователя: {query}"
    )
    chat_messages.append({
        "role": "user",
        "content": user_message,
    })

    return chat_messages


def _generate_stream(
    messages: list[ChatCompletionMessageParam],
    model: str,
    sources: list[Source],
    chat_id: str,
):
    """Генератор для потоковой передачи ответа LLM."""
    full_response = ""

    try:
        logger.info(f"Отправка запроса к LLM (модель: {model}, сообщений: {len(messages)})")
        stream = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.llm.temperature,
            stream=True,
            max_tokens=config.llm.max_tokens,
        )

        token_count = 0
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                token_count += 1
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        logger.info(f"LLM ответ получен: {token_count} токенов, {len(full_response)} символов")

        # Сохраняем ответ ассистента — используем отдельное соединение
        if full_response:
            _save_assistant_message(chat_id, full_response, sources)
            logger.debug("Ответ ассистента сохранён")

        # Отправляем источники
        if sources:
            logger.debug(f"Отправка {len(sources)} источников")
            yield f"data: {json.dumps({'type': 'sources', 'content': [s.dict() for s in sources]})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


def _save_assistant_message(
    chat_id: str,
    text: str,
    sources: list[Source],
) -> None:
    """Сохраняет ответ ассистента в БД через новое соединение."""
    try:
        # Создаём новое соединение для этого потока
        conn = database.new_connection()
        chat_repo = ChatRepository(conn)
        msg_repo = MessageRepository(conn, chat_repo)

        msg_repo.create(
            chat_id=chat_id,
            data=MessageCreate(
                text=text,
                role="assistant",
            ),
        )
        conn.close()
        logger.debug("Ответ ассистента сохранён через отдельное соединение")
    except Exception as e:
        logger.error(f"Ошибка сохранения ответа ассистента: {e}", exc_info=True)


# ============================================================
# Точка входа
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
