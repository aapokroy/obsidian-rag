"""Prompt context and source assembly for chat answers."""

from openai.types.chat import ChatCompletionMessageParam

from lib.config import config
from lib.schema import MessageResponse, RerankResultItem, SourceResponse


SYSTEM_PROMPT = (
    "Ты - ассистент, который отвечает строго на основе фрагментов базы знаний. "
    "Не ссылайся на источники в формате 'В источнике [1: ...]', просто пиши текст без ссылок. "
    "Ответ оформляй в формате markdown. "
    "Всегда отвечай на русском языке. "
    "Тебе будет предоставлено несколько источников, информация в них не всегда полностью релевантна запросу, "
    "не включай в ответ лишнюю информацию."
)


def filter_sources(results: list[RerankResultItem]) -> list[SourceResponse]:
    """Keeps sources whose relevance passes the configured threshold."""
    return [
        SourceResponse(
            document_path=result.document_path,
            text=result.text,
            relevance=result.relevance,
        )
        for result in results
        if result.relevance >= config.search.min_relevance
    ]


def build_context(
    messages: list[MessageResponse],
    query: str,
    reranked: list[RerankResultItem],
) -> list[ChatCompletionMessageParam]:
    """Builds the system prompt, chat history, and retrieved context for the LLM."""
    chat_messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
    ]
    chat_messages.extend(_history_messages(messages))
    chat_messages.append(_context_message(query=query, reranked=reranked))
    return chat_messages


def _history_messages(messages: list[MessageResponse]) -> list[ChatCompletionMessageParam]:
    """Converts recent chat messages to the OpenAI Chat API format."""
    return [
        {
            "role": message.role,
            "content": message.text,
        }
        for message in messages[-config.chat.context_limit:]
    ]


def _context_message(
    query: str,
    reranked: list[RerankResultItem],
) -> ChatCompletionMessageParam:
    """Builds the user message with RAG context and the original query."""
    context_text = "\n\n".join(
        f"Source ({result.document_path}):\n{result.text}"
        for result in reranked
    )
    return {
        "role": "user",
        "content": (
            f"Knowledge base context:\n{context_text}\n\n"
            f"User query: {query}"
        ),
    }
