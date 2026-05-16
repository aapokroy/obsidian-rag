"""Сборка промпта для LLM."""


def build_prompt_with_history(question: str, chunks: list[dict], history: list[dict]) -> list[dict]:
    """Сборка messages с RAG-контекстом и историей."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Источник {i}: {chunk['title']} ({chunk['source']})]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    system_prompt = (
        "Ты — ассистент, который отвечает строго на основе фрагментов базы знаний. "
        "Не ссылайся на источники в формате 'В источнике [1: ...]', просто пиши текст без ссылок. "
        "Ответ оформляй в формате markdown. "
        "Всегда отвечай на русском языке. "
        "Тебе будет предоставлено несколько источников, информация в них не всегда полностью релевантна запросу, "
        "не включай в ответ лишнюю информацию."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    user_message = (
        f"Вопрос: {question}\n\n"
        f"Фрагменты базы знаний:\n{context}\n\n"
        f"Ответь строго по фрагментам."
    )
    messages.append({"role": "user", "content": user_message})
    
    return messages