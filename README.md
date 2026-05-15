# Obsidian RAG Chat

Локальный RAG-чат для базы знаний Obsidian с гибридным поиском, реранкингом и стримингом ответов.

## Возможности

- **Чат с базой знаний** — задаёшь вопрос, получаешь ответ на основе заметок
- **Гибридный поиск** — BM25 + векторный поиск + Reciprocal Rank Fusion
- **Реранкинг** — BGE-reranker для точного отбора релевантных фрагментов
- **Стриминг ответов** — текст генерируется посимвольно, как в ChatGPT
- **История чатов** — сохраняется на диск, переживает перезапуски
- **Инкрементальная индексация** — только изменённые файлы, не пересчитывает всё
- **Поддержка Markdown** — ответы с форматированием, кодом, таблицами
- **Полностью локально** — все модели работают на твоём Mac

## Архитектура

```
Obsidian Vault (.md)
        │
        ▼
[Индексатор: чанки + BM25 + ChromaDB]
        │
        ▼
[FastAPI сервер]
  ├─ /stream — стриминг ответов с SSE
  ├─ /reindex — инкрементальная индексация
  ├─ /reindex/full — полная переиндексация
  ├─ /chats — управление историей чатов
  └─ /chat_ui.html — веб-интерфейс
        │
        ▼
[Ollama (нативно на Mac)]
  └─ LLM: T-lite-it-2.1 / qwen2.5 / gemma4
```

## Стек

| Компонент | Технология |
|-----------|------------|
| Сервер | FastAPI + uvicorn |
| Векторная БД | ChromaDB |
| Эмбеддинги | Sentence Transformers (multilingual-e5) |
| Гибридный поиск | BM25 (rank-bm25) + RRF |
| Реранкер | BGE-reranker-v2-m3 (CrossEncoder) |
| LLM | Ollama (T-lite-it-2.1) |
| Чанкинг | LangChain MarkdownTextSplitter |
| Управление зависимостями | Poetry |

## Быстрый старт

### 1. Установи Ollama (нативно на Mac)

```bash
brew install ollama
ollama serve  # в отдельном терминале
ollama pull t-tech/T-lite-it-2.1:q5_0
```

### 2. Клонируй репозиторий

```bash
git clone https://github.com/aapokroy/obsidian-rag.git
cd obsidian-rag
```

### 3. Настрой .env

```bash
echo "VAULT_PATH=/Users/you/ObsidianVault" > .env
```

### 4. Запусти

```bash
docker compose up -d
```

### 5. Открой в браузере

```
http://localhost:8000
```

## Конфигурация (config.yaml)

```yaml
vault_path: "/vault"
db_path: "/app/chroma_db"
embeddings_model: "intfloat/multilingual-e5-small"
reranker_model: "BAAI/bge-reranker-v2-m3"
llm_model: "t-tech/T-lite-it-2.1:q5_0"
ollama_url: "http://host.docker.internal:11434"
chunk_size: 600
chunk_overlap: 100
top_k_retrieval: 50
top_k_final: 12
min_relevance: 0.01
bm25_weight: 0.3
rrf_k: 60
```

## Эндпоинты API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Веб-интерфейс |
| POST | `/stream` | Стриминг ответа с SSE |
| GET | `/chats` | Список чатов |
| POST | `/chat/new` | Новый чат |
| GET | `/chat/{id}` | История чата |
| DELETE | `/chat/{id}` | Удалить чат |
| POST | `/reindex` | Инкрементальная индексация |
| POST | `/reindex/full` | Полная переиндексация |
| GET | `/reindex/stream` | SSE-поток прогресса индексации |

## Выбор моделей

### LLM (через Ollama)

| Модель | Размер | Скорость | Качество русского |
|--------|--------|----------|-------------------|
| `t-tech/T-lite-it-2.1:q5_0` | ~5.7 GB | ⚡⚡⚡ | ✅ Отличное |
| `qwen2.5:14b` | ~8.9 GB | ⚡⚡ | ✅ Отличное |
| `gemma4:e4b` | ~5 GB | ⚡⚡⚡ | ✅ Хорошее |

### Эмбеддинги

| Модель | Размер | Контекст | Рекомендация |
|--------|--------|----------|--------------|
| `multilingual-e5-small` | ~470 MB | 512 токенов | Быстро, хорошее качество |
| `multilingual-e5-large` | ~2.2 GB | 512 токенов | Точнее, но медленнее |
| `BAAI/bge-m3` | ~2.2 GB | 8192 токенов | Длинные документы |

### Реранкер

| Модель | Скорость | Качество |
|--------|----------|----------|
| `BAAI/bge-reranker-v2-m3` | 3-5 сек | Отличное |
| `ms-marco-MultiBERT-L-12` (FlashRank) | 1-2 сек | Среднее для русского |

## Разработка

```bash
# Установка зависимостей
poetry install

# Генерация lock-файла
poetry lock

# Локальный запуск (без Docker)
poetry shell
python server.py
```