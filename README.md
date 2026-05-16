# Obsidian RAG Chat

Локальный RAG-чат для базы знаний Obsidian с гибридным поиском, реранкингом и стримингом ответов.

> ⚠️ **Vibe-coded проект.** Этот код почти полностью написан через общение с AI.

## Возможности

- **Чат с базой знаний Obsidian** — задаёшь вопрос, получаешь ответ на основе заметок
- **Гибридный поиск** — BM25 + векторный поиск + Reciprocal Rank Fusion
- **Реранкинг** — отдельный микро-сервер на Nemotron 1B с MPS-ускорением
- **Стриминг ответов** — текст генерируется посимвольно, как в ChatGPT
- **История чатов** — сохраняется на диск, переживает перезапуски
- **Инкрементальная индексация** — только изменённые файлы, не пересчитывает всё
- **Поддержка Markdown** — ответы с форматированием, кодом, таблицами
- **Полностью локально** — LLM, эмбеддинги и реранкер работают на твоём Mac

## Архитектура

```
┌─────────────────────────────────────────────┐
│  Mac (хост)                                 │
│                                             │
│  ┌──────────────┐  ┌─────────────────────┐  │
│  │  LM Studio   │  │  rerank-server      │  │
│  │  LLM + Emb.  │  │  (Nemotron, MPS)    │  │
│  │  :1234       │  │  :8001              │  │
│  └──────┬───────┘  └──────────┬──────────┘  │
│         │                     │             │
│  ┌──────┴─────────────────────┴──────────┐  │
│  │         Docker                         │  │
│  │  ┌──────────────────────────────────┐  │  │
│  │  │  FastAPI server                  │  │  │
│  │  │  ChromaDB + BM25                 │  │  │
│  │  │  :8000                           │  │  │
│  │  └──────────────────────────────────┘  │  │
│  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Стек (2026)

| Компонент | Технология | Модель |
|-----------|------------|--------|
| LLM | LM Studio (OpenAI API) | `t-lite-it-2.1` (8B, Q5_K_M) |
| Эмбеддинги | LM Studio API | `text-embedding-user-bge-m3` |
| Реранкер | FastAPI + Transformers (MPS) | `nvidia/Llama-Nemotron-Rerank-1B-v2` |
| Векторная БД | ChromaDB | — |
| Гибридный поиск | BM25 + RRF | — |
| Чанкинг | LangChain MarkdownTextSplitter | — |
| Сервер | FastAPI + uvicorn | — |
| Зависимости | Poetry | — |

## Быстрый старт

### 1. Установи LM Studio

Скачай с [lmstudio.ai](https://lmstudio.ai), установи как обычное приложение Mac.

### 2. Скачай модели в LM Studio

- **LLM**: найди `t-lite-it-2.1` → выбери `Q5_K_M` (~5.9 ГБ)
- **Embeddings**: найди `text-embedding-user-bge-m3` → выбери `Q4_K_M` (~2.2 ГБ)

### 3. Запусти серверы LM Studio

Вкладка **Developer** (</>):
- Выбери LLM-модель → **GPU Offload: Max** → **Context Length: 8192** → **Start Server**
- Embeddings-модель загрузится автоматически при запросе

### 4. Запусти реранк-сервер

```bash
cd rerank-server
source venv/bin/activate
pip install -r requirements.txt
python rerank_server.py
```

### 5. Настрой .env

```bash
echo "VAULT_PATH=/Users/you/ObsidianVault" > .env
```

### 6. Запусти RAG-сервер

```bash
docker compose up -d
```

### 7. Открой в браузере

```
http://localhost:8000
```

## Конфигурация (config.yaml)

```yaml
# Пути
vault_path: "/vault"
db_path: "/app/chroma_db"

# URL-ы сервисов
llm_url: "http://host.docker.internal:1234/v1"
embeddings_url: "http://host.docker.internal:1234/v1/embeddings"
reranker_url: "http://host.docker.internal:8001/rerank"

# Модели
llm_model: "t-lite-it-2.1"
embeddings_model: "text-embedding-user-bge-m3"
reranker_model: "nvidia/Llama-Nemotron-Rerank-1B-v2"

# Индексация
chunk_size: 600
chunk_overlap: 100

# Поиск
top_k_retrieval: 50
top_k_rerank: 12
min_relevance: 0.01
bm25_weight: 0.3
rrf_k: 60
```

## Структура проекта

```
obsidian-rag/
├── server.py              # Точка входа
├── lib/                   # Библиотека
│   ├── config.py          # Pydantic-конфиг
│   ├── chat_store.py      # Хранение чатов
│   ├── embeddings.py      # Эмбеддинги через API
│   ├── reranker.py        # Реранкинг через API
│   ├── retriever.py       # Гибридный поиск
│   ├── indexer.py         # Индексация
│   └── prompt.py          # Сборка промпта
├── rerank-server/         # Микро-сервер реранкера
│   └── rerank_server.py
├── config.yaml
├── chat_ui.html
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## Выбор моделей (альтернативы)

### LLM

| Модель | Размер | Скорость | Русский |
|--------|--------|----------|---------|
| `t-lite-it-2.1` (Q5_K_M) | ~5.9 ГБ | ⚡⚡⚡ | ✅ Отличное |
| `qwen2.5:14b` | ~8.9 ГБ | ⚡⚡ | ✅ Отличное |
| `gemma4:e4b` | ~5 ГБ | ⚡⚡⚡ | ✅ Хорошее |

### Эмбеддинги

| Модель | Размер | Контекст |
|--------|--------|----------|
| `text-embedding-user-bge-m3` | ~2.2 ГБ | 8192 токенов |
| `multilingual-e5-large` | ~2.1 ГБ | 514 токенов |
| `enbeddrus` | ~0.4 ГБ | 512 токенов |

### Реранкер

| Модель | Размер | Скорость (MPS) |
|--------|--------|----------------|
| `nvidia/Llama-Nemotron-Rerank-1B-v2` | ~2.2 ГБ | ~0.1 сек/чанк |
| `BAAI/bge-reranker-v2-m3` | ~1.2 ГБ | ~0.05 сек/чанк |

## Разработка

```bash
# Установка зависимостей
poetry install

# Генерация lock-файла
poetry lock

# Сборка Docker
docker compose build

# Запуск
docker compose up -d
```