FROM python:3.11-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR="/tmp/poetry_cache"
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Копируем файлы зависимостей
COPY pyproject.toml poetry.lock ./

# Установка зависимостей
RUN --mount=type=cache,target=/tmp/poetry_cache,uid=1000,gid=1000,mode=0755 \
    poetry install --no-interaction --no-ansi --no-root

# Копируем код приложения
COPY lib/ ./lib/
COPY server.py ./
COPY chat_ui.html ./
COPY config.yaml ./

# Порт для FastAPI
EXPOSE 8000

CMD ["python", "server.py"]