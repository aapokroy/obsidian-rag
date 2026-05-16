FROM python:3.11-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR="/tmp/poetry_cache"
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Зависимости
COPY pyproject.toml poetry.lock ./
RUN --mount=type=cache,target=/tmp/poetry_cache \
    /usr/local/bin/poetry install --no-interaction --no-ansi --no-root

# Код приложения
COPY lib/ ./lib/
COPY server.py chat_ui.html config.yaml ./

EXPOSE 8000

CMD ["python", "server.py"]
