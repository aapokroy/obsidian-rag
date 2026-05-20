FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VERSION="2.4.1" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR="/tmp/poetry_cache"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock README.md ./
RUN --mount=type=cache,target=/tmp/poetry_cache \
    poetry install --only main --no-interaction --no-ansi --no-root

COPY lib/ ./lib/
COPY server.py chat_ui.html config.yaml ./

EXPOSE 8000

CMD ["python", "server.py"]
