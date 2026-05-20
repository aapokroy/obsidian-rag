"""Application configuration and YAML loading."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Paths to the vault, runtime data, and SQLite database."""

    vault_path: Path = Path("/vault")
    data_path: Path = Path("/app/data")
    db_path: Path = Path("/app/data/app.db")


class URLsConfig(BaseModel):
    """URLs of external HTTP services."""

    llm_url: str = "http://host.docker.internal:1234/v1"
    embeddings_url: str = "http://host.docker.internal:1234/v1/embeddings"
    reranker_url: str = "http://host.docker.internal:8001/rerank"


class ModelsConfig(BaseModel):
    """Model names for the LLM, embeddings, and reranker."""

    llm_model: str = "t-lite-it-2.1"
    embeddings_model: str = "text-embedding-user-bge-m3"
    reranker_model: str = "nvidia/Llama-Nemotron-Rerank-1B-v2"


class EmbeddingsConfig(BaseModel):
    """Embedding vector parameters."""

    dimension: int = 1024


class IndexingConfig(BaseModel):
    """Document splitting and batched indexing parameters."""

    chunk_size: int = 600
    chunk_overlap: int = 100
    batch_size: int = 16


class SearchConfig(BaseModel):
    """Hybrid search and source filtering parameters."""

    top_k_retrieval: int = 50
    top_k_rerank: int = 12
    min_relevance: float = 0.01
    bm25_weight: float = 0.3
    rrf_k: int = 60


class ChatConfig(BaseModel):
    """Chat and dialog context parameters."""

    default_title: str = "Без названия"
    title_max_length: int = 50
    context_limit: int = 20


class LLMConfig(BaseModel):
    """LLM answer generation parameters."""

    max_tokens: int = 8192
    temperature: float = 0.1


class AppConfig(BaseModel):
    """Complete application configuration."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    urls: URLsConfig = Field(default_factory=URLsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "AppConfig":
        """Loads configuration from a single YAML file."""
        return cls(**_read_yaml_mapping(Path(path)))

    @classmethod
    def load(cls, *paths: str | Path) -> "AppConfig":
        """Loads cascading configs where each next file overrides previous values."""
        merged: dict[str, Any] = {}

        for path in paths:
            data = _read_yaml_mapping(Path(path))
            merged = _deep_merge(merged, data)

        return cls(**merged)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Reads a YAML file and ensures the top level is a mapping."""
    if not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merges dictionaries: override wins over base."""
    result = base.copy()

    for key, value in override.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            result[key] = _deep_merge(base_value, value)
        else:
            result[key] = value

    return result


config = AppConfig.load("config.yaml", "config.local.yaml")
