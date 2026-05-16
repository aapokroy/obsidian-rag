"""Конфигурация приложения."""
from pathlib import Path
from pydantic import BaseModel, Field
import yaml


class PathsConfig(BaseModel):
    vault_path: Path = Path("/vault")
    data_path: Path = Path("/app/data")
    db_path: Path = Path("/app/data/app.db")


class URLsConfig(BaseModel):
    llm_url: str = "http://host.docker.internal:1234/v1"
    embeddings_url: str = "http://host.docker.internal:1234/v1/embeddings"
    reranker_url: str = "http://host.docker.internal:8001/rerank"


class ModelsConfig(BaseModel):
    llm_model: str = "t-lite-it-2.1"
    embeddings_model: str = "text-embedding-user-bge-m3"
    reranker_model: str = "nvidia/Llama-Nemotron-Rerank-1B-v2"


class EmbeddingsConfig(BaseModel):
    dimension: int = 1024


class IndexingConfig(BaseModel):
    chunk_size: int = 600
    chunk_overlap: int = 100
    batch_size: int = 16


class SearchConfig(BaseModel):
    top_k_retrieval: int = 50
    top_k_rerank: int = 12
    min_relevance: float = 0.01
    bm25_weight: float = 0.3
    rrf_k: int = 60


class ChatConfig(BaseModel):
    default_title: str = "Без названия"
    title_max_length: int = 50
    context_limit: int = 20


class LLMConfig(BaseModel):
    max_tokens: int = 8192
    temperature: float = 0.1


class AppConfig(BaseModel):
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
        """Загружает конфиг из YAML-файла с поддержкой каскадного переопределения."""
        config_path = Path(path)
        data: dict = {}

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        return cls(**data)

    @classmethod
    def load(cls, *paths: str | Path) -> "AppConfig":
        """
        Каскадная загрузка: каждый следующий файл переопределяет предыдущий.

        config = AppConfig.load("config.yaml", "config.local.yaml")
        """
        merged: dict = {}

        for path in paths:
            config_path = Path(path)
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                merged = cls._deep_merge(merged, data)

        return cls(**merged)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Рекурсивно сливает словари (override переопределяет base)."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = AppConfig._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


config = AppConfig.from_yaml()
