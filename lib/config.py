"""Pydantic-конфиг приложения."""

from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class Config(BaseModel):
    vault_path: str = "/vault"
    db_path: str = "/app/chroma_db"
    llm_url: str = "http://host.docker.internal:1234/v1"
    embeddings_url: str = "http://host.docker.internal:1234/v1/embeddings"
    reranker_url: str = "http://host.docker.internal:8001/rerank"
    llm_model: str = "t-tech/T-lite-it-2.1:q5_0"
    embeddings_model: str = "bge-m3"
    reranker_model: str = "nvidia/Llama-Nemotron-Rerank-1B-v2"
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k_retrieval: int = 50
    top_k_final: int = 12
    min_relevance: float = 0.01
    bm25_weight: float = 0.3
    rrf_k: int = 60
    llm_context_length: int = 8192

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)