"""Реранкинг через внешний API."""

import numpy as np
import requests

from lib.config import config
from lib.schema import RetrieveResultItem, RerankResultItem


class Reranker:
    """Реранкер, использующий внешний API для пересортировки результатов поиска."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        top_k: int | None = None,
        timeout: int = 30,
    ) -> None:
        self.url = url or config.urls.reranker_url
        self.model = model or config.models.reranker_model
        self.top_k = top_k or config.search.top_k_rerank
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        retrieve_results: list[RetrieveResultItem],
    ) -> list[RerankResultItem]:
        """Переранжирует результаты поиска."""
        if not retrieve_results:
            return []

        texts = [res.text for res in retrieve_results]
        scores = self._call_api(query, texts)
        scores = self._normalize(np.array(scores)).tolist()

        pairs = list(zip(retrieve_results, scores))
        pairs.sort(key=lambda pair: pair[1], reverse=True)

        return [
            RerankResultItem(
                chunk_id=res.chunk_id,
                document_id=res.document_id,
                text=res.text,
                document_path=res.document_path,
                relevance=round(score, 5),
            )
            for res, score in pairs[: self.top_k]
        ]

    def _call_api(
        self,
        query: str,
        texts: list[str],
    ) -> list[float]:
        """Отправляет запрос к API реранкера и возвращает сырые скоры."""
        response = requests.post(
            self.url,
            json={
                "query": query,
                "documents": texts,
                "model": self.model,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])
        return [r.get("relevance_score", 0.0) for r in results]

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """
        Сигмоида — абсолютная шкала релевантности.

        Пороги для min_relevance:
        0.7 — только высокорелевантные
        0.5 — релевантные и нейтральные
        0.3 — пропускать почти всё
        """
        scores = np.nan_to_num(scores, nan=0.0, posinf=15.0, neginf=-15.0)
        scores = np.clip(scores, -15.0, 15.0)
        return 1.0 / (1.0 + np.exp(-scores))
