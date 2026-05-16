"""Эмбеддинги через внешний API (LM Studio)."""

import numpy as np
import numpy.typing as npt
import requests

from more_itertools import one

from lib.config import config


class Embedder:
    """Эмбеддер, использующий внешний API для получения векторных представлений."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.url = url or config.urls.embeddings_url
        self.model = model or config.models.embeddings_model
        self.timeout = timeout

    def embed(
        self,
        text: str,
        *,
        normalize: bool = True,
    ) -> npt.NDArray[np.float32]:
        """Получает эмбеддинг для одного текста."""
        return one(self.embed_many([text], normalize=normalize))

    def embed_many(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
    ) -> npt.NDArray[np.float32]:
        """Получает эмбеддинги для списка текстов."""
        if not texts:
            return np.empty((0,), dtype=np.float32)

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        embeddings = np.array(
            [item["embedding"] for item in data["data"]],
            dtype=np.float32,
        )

        if normalize:
            embeddings = self._l2_normalize(embeddings)

        return embeddings

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2-нормализация для косинусного сходства с защитой от нулевых векторов."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms
