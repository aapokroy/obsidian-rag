"""Client for the LM Studio embeddings API."""

import logging

import numpy as np
import numpy.typing as npt
import requests

from lib.config import config

logger = logging.getLogger("obsidian-rag")


class Embedder:
    """HTTP client for retrieving embedding vectors."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        timeout: int = 60,
        session: requests.Session | None = None,
    ) -> None:
        """Initializes the embeddings API client."""
        self.url = url or config.urls.embeddings_url
        self.model = model or config.models.embeddings_model
        self.timeout = timeout
        self.session = session or requests.Session()
        logger.debug("Embedder initialized: url=%s, model=%s", self.url, self.model)

    def embed(
        self,
        text: str,
        *,
        normalize: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Returns an embedding for a single text."""
        embeddings = self.embed_many([text], normalize=normalize)
        return embeddings[0]

    def embed_many(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Returns embeddings for a list of texts."""
        if not texts:
            logger.debug("Embedding skipped: empty input")
            return np.empty((0, 0), dtype=np.float64)

        logger.debug("Requesting embeddings: count=%s, normalize=%s", len(texts), normalize)
        response = self.session.post(
            self.url,
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        embeddings = self._parse_embeddings(response.json(), expected_count=len(texts))
        if normalize:
            embeddings = self._l2_normalize(embeddings)

        logger.debug("Embeddings received: shape=%s", embeddings.shape)
        return embeddings

    @staticmethod
    def _parse_embeddings(
        payload: dict,
        *,
        expected_count: int,
    ) -> npt.NDArray[np.float64]:
        """Extracts embedding vectors from an API response and validates their count."""
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("Embeddings response does not contain a data list")
        if len(data) != expected_count:
            raise ValueError(
                "Embeddings response size mismatch: "
                f"expected {expected_count}, got {len(data)}"
            )

        return np.array(
            [item["embedding"] for item in data],
            dtype=np.float64,
        )

    @staticmethod
    def _l2_normalize(embeddings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """L2-normalizes embeddings with zero-vector protection."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms
