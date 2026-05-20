"""Client for the external reranker API."""

import logging

import numpy as np
import numpy.typing as npt
import requests

from lib.config import config
from lib.schema import RerankResultItem, RetrieveResultItem

logger = logging.getLogger("obsidian-rag")


class Reranker:
    """HTTP client for reordering search results."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        top_k: int | None = None,
        timeout: int = 30,
        session: requests.Session | None = None,
    ) -> None:
        """Initializes the reranker client."""
        self.url = url or config.urls.reranker_url
        self.model = model or config.models.reranker_model
        self.top_k = top_k or config.search.top_k_rerank
        self.timeout = timeout
        self.session = session or requests.Session()
        logger.debug("Reranker initialized: url=%s, model=%s, top_k=%s", self.url, self.model, self.top_k)

    def rerank(
        self,
        query: str,
        retrieve_results: list[RetrieveResultItem],
    ) -> list[RerankResultItem]:
        """Reranks search results by relevance to the query."""
        if not retrieve_results:
            logger.debug("Reranking skipped: empty retrieve results")
            return []

        logger.debug("Reranking started: candidates=%s", len(retrieve_results))
        texts = [result.text for result in retrieve_results]
        raw_scores = self._call_api(query, texts)
        scores = self._normalize(np.array(raw_scores, dtype=np.float64)).tolist()

        pairs = sorted(
            zip(retrieve_results, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        reranked = [
            RerankResultItem(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=result.text,
                document_path=result.document_path,
                relevance=round(score, 5),
            )
            for result, score in pairs[: self.top_k]
        ]
        logger.debug("Reranking finished: returned=%s", len(reranked))
        return reranked

    def _call_api(
        self,
        query: str,
        texts: list[str],
    ) -> list[float]:
        """Calls the reranker API and returns scores in the original document order."""
        response = self.session.post(
            self.url,
            json={
                "query": query,
                "documents": texts,
                "model": self.model,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._scores_from_response(response.json(), count=len(texts))

    @staticmethod
    def _scores_from_response(payload: dict, *, count: int) -> list[float]:
        """Extracts scores from the API response while restoring the original document order."""
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ValueError("Reranker response does not contain a results list")

        scores = [0.0] * count
        for result in results:
            index = result.get("index")
            if isinstance(index, int) and 0 <= index < count:
                scores[index] = float(result.get("relevance_score", 0.0))

        return scores

    @staticmethod
    def _normalize(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Maps raw scores to a stable 0..1 scale with a sigmoid."""
        scores = np.nan_to_num(scores, nan=0.0, posinf=15.0, neginf=-15.0)
        scores = np.clip(scores, -15.0, 15.0)
        return 1.0 / (1.0 + np.exp(-scores))
