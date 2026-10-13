"""Deterministic embeddings so tests can cluster and retrieve without a model download."""

from llama_index.core.embeddings import BaseEmbedding


class KeywordEmbedding(BaseEmbedding):
    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        if "страхов" in lowered:
            return [1.0, 0.0, 0.0]
        if "лыж" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)
