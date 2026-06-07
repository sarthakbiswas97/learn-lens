"""Wrapper for nomic-embed-text-v1.5 embedding model."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from learnlens.config import LearnLensConfig

logger = logging.getLogger(__name__)

_PREFIX_DOCUMENT = "search_document: "
_PREFIX_QUERY = "search_query: "


class ConnectorModel:
    """Wrapper for nomic-embed-text-v1.5 embedding model."""

    def __init__(self, config: LearnLensConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        """Load model to device."""
        logger.info("Loading embedding model: %s", self._config.embedding_model_id)
        self._model = SentenceTransformer(
            self._config.embedding_model_id,
            trust_remote_code=True,
        )
        logger.info("Embedding model loaded")

    def _ensure_loaded(self) -> SentenceTransformer:
        if self._model is None:
            self.load()
        return self._model  # type: ignore[return-value]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed documents with 'search_document:' prefix.

        Args:
            texts: raw document texts (prefix added internally)

        Returns:
            numpy array of shape (len(texts), config.embedding_dim)
        """
        model = self._ensure_loaded()
        prefixed = [_PREFIX_DOCUMENT + text for text in texts]
        embeddings = model.encode(
            prefixed,
            show_progress_bar=False,
            batch_size=32,
            convert_to_numpy=True,
        )
        return self._truncate(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with 'search_query:' prefix.

        Returns:
            numpy array of shape (1, config.embedding_dim)
        """
        model = self._ensure_loaded()
        prefixed = [_PREFIX_QUERY + query]
        embedding = model.encode(
            prefixed,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return self._truncate(embedding)

    def _truncate(self, embeddings: np.ndarray) -> np.ndarray:
        """Truncate and re-normalize for Matryoshka dimensionality reduction."""
        target_dim = self._config.embedding_dim
        if embeddings.shape[-1] == target_dim:
            return embeddings

        tensor = torch.tensor(embeddings)
        normalized = F.layer_norm(tensor, normalized_shape=(tensor.shape[-1],))
        truncated = normalized[..., :target_dim]
        result = F.normalize(truncated, p=2, dim=-1)
        return result.numpy()
