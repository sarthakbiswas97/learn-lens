"""Batch embedding and similarity search."""

from __future__ import annotations

import logging

import numpy as np

from learnlens.config import LearnLensConfig
from learnlens.models.connector import ConnectorModel
from learnlens.models.types import ContentItem, Goal
from learnlens.storage.database import Database
from learnlens.storage.queries import (
    get_all_embeddings,
    get_content_by_id,
    get_items_without_interactions,
    get_unembedded_content,
    insert_embedding,
)

logger = logging.getLogger(__name__)


def embed_new_content(db: Database, model: ConnectorModel) -> int:
    """Embed all content that doesn't have embeddings yet.

    Returns count of newly embedded items.
    """
    items = get_unembedded_content(db.connection)
    if not items:
        return 0

    logger.info("Embedding %d new documents", len(items))
    texts = [item.body_text for item in items]
    embeddings = model.embed_documents(texts)

    for item, vector in zip(items, embeddings, strict=False):
        insert_embedding(
            conn=db.connection,
            content_id=item.id,
            vector=vector,
            dim_size=vector.shape[-1],
            model_version="nomic-embed-text-v1.5",
        )

    logger.info("Embedded %d documents", len(items))
    return len(items)


def find_similar(
    query_vec: np.ndarray,
    db: Database,
    top_k: int = 10,
) -> list[tuple[ContentItem, float]]:
    """Find top-K most similar content items to query vector.

    Returns list of (ContentItem, cosine_similarity) pairs, sorted descending.
    Uses brute-force cosine similarity (fast for <100K docs).
    """
    content_ids, all_vecs = get_all_embeddings(db.connection)
    if not content_ids or all_vecs.size == 0:
        return []

    similarities = cosine_similarity(query_vec, all_vecs)

    if top_k < len(similarities):
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    else:
        top_indices = np.argsort(similarities)[::-1]

    results: list[tuple[ContentItem, float]] = []
    for idx in top_indices:
        item = get_content_by_id(db.connection, content_ids[idx])
        if item:
            results.append((item, float(similarities[idx])))
    return results


def find_forgotten_items(
    db: Database,
    model: ConnectorModel,
    goals: list[Goal],
    config: LearnLensConfig,
) -> list[ContentItem]:
    """Find items ingested N+ days ago with high goal relevance but no interactions.

    Embeds active goals, computes similarity to all old content,
    filters by interaction count = 0.
    """
    if not goals:
        return []

    candidate_ids = get_items_without_interactions(
        db.connection, days_threshold=config.forgotten_days_threshold
    )
    if not candidate_ids:
        return []

    content_ids, all_vecs = get_all_embeddings(db.connection)
    if not content_ids or all_vecs.size == 0:
        return []

    # Build mask for candidate content
    id_to_idx = {cid: idx for idx, cid in enumerate(content_ids)}
    candidate_indices = [id_to_idx[cid] for cid in candidate_ids if cid in id_to_idx]
    if not candidate_indices:
        return []

    candidate_vecs = all_vecs[candidate_indices]

    # Embed goals and compute max similarity
    goal_texts = [g.goal_text for g in goals]
    goal_embeddings = model.embed_documents(goal_texts)

    # Cosine similarity matrix: candidates x goals
    similarities = cosine_similarity_matrix(candidate_vecs, goal_embeddings)
    max_sims = similarities.max(axis=1)

    threshold = config.forgotten_similarity_threshold
    forgotten: list[ContentItem] = []
    for i, idx in enumerate(candidate_indices):
        if max_sims[i] >= threshold:
            item = get_content_by_id(db.connection, content_ids[idx])
            if item:
                forgotten.append(item)

    logger.info("Found %d forgotten items", len(forgotten))
    return forgotten


def cosine_similarity(query_vec: np.ndarray, all_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all vectors.

    Args:
        query_vec: shape (1, dim) or (dim,)
        all_vecs: shape (N, dim)

    Returns:
        similarities of shape (N,)
    """
    query = query_vec.flatten()
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    vecs_norm = all_vecs / np.where(norms > 0, norms, 1)
    return vecs_norm @ query_norm


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        a: shape (M, dim)
        b: shape (N, dim)

    Returns:
        similarity matrix of shape (M, N)
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T



