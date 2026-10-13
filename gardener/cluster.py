"""Greedy cosine clustering and a stable fingerprint for dedupe."""

from __future__ import annotations

import hashlib

import numpy as np

from .questions import normalize


def fingerprint(texts: list[str]) -> str:
    body = "\n".join(sorted(normalize(t) for t in texts))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def greedy_clusters(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Group row vectors that already have unit length.

    A point joins the existing cluster whose nearest member is at least
    `threshold`, preferring the closest such cluster. Order is the row order.
    """
    clusters: list[list[int]] = []
    for i in range(len(vectors)):
        best_j: int | None = None
        best_sim = threshold
        for ci, members in enumerate(clusters):
            sim = float(np.max(vectors[members] @ vectors[i]))
            if sim + 1e-9 >= threshold and (best_j is None or sim > best_sim):
                best_sim = sim
                best_j = ci
        if best_j is None:
            clusters.append([i])
        else:
            clusters[best_j].append(i)
    return clusters
