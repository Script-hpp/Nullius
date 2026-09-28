"""Fester, getesteter Metrik-Code. Agent darf diesen Code nicht sehen oder aendern."""

from __future__ import annotations

import math


def accuracy(predictions: list, labels: list) -> float:
    if len(predictions) != len(labels):
        raise ValueError("predictions und labels muessen gleich lang sein")
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(labels)


def mae(predictions: list[float], labels: list[float]) -> float:
    if len(predictions) != len(labels):
        raise ValueError("predictions und labels muessen gleich lang sein")
    return sum(abs(p - l) for p, l in zip(predictions, labels)) / len(labels)


def mse(predictions: list[float], labels: list[float]) -> float:
    if len(predictions) != len(labels):
        raise ValueError("predictions und labels muessen gleich lang sein")
    return sum((p - l) ** 2 for p, l in zip(predictions, labels)) / len(labels)


def recall_at_k(ranked_lists: list[list], relevant: list[set], k: int) -> float:
    """ranked_lists[i] = geordnete Kandidaten fuer Nutzer i, relevant[i] = Menge der Treffer."""
    if len(ranked_lists) != len(relevant):
        raise ValueError("ranked_lists und relevant muessen gleich lang sein")
    scores = []
    for ranked, rel in zip(ranked_lists, relevant):
        if not rel:
            continue
        top_k = set(ranked[:k])
        scores.append(len(top_k & rel) / len(rel))
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(ranked_lists: list[list], relevant: list[set], k: int) -> float:
    scores = []
    for ranked, rel in zip(ranked_lists, relevant):
        if not rel:
            continue
        dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in rel)
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
        scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0
