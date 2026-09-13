import numpy as np


def mmr_rerank(
    item_ids: list[int],
    relevance_scores: dict[int, float],
    item_vectors: dict[int, np.ndarray],
    *,
    limit: int = 20,
    relevance_weight: float = 0.8,
) -> list[int]:
    if not 0.0 <= relevance_weight <= 1.0:
        raise ValueError("relevance_weight must be between 0 and 1")
    remaining = list(dict.fromkeys(item_ids))
    selected: list[int] = []
    while remaining and len(selected) < limit:
        best_item = max(
            remaining,
            key=lambda item_id: _mmr_score(
                item_id,
                selected,
                relevance_scores,
                item_vectors,
                relevance_weight,
            ),
        )
        selected.append(best_item)
        remaining.remove(best_item)
    return selected


def _mmr_score(
    item_id: int,
    selected: list[int],
    relevance_scores: dict[int, float],
    item_vectors: dict[int, np.ndarray],
    relevance_weight: float,
) -> float:
    relevance = relevance_scores.get(item_id, 0.0)
    if not selected or item_id not in item_vectors:
        return relevance_weight * relevance
    similarities = [
        _cosine_similarity(item_vectors[item_id], item_vectors[selected_item])
        for selected_item in selected
        if selected_item in item_vectors
    ]
    redundancy = max(similarities, default=0.0)
    return relevance_weight * relevance - (1.0 - relevance_weight) * redundancy


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)
