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
    if relevance_weight == 1.0:
        return sorted(
            remaining,
            key=lambda item_id: -relevance_scores.get(item_id, 0.0),
        )[:limit]
    if not remaining:
        return []

    dimensions = next(
        (len(item_vectors[item_id]) for item_id in remaining if item_id in item_vectors),
        0,
    )
    matrix = np.zeros((len(remaining), dimensions), dtype=np.float32)
    for index, item_id in enumerate(remaining):
        if item_id in item_vectors:
            matrix[index] = item_vectors[item_id]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    similarities = (matrix / norms) @ (matrix / norms).T
    relevance = np.asarray(
        [relevance_scores.get(item_id, 0.0) for item_id in remaining],
        dtype=np.float32,
    )
    redundancy = np.zeros(len(remaining), dtype=np.float32)
    available = np.ones(len(remaining), dtype=bool)
    selected_indices = []
    while available.any() and len(selected_indices) < limit:
        scores = relevance_weight * relevance - (1.0 - relevance_weight) * redundancy
        scores[~available] = -np.inf
        selected_index = int(np.argmax(scores))
        selected_indices.append(selected_index)
        available[selected_index] = False
        redundancy = np.maximum(redundancy, similarities[:, selected_index])
    return [remaining[index] for index in selected_indices]
