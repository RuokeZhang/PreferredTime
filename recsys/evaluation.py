import math
from time import perf_counter

import numpy as np

from recsys.retrieval import VectorIndex


def recall_at_k(recommendations: list[int], relevant_items: set[int], k: int) -> float:
    if not relevant_items:
        return 0.0
    return len(set(recommendations[:k]).intersection(relevant_items)) / len(relevant_items)


def ndcg_at_k(recommendations: list[int], relevant_items: set[int], k: int) -> float:
    if not relevant_items:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item_id in enumerate(recommendations[:k])
        if item_id in relevant_items
    )
    ideal_length = min(k, len(relevant_items))
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_length))
    return dcg / ideal


def intra_list_diversity(
    recommendations: list[int], item_vectors: dict[int, np.ndarray]
) -> float:
    vectors = [item_vectors[item_id] for item_id in recommendations if item_id in item_vectors]
    if len(vectors) < 2:
        return 0.0
    distances = []
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1 :]:
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            similarity = 0.0 if denominator == 0 else float(np.dot(left, right) / denominator)
            distances.append(1.0 - similarity)
    return float(np.mean(distances))


def evaluate_recommendations(
    recommendations_by_user: dict[int, list[int]],
    relevant_by_user: dict[int, set[int]],
    item_vectors: dict[int, np.ndarray],
    *,
    catalog_size: int,
    recall_k: int = 20,
    ndcg_k: int = 10,
) -> dict[str, float]:
    users = sorted(set(recommendations_by_user).intersection(relevant_by_user))
    if not users:
        raise ValueError("no users have both recommendations and relevant items")
    recall_values = [
        recall_at_k(recommendations_by_user[user_id], relevant_by_user[user_id], recall_k)
        for user_id in users
    ]
    ndcg_values = [
        ndcg_at_k(recommendations_by_user[user_id], relevant_by_user[user_id], ndcg_k)
        for user_id in users
    ]
    diversity_values = [
        intra_list_diversity(recommendations_by_user[user_id], item_vectors)
        for user_id in users
    ]
    unique_recommendations = {
        item_id
        for recommendations in recommendations_by_user.values()
        for item_id in recommendations
    }
    return {
        f"recall@{recall_k}": float(np.mean(recall_values)),
        f"ndcg@{ndcg_k}": float(np.mean(ndcg_values)),
        "coverage": len(unique_recommendations) / catalog_size,
        "diversity": float(np.mean(diversity_values)),
        "evaluated_users": float(len(users)),
    }


def benchmark_ann(
    item_vectors: dict[int, np.ndarray],
    query_vectors: list[np.ndarray],
    *,
    k: int = 100,
    ef_search: int = 200,
) -> dict[str, float]:
    exact_index = VectorIndex(item_vectors, backend="exact")
    approximate_index = VectorIndex(
        item_vectors, backend="hnsw", ef_search=ef_search
    )
    recalls = []
    exact_latencies = []
    approximate_latencies = []
    for query_vector in query_vectors:
        start = perf_counter()
        exact_ids = {item_id for item_id, _ in exact_index.query(query_vector, k)}
        exact_latencies.append((perf_counter() - start) * 1000)
        start = perf_counter()
        approximate_ids = {
            item_id for item_id, _ in approximate_index.query(query_vector, k)
        }
        approximate_latencies.append((perf_counter() - start) * 1000)
        recalls.append(len(exact_ids.intersection(approximate_ids)) / len(exact_ids))
    return {
        f"ann_recall@{k}": float(np.mean(recalls)),
        "ef_search": float(ef_search),
        "exact_p50_ms": float(np.percentile(exact_latencies, 50)),
        "exact_p99_ms": float(np.percentile(exact_latencies, 99)),
        "ann_p50_ms": float(np.percentile(approximate_latencies, 50)),
        "ann_p99_ms": float(np.percentile(approximate_latencies, 99)),
    }
