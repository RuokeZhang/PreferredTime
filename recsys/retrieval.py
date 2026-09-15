from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class Candidate:
    item_id: int
    source_scores: dict[str, float] = field(default_factory=dict)
    source_ranks: dict[str, int] = field(default_factory=dict)

    @property
    def fusion_score(self) -> float:
        return sum(1.0 / (60 + rank) for rank in self.source_ranks.values())


class VectorIndex:
    def __init__(
        self,
        vectors: dict[int, np.ndarray],
        backend: str = "exact",
        *,
        hnsw_m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 200,
    ):
        if not vectors:
            raise ValueError("vector index requires at least one item")
        self.item_ids = np.asarray(sorted(vectors), dtype=np.int64)
        matrix = np.vstack([vectors[int(item_id)] for item_id in self.item_ids]).astype(
            np.float32
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms
        self.backend = backend
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.fallback_count = 0
        self._index = None
        if backend == "hnsw":
            self._build_hnsw_index()
        elif backend != "exact":
            raise ValueError(f"unsupported vector-index backend: {backend}")

    def _build_hnsw_index(self) -> None:
        try:
            import hnswlib
        except ImportError as error:
            raise RuntimeError("hnsw backend requires hnswlib") from error
        index = hnswlib.Index(space="cosine", dim=self.matrix.shape[1])
        index.init_index(
            max_elements=len(self.item_ids),
            ef_construction=self.ef_construction,
            M=self.hnsw_m,
        )
        index.add_items(self.matrix, np.arange(len(self.item_ids)))
        index.set_ef(min(self.ef_search, len(self.item_ids)))
        self._index = index

    def query(self, vector: np.ndarray, limit: int) -> list[tuple[int, float]]:
        normalized = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(normalized))
        if norm == 0:
            return []
        normalized = normalized / norm
        result_limit = min(limit, len(self.item_ids))
        if self.backend == "hnsw":
            self._index.set_ef(
                min(max(self.ef_search, result_limit), len(self.item_ids))
            )
            try:
                labels, distances = self._index.knn_query(normalized, k=result_limit)
                return [
                    (int(self.item_ids[label]), float(1.0 - distance))
                    for label, distance in zip(labels[0], distances[0])
                ]
            except RuntimeError:
                self.fallback_count += 1
                return self._exact_query(normalized, result_limit)
        return self._exact_query(normalized, result_limit)

    def _exact_query(
        self, normalized_vector: np.ndarray, result_limit: int
    ) -> list[tuple[int, float]]:
        scores = self.matrix @ normalized_vector
        order = np.argsort(-scores, kind="stable")[:result_limit]
        return [(int(self.item_ids[index]), float(scores[index])) for index in order]


class MultiRouteRetriever:
    ROUTES = ("u2i", "i2i", "semantic", "popularity")

    def __init__(
        self,
        *,
        user_vectors: dict[int, np.ndarray],
        collaborative_item_vectors: dict[int, np.ndarray],
        semantic_item_vectors: dict[int, np.ndarray],
        popularity: list[int],
        backend: str = "exact",
        enabled_routes: tuple[str, ...] = ROUTES,
    ):
        unknown_routes = set(enabled_routes).difference(self.ROUTES)
        if unknown_routes:
            raise ValueError(f"unsupported retrieval routes: {sorted(unknown_routes)}")
        self.user_vectors = user_vectors
        self.collaborative_vectors = collaborative_item_vectors
        self.semantic_vectors = semantic_item_vectors
        self.popularity = popularity
        self.enabled_routes = enabled_routes
        self.collaborative_index = VectorIndex(collaborative_item_vectors, backend)
        self.semantic_index = VectorIndex(semantic_item_vectors, backend)

    def retrieve(
        self,
        user_id: int,
        history: Iterable[int],
        *,
        positive_history: Iterable[int] | None = None,
        per_route: int = 150,
        max_candidates: int = 500,
    ) -> list[Candidate]:
        history_items = [int(item_id) for item_id in history]
        profile_items = (
            history_items
            if positive_history is None
            else [int(item_id) for item_id in positive_history]
        )
        seen_items = set(history_items)
        retrieval_limit = per_route + min(len(seen_items), per_route * 2)
        candidates: dict[int, Candidate] = {}

        user_vector = self.user_vectors.get(user_id)
        if "u2i" in self.enabled_routes and user_vector is not None:
            self._merge_route(
                candidates,
                "u2i",
                self.collaborative_index.query(user_vector, retrieval_limit),
                seen_items,
                per_route,
            )

        collaborative_history = [
            self.collaborative_vectors[item_id]
            for item_id in profile_items[-10:]
            if item_id in self.collaborative_vectors
        ]
        if "i2i" in self.enabled_routes and collaborative_history:
            self._merge_route(
                candidates,
                "i2i",
                self.collaborative_index.query(
                    np.mean(collaborative_history, axis=0), retrieval_limit
                ),
                seen_items,
                per_route,
            )

        semantic_history = [
            self.semantic_vectors[item_id]
            for item_id in profile_items[-10:]
            if item_id in self.semantic_vectors
        ]
        if "semantic" in self.enabled_routes and semantic_history:
            self._merge_route(
                candidates,
                "semantic",
                self.semantic_index.query(
                    np.mean(semantic_history, axis=0), retrieval_limit
                ),
                seen_items,
                per_route,
            )

        if "popularity" in self.enabled_routes:
            popularity_results = [
                (item_id, 1.0 / rank)
                for rank, item_id in enumerate(self.popularity, start=1)
                if item_id not in seen_items
            ]
            self._merge_route(
                candidates,
                "popularity",
                popularity_results,
                seen_items,
                per_route,
            )
        return sorted(candidates.values(), key=lambda item: -item.fusion_score)[
            :max_candidates
        ]

    @staticmethod
    def _merge_route(
        candidates: dict[int, Candidate],
        route: str,
        results: Iterable[tuple[int, float]],
        seen_items: set[int],
        limit: int,
    ) -> None:
        accepted = 0
        for item_id, score in results:
            if item_id in seen_items:
                continue
            accepted += 1
            candidate = candidates.setdefault(item_id, Candidate(item_id=item_id))
            candidate.source_scores[route] = float(score)
            candidate.source_ranks[route] = accepted
            if accepted == limit:
                break
