from dataclasses import dataclass
from time import perf_counter

import numpy as np

from recsys.ranking import RankingExample, build_feature_matrix
from recsys.reranking import mmr_rerank
from recsys.retrieval import Candidate, MultiRouteRetriever


@dataclass(frozen=True)
class RecommendationResult:
    item_ids: list[int]
    model_version: str
    stage_latency_ms: dict[str, float]
    candidate_count: int
    route_candidate_counts: dict[str, int]


class RecommendationPipeline:
    def __init__(
        self,
        *,
        retriever: MultiRouteRetriever,
        item_features: dict[int, dict[str, float]],
        item_vectors: dict[int, np.ndarray],
        ranker=None,
        model_version: str = "unversioned",
        rerank_pool_size: int = 100,
        mmr_relevance_weight: float = 0.8,
        max_candidates: int = 500,
        candidates_per_route: int = 150,
    ):
        self.retriever = retriever
        self.item_features = item_features
        self.item_vectors = item_vectors
        self.ranker = ranker
        self.model_version = model_version
        self.rerank_pool_size = rerank_pool_size
        self.mmr_relevance_weight = mmr_relevance_weight
        self.max_candidates = max_candidates
        self.candidates_per_route = candidates_per_route

    def recommend(
        self,
        user_id: int,
        history: list[int],
        user_features: dict[str, float],
        *,
        positive_history: list[int] | None = None,
        limit: int = 20,
    ) -> RecommendationResult:
        request_start = perf_counter()
        retrieval_start = perf_counter()
        candidates = self.retriever.retrieve(
            user_id,
            history,
            positive_history=positive_history,
            per_route=self.candidates_per_route,
            max_candidates=self.max_candidates,
        )
        retrieval_ms = (perf_counter() - retrieval_start) * 1000

        feature_start = perf_counter()
        examples = [
            RankingExample(
                candidate=candidate,
                user_features=user_features,
                item_features=self.item_features.get(candidate.item_id, {}),
            )
            for candidate in candidates
        ]
        features = None if self.ranker is None else build_feature_matrix(examples)
        feature_assembly_ms = (perf_counter() - feature_start) * 1000

        ranking_start = perf_counter()
        if self.ranker is None:
            relevance_scores = {
                candidate.item_id: candidate.fusion_score for candidate in candidates
            }
        else:
            scores = self.ranker.predict(features)
            relevance_scores = {
                candidate.item_id: float(score)
                for candidate, score in zip(candidates, scores)
            }
        ranked_items = sorted(
            (candidate.item_id for candidate in candidates),
            key=lambda item_id: -relevance_scores[item_id],
        )[: self.rerank_pool_size]
        ranking_ms = (perf_counter() - ranking_start) * 1000

        reranking_start = perf_counter()
        item_ids = mmr_rerank(
            ranked_items,
            relevance_scores,
            self.item_vectors,
            limit=limit,
            relevance_weight=self.mmr_relevance_weight,
        )
        reranking_ms = (perf_counter() - reranking_start) * 1000
        total_ms = (perf_counter() - request_start) * 1000
        return RecommendationResult(
            item_ids=item_ids,
            model_version=self.model_version,
            stage_latency_ms={
                "retrieval": retrieval_ms,
                "feature_assembly": feature_assembly_ms,
                "ranking": ranking_ms,
                "reranking": reranking_ms,
                "total": total_ms,
            },
            candidate_count=len(candidates),
            route_candidate_counts={
                route: sum(route in candidate.source_scores for candidate in candidates)
                for route in self.retriever.ROUTES
            },
        )
