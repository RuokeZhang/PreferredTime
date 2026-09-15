from dataclasses import dataclass
from typing import Iterable

import numpy as np

from recsys.retrieval import Candidate


FEATURE_NAMES = tuple(
    feature
    for route in ("u2i", "i2i", "semantic", "popularity")
    for feature in (f"{route}_score", f"{route}_reciprocal_rank", f"{route}_hit")
) + (
    "user_interaction_count",
    "user_average_rating",
    "user_rating_std",
    "user_active_days",
    "user_recency_days",
    "item_interaction_count",
    "item_average_rating",
    "item_rating_std",
    "item_active_days",
    "item_recency_days",
    "item_popularity",
)


@dataclass(frozen=True)
class RankingExample:
    candidate: Candidate
    user_features: dict[str, float]
    item_features: dict[str, float]


def build_feature_matrix(examples: Iterable[RankingExample]) -> np.ndarray:
    rows = []
    for example in examples:
        values = []
        for route in ("u2i", "i2i", "semantic", "popularity"):
            rank = example.candidate.source_ranks.get(route)
            values.extend(
                [
                    example.candidate.source_scores.get(route, 0.0),
                    0.0 if rank is None else 1.0 / rank,
                    0.0 if rank is None else 1.0,
                ]
            )
        values.extend(
            [
                example.user_features.get("interaction_count", 0.0),
                example.user_features.get("average_rating", 0.0),
                example.user_features.get("rating_std", 0.0),
                example.user_features.get("active_days", 0.0),
                example.user_features.get("recency_days", 0.0),
                example.item_features.get("interaction_count", 0.0),
                example.item_features.get("average_rating", 0.0),
                example.item_features.get("rating_std", 0.0),
                example.item_features.get("active_days", 0.0),
                example.item_features.get("recency_days", 0.0),
                example.item_features.get("popularity", 0.0),
            ]
        )
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


class LambdaRanker:
    def __init__(self, **parameters):
        self.parameters = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 7,
            "n_jobs": 4,
            "verbosity": -1,
            **parameters,
        }
        self.model = None

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        group_sizes: list[int],
    ) -> "LambdaRanker":
        if sum(group_sizes) != len(labels):
            raise ValueError("group sizes must sum to the number of labels")
        try:
            from lightgbm import LGBMRanker
        except ImportError as error:
            raise RuntimeError("LambdaRanker requires lightgbm") from error
        self.model = LGBMRanker(**self.parameters)
        self.model.fit(features, labels, group=group_sizes)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ranker has not been fitted")
        return np.asarray(self.model.predict(features), dtype=np.float64)


def sample_ranking_group(
    candidates: list[Candidate],
    positive_items: set[int],
    *,
    negative_ratio: int = 4,
    random_negative_items: Iterable[int] = (),
    random_negative_ratio: int = 0,
    seed: int = 7,
) -> tuple[list[Candidate], np.ndarray]:
    if not 0 <= random_negative_ratio <= negative_ratio:
        raise ValueError("random negative ratio must be between zero and total ratio")
    positives = [candidate for candidate in candidates if candidate.item_id in positive_items]
    negatives = [candidate for candidate in candidates if candidate.item_id not in positive_items]
    random = np.random.default_rng(seed)
    positive_count = max(1, len(positives))
    hard_negative_count = min(
        len(negatives), positive_count * (negative_ratio - random_negative_ratio)
    )
    if hard_negative_count < len(negatives):
        selected = random.choice(
            len(negatives), size=hard_negative_count, replace=False
        )
        negatives = [negatives[int(index)] for index in selected]
    candidate_ids = {candidate.item_id for candidate in candidates}
    random_pool = sorted(
        set(int(item_id) for item_id in random_negative_items)
        .difference(positive_items)
        .difference(candidate_ids)
    )
    random_count = min(len(random_pool), positive_count * random_negative_ratio)
    selected_random = (
        random.choice(random_pool, size=random_count, replace=False)
        if random_count
        else []
    )
    group = positives + negatives + [
        Candidate(item_id=int(item_id)) for item_id in selected_random
    ]
    labels = np.asarray(
        [1 if candidate.item_id in positive_items else 0 for candidate in group],
        dtype=np.int32,
    )
    return group, labels
