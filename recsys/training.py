from dataclasses import dataclass

import numpy as np
import pandas as pd

from recsys.ranking import RankingExample, build_feature_matrix, sample_ranking_group
from recsys.retrieval import MultiRouteRetriever


@dataclass(frozen=True)
class FactorizationResult:
    user_vectors: dict[int, np.ndarray]
    item_vectors: dict[int, np.ndarray]


def train_als_vectors(
    ratings: pd.DataFrame,
    *,
    positive_threshold: float = 4.0,
    factors: int = 64,
    regularization: float = 0.05,
    iterations: int = 20,
    alpha: float = 20.0,
    seed: int = 7,
) -> FactorizationResult:
    try:
        from implicit.als import AlternatingLeastSquares
        from scipy.sparse import csr_matrix
    except ImportError as error:
        raise RuntimeError("ALS training requires implicit and scipy") from error

    positives = ratings[ratings["rating"] >= positive_threshold]
    if positives.empty:
        raise ValueError("ALS training requires positive interactions")
    user_ids = sorted(int(user_id) for user_id in positives["userId"].unique())
    item_ids = sorted(int(item_id) for item_id in positives["movieId"].unique())
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    rows = positives["userId"].map(user_index).to_numpy()
    columns = positives["movieId"].map(item_index).to_numpy()
    confidence = 1.0 + alpha * np.ones(len(positives), dtype=np.float32)
    user_items = csr_matrix(
        (confidence, (rows, columns)), shape=(len(user_ids), len(item_ids))
    )
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=seed,
    )
    model.fit(user_items, show_progress=False)
    return FactorizationResult(
        user_vectors={
            user_id: model.user_factors[index].astype(np.float32)
            for index, user_id in enumerate(user_ids)
        },
        item_vectors={
            item_id: model.item_factors[index].astype(np.float32)
            for index, item_id in enumerate(item_ids)
        },
    )


def build_content_vectors(
    movies: pd.DataFrame, genome_scores: pd.DataFrame | None = None
) -> dict[int, np.ndarray]:
    required_columns = {"movieId", "genres"}
    missing_columns = required_columns.difference(movies.columns)
    if missing_columns:
        raise ValueError(f"movies are missing columns: {sorted(missing_columns)}")
    movie_ids = [int(movie_id) for movie_id in movies["movieId"]]
    genres_by_movie = {
        int(row.movieId): set(str(row.genres).split("|")) - {"(no genres listed)"}
        for row in movies.itertuples()
    }
    genre_names = sorted({genre for genres in genres_by_movie.values() for genre in genres})
    genre_index = {genre: index for index, genre in enumerate(genre_names)}
    genre_matrix = np.zeros((len(movie_ids), len(genre_names)), dtype=np.float32)
    for row_index, movie_id in enumerate(movie_ids):
        for genre in genres_by_movie[movie_id]:
            genre_matrix[row_index, genre_index[genre]] = 1.0

    blocks = [genre_matrix]
    if genome_scores is not None and not genome_scores.empty:
        required_genome_columns = {"movieId", "tagId", "relevance"}
        missing_genome_columns = required_genome_columns.difference(genome_scores.columns)
        if missing_genome_columns:
            raise ValueError(
                f"genome scores are missing columns: {sorted(missing_genome_columns)}"
            )
        pivot = genome_scores.pivot(index="movieId", columns="tagId", values="relevance")
        pivot = pivot.reindex(movie_ids).fillna(0.0)
        blocks.append(pivot.to_numpy(dtype=np.float32))

    matrix = np.hstack(blocks)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    return {movie_id: matrix[index] for index, movie_id in enumerate(movie_ids)}


def build_training_features(
    ratings: pd.DataFrame,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]], list[int]]:
    aggregate_columns = ["count", "mean", "std"]
    user_aggregates = ratings.groupby("userId")["rating"].agg(aggregate_columns)
    item_aggregates = ratings.groupby("movieId")["rating"].agg(aggregate_columns)
    if "timestamp" in ratings:
        reference_timestamp = float(ratings["timestamp"].max())
        user_time = ratings.groupby("userId")["timestamp"].agg(["min", "max"])
        item_time = ratings.groupby("movieId")["timestamp"].agg(["min", "max"])
    else:
        reference_timestamp = 0.0
        user_time = None
        item_time = None
    max_item_count = max(float(item_aggregates["count"].max()), 1.0)
    user_features = {
        int(user_id): {
            "interaction_count": float(row["count"]),
            "average_rating": float(row["mean"]),
            "rating_std": float(0.0 if pd.isna(row["std"]) else row["std"]),
            "active_days": _active_days(user_time, user_id),
            "recency_days": _recency_days(user_time, user_id, reference_timestamp),
        }
        for user_id, row in user_aggregates.iterrows()
    }
    item_features = {
        int(item_id): {
            "interaction_count": float(row["count"]),
            "average_rating": float(row["mean"]),
            "rating_std": float(0.0 if pd.isna(row["std"]) else row["std"]),
            "active_days": _active_days(item_time, item_id),
            "recency_days": _recency_days(item_time, item_id, reference_timestamp),
            "popularity": float(np.log1p(row["count"]) / np.log1p(max_item_count)),
        }
        for item_id, row in item_aggregates.iterrows()
    }
    popularity = [
        int(item_id)
        for item_id in item_aggregates.sort_values(
            ["count", "mean"], ascending=False, kind="stable"
        ).index
    ]
    return user_features, item_features, popularity


def _active_days(time_aggregates, entity_id: int) -> float:
    if time_aggregates is None:
        return 0.0
    row = time_aggregates.loc[entity_id]
    return max(0.0, float(row["max"] - row["min"]) / 86_400.0)


def _recency_days(time_aggregates, entity_id: int, reference_timestamp: float) -> float:
    if time_aggregates is None:
        return 0.0
    return max(0.0, (reference_timestamp - float(time_aggregates.loc[entity_id, "max"])) / 86_400.0)


def build_ranker_dataset(
    retriever: MultiRouteRetriever,
    training_ratings: pd.DataFrame,
    validation_ratings: pd.DataFrame,
    user_features: dict[int, dict[str, float]],
    item_features: dict[int, dict[str, float]],
    *,
    positive_threshold: float = 4.0,
    negative_ratio: int = 4,
    random_negative_ratio: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    history_by_user = {
        int(user_id): list(group.sort_values("timestamp")["movieId"].astype(int))
        for user_id, group in training_ratings.groupby("userId")
    }
    positive_history_by_user = {
        int(user_id): list(
            group.loc[group["rating"] >= positive_threshold]
            .sort_values("timestamp")["movieId"]
            .astype(int)
        )
        for user_id, group in training_ratings.groupby("userId")
    }
    validation_positives = {
        int(user_id): set(group.loc[group["rating"] >= positive_threshold, "movieId"].astype(int))
        for user_id, group in validation_ratings.groupby("userId")
    }
    all_examples: list[RankingExample] = []
    all_labels: list[int] = []
    group_sizes = []
    for user_id in sorted(validation_positives):
        positives = validation_positives[user_id]
        if not positives or user_id not in history_by_user:
            continue
        candidates = retriever.retrieve(
            user_id,
            history_by_user[user_id],
            positive_history=positive_history_by_user[user_id],
        )
        group, labels = sample_ranking_group(
            candidates,
            positives,
            negative_ratio=negative_ratio,
            random_negative_items=set(item_features).difference(history_by_user[user_id]),
            random_negative_ratio=random_negative_ratio,
            seed=user_id,
        )
        if not group or labels.sum() == 0:
            continue
        all_examples.extend(
            RankingExample(
                candidate=candidate,
                user_features=user_features.get(user_id, {}),
                item_features=item_features.get(candidate.item_id, {}),
            )
            for candidate in group
        )
        all_labels.extend(int(label) for label in labels)
        group_sizes.append(len(group))
    if not group_sizes:
        raise ValueError("no validation positives were retrieved for ranker training")
    return (
        build_feature_matrix(all_examples),
        np.asarray(all_labels, dtype=np.int32),
        group_sizes,
    )
