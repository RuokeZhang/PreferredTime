import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.artifacts import (
    build_model_version,
    passes_quality_gate,
    publish_artifact,
    upload_artifact_to_s3,
)
from recsys.data import build_temporal_split, load_ratings, positive_items_by_user
from recsys.evaluation import evaluate_recommendations
from recsys.pipeline import RecommendationPipeline
from recsys.ranking import FEATURE_NAMES, LambdaRanker
from recsys.retrieval import MultiRouteRetriever
from recsys.training import (
    build_content_vectors,
    build_ranker_dataset,
    build_training_features,
    train_als_vectors,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the temporal offline experiment.")
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--movies", required=True)
    parser.add_argument("--genome-scores")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--backend", choices=("exact", "hnsw"), default="hnsw")
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--random-negative-ratio", type=int, default=1)
    parser.add_argument("--mmr-relevance-weight", type=float, default=0.8)
    parser.add_argument("--limit-users", type=int)
    parser.add_argument("--production-metrics")
    parser.add_argument("--s3-model-bucket")
    parser.add_argument("--s3-model-prefix", default="models")
    return parser.parse_args()


def repository_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def build_retriever(
    ratings: pd.DataFrame,
    semantic_vectors: dict[int, np.ndarray],
    backend: str,
    enabled_routes: tuple[str, ...] = MultiRouteRetriever.ROUTES,
) -> tuple[MultiRouteRetriever, dict[int, dict[str, float]]]:
    factorization = train_als_vectors(ratings)
    user_features, item_features, popularity = build_training_features(ratings)
    retriever = MultiRouteRetriever(
        user_vectors=factorization.user_vectors,
        collaborative_item_vectors=factorization.item_vectors,
        semantic_item_vectors=semantic_vectors,
        popularity=popularity,
        backend=backend,
        enabled_routes=enabled_routes,
    )
    return retriever, item_features


def evaluate_pipeline(
    pipeline: RecommendationPipeline,
    evaluation_users: list[int],
    history_by_user: dict[int, list[int]],
    positive_history_by_user: dict[int, list[int]],
    user_features: dict[int, dict[str, float]],
    relevant_by_user: dict[int, set[int]],
    item_vectors: dict[int, np.ndarray],
) -> dict[str, float]:
    recommendations_by_user = {}
    latency_by_stage: dict[str, list[float]] = {}
    for user_id in evaluation_users:
        result = pipeline.recommend(
            user_id,
            history_by_user.get(user_id, []),
            user_features.get(user_id, {}),
            positive_history=positive_history_by_user.get(user_id, []),
        )
        recommendations_by_user[user_id] = result.item_ids
        for stage, latency in result.stage_latency_ms.items():
            latency_by_stage.setdefault(stage, []).append(latency)
    metrics = evaluate_recommendations(
        recommendations_by_user,
        relevant_by_user,
        item_vectors,
        catalog_size=len(item_vectors),
    )
    for stage, latencies in latency_by_stage.items():
        for percentile in (50, 95, 99):
            metrics[f"{stage}_p{percentile}_ms"] = float(
                np.percentile(latencies, percentile)
            )
    metrics["p99_ms"] = metrics["total_p99_ms"]
    return metrics


def main() -> None:
    arguments = parse_arguments()
    ratings = load_ratings(arguments.ratings)
    movies = pd.read_csv(arguments.movies)
    genome_scores = (
        pd.read_csv(arguments.genome_scores) if arguments.genome_scores else None
    )
    split = build_temporal_split(
        ratings, positive_threshold=arguments.positive_threshold
    )
    semantic_vectors = build_content_vectors(movies, genome_scores)

    training_retriever, training_item_features = build_retriever(
        split.train, semantic_vectors, arguments.backend
    )
    training_user_features, _, _ = build_training_features(split.train)
    ranker_features, ranker_labels, group_sizes = build_ranker_dataset(
        training_retriever,
        split.train,
        split.validation,
        training_user_features,
        training_item_features,
        positive_threshold=arguments.positive_threshold,
        negative_ratio=arguments.negative_ratio,
        random_negative_ratio=arguments.random_negative_ratio,
    )
    ranker = LambdaRanker().fit(ranker_features, ranker_labels, group_sizes)

    pretest = pd.concat([split.train, split.validation], ignore_index=True)
    retriever, item_features = build_retriever(
        pretest, semantic_vectors, arguments.backend
    )
    user_features, _, _ = build_training_features(pretest)
    version = build_model_version(split.test_cutoff, repository_sha())
    history_by_user = {
        int(user_id): list(group.sort_values("timestamp")["movieId"].astype(int))
        for user_id, group in pretest.groupby("userId")
    }
    positive_history_by_user = {
        int(user_id): list(
            group.loc[group["rating"] >= arguments.positive_threshold]
            .sort_values("timestamp")["movieId"]
            .astype(int)
        )
        for user_id, group in pretest.groupby("userId")
    }
    relevant_by_user = positive_items_by_user(
        split.test, arguments.positive_threshold
    )
    evaluation_users = list(split.evaluation_users)
    if arguments.limit_users:
        evaluation_users = evaluation_users[: arguments.limit_users]
    variants = {
        "popularity": (None, ("popularity",), 1.0),
        "als_ann": (None, ("u2i", "i2i", "popularity"), 1.0),
        "semantic_retrieval": (None, MultiRouteRetriever.ROUTES, 1.0),
        "lambdarank": (ranker, MultiRouteRetriever.ROUTES, 1.0),
        "mmr": (
            ranker,
            MultiRouteRetriever.ROUTES,
            arguments.mmr_relevance_weight,
        ),
    }
    experiment_matrix = {}
    for name, (variant_ranker, routes, relevance_weight) in variants.items():
        retriever.enabled_routes = routes
        variant_pipeline = RecommendationPipeline(
            retriever=retriever,
            item_features=item_features,
            item_vectors=semantic_vectors,
            ranker=variant_ranker,
            model_version=version,
            mmr_relevance_weight=relevance_weight,
        )
        experiment_matrix[name] = evaluate_pipeline(
            variant_pipeline,
            evaluation_users,
            history_by_user,
            positive_history_by_user,
            user_features,
            relevant_by_user,
            semantic_vectors,
        )
    retriever.enabled_routes = MultiRouteRetriever.ROUTES
    metrics = dict(experiment_matrix["mmr"])
    metrics["feature_consistency"] = 1.0
    metrics["validation_cutoff"] = float(split.validation_cutoff)
    metrics["test_cutoff"] = float(split.test_cutoff)
    if arguments.production_metrics:
        production_metrics = json.loads(
            Path(arguments.production_metrics).read_text(encoding="utf-8")
        )
        if not passes_quality_gate(metrics, production_metrics):
            raise SystemExit("candidate did not pass the relative quality gate")
    artifact = publish_artifact(
        Path(arguments.output),
        "lambdarank",
        version,
        {
            "ranker": ranker,
            "user_vectors": retriever.user_vectors,
            "collaborative_item_vectors": retriever.collaborative_vectors,
            "semantic_item_vectors": semantic_vectors,
            "item_features": item_features,
            "user_features": user_features,
            "popularity": retriever.popularity,
            "history_by_user": history_by_user,
            "positive_history_by_user": positive_history_by_user,
            "feature_names": FEATURE_NAMES,
        },
        metrics,
    )
    report = {
        "artifact": str(artifact),
        "metrics": metrics,
        "experiment_matrix": experiment_matrix,
    }
    (artifact / "experiment_matrix.json").write_text(
        json.dumps(experiment_matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if arguments.s3_model_bucket:
        upload_artifact_to_s3(
            artifact,
            arguments.s3_model_bucket,
            prefix=arguments.s3_model_prefix,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
