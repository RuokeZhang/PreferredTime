import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from recsys.data import build_temporal_split
from recsys.evaluation import evaluate_recommendations, ndcg_at_k, recall_at_k
from recsys.pipeline import RecommendationPipeline
from recsys.ranking import RankingExample, build_feature_matrix, sample_ranking_group
from recsys.reranking import mmr_rerank
from recsys.retrieval import Candidate, MultiRouteRetriever
from recsys.artifacts import (
    build_model_version,
    download_artifact_from_s3,
    load_artifact,
    passes_quality_gate,
    publish_artifact,
    upload_artifact_to_s3,
)
from recsys.training import build_content_vectors, build_training_features


class TemporalSplitTests(unittest.TestCase):
    def test_global_split_preserves_time_and_filters_evaluation_users(self):
        ratings = pd.DataFrame(
            [
                {
                    "userId": 1,
                    "movieId": timestamp,
                    "rating": 5.0,
                    "timestamp": timestamp,
                }
                for timestamp in range(1, 21)
            ]
            + [
                {"userId": 2, "movieId": 100, "rating": 5.0, "timestamp": 1},
                {"userId": 2, "movieId": 101, "rating": 5.0, "timestamp": 20},
            ]
        )
        split = build_temporal_split(ratings, min_history=5)

        self.assertLess(split.train["timestamp"].max(), split.validation_cutoff)
        self.assertLess(split.validation["timestamp"].max(), split.test_cutoff)
        self.assertGreaterEqual(split.test["timestamp"].min(), split.test_cutoff)
        self.assertEqual(split.evaluation_users, (1,))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.collaborative_vectors = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.9, 0.1]),
            3: np.array([0.0, 1.0]),
            4: np.array([0.1, 0.9]),
        }
        self.semantic_vectors = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.8, 0.2]),
            3: np.array([0.0, 1.0]),
            4: np.array([0.2, 0.8]),
        }

    def test_multi_route_retrieval_excludes_history_and_caps_candidates(self):
        retriever = MultiRouteRetriever(
            user_vectors={7: np.array([1.0, 0.0])},
            collaborative_item_vectors=self.collaborative_vectors,
            semantic_item_vectors=self.semantic_vectors,
            popularity=[3, 4, 2, 1],
        )
        candidates = retriever.retrieve(7, [1], per_route=3, max_candidates=2)

        self.assertEqual(len(candidates), 2)
        self.assertNotIn(1, {candidate.item_id for candidate in candidates})
        self.assertTrue(any("u2i" in candidate.source_scores for candidate in candidates))

    def test_semantic_route_can_retrieve_items_without_collaborative_history(self):
        semantic_vectors = dict(self.semantic_vectors)
        semantic_vectors[5] = np.array([0.95, 0.05])
        retriever = MultiRouteRetriever(
            user_vectors={},
            collaborative_item_vectors=self.collaborative_vectors,
            semantic_item_vectors=semantic_vectors,
            popularity=[1, 2, 3, 4],
            enabled_routes=("semantic",),
        )

        candidates = retriever.retrieve(7, [1], per_route=2)

        self.assertIn(5, {candidate.item_id for candidate in candidates})

    def test_semantic_profile_uses_positive_history_but_excludes_all_seen_items(self):
        semantic_vectors = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.0, 1.0]),
            3: np.array([0.95, 0.05]),
            4: np.array([0.05, 0.95]),
        }
        retriever = MultiRouteRetriever(
            user_vectors={},
            collaborative_item_vectors=self.collaborative_vectors,
            semantic_item_vectors=semantic_vectors,
            popularity=[],
            enabled_routes=("semantic",),
        )

        candidates = retriever.retrieve(
            7, [1, 2], positive_history=[1], per_route=1
        )

        self.assertEqual([candidate.item_id for candidate in candidates], [3])

    def test_sampled_ranking_group_uses_hard_negative_ratio(self):
        candidates = [Candidate(item_id=item_id) for item_id in range(1, 10)]
        group, labels = sample_ranking_group(
            candidates, {1, 2}, negative_ratio=2, seed=3
        )

        self.assertEqual(len(group), 6)
        self.assertEqual(int(labels.sum()), 2)

    def test_ranking_group_mixes_hard_and_random_negatives(self):
        candidates = [Candidate(item_id=item_id) for item_id in range(1, 8)]
        group, labels = sample_ranking_group(
            candidates,
            {1},
            negative_ratio=4,
            random_negative_items={8, 9, 10},
            random_negative_ratio=1,
            seed=3,
        )

        self.assertEqual(len(group), 5)
        self.assertEqual(int(labels.sum()), 1)
        self.assertTrue({candidate.item_id for candidate in group}.intersection({8, 9, 10}))

    def test_training_and_serving_share_deterministic_feature_transform(self):
        example = RankingExample(
            candidate=Candidate(
                item_id=8,
                source_scores={"u2i": 0.7},
                source_ranks={"u2i": 2},
            ),
            user_features={"interaction_count": 5.0, "rating_std": 0.8},
            item_features={"popularity": 0.4, "recency_days": 3.0},
        )

        offline_features = build_feature_matrix([example])
        serving_features = build_feature_matrix([example])

        np.testing.assert_allclose(offline_features, serving_features, rtol=1e-6)


class RerankingAndEvaluationTests(unittest.TestCase):
    def test_mmr_trades_relevance_for_diversity(self):
        item_vectors = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.99, 0.01]),
            3: np.array([0.0, 1.0]),
        }
        reranked = mmr_rerank(
            [1, 2, 3],
            {1: 1.0, 2: 0.99, 3: 0.8},
            item_vectors,
            limit=2,
            relevance_weight=0.5,
        )

        self.assertEqual(reranked, [1, 3])

    def test_offline_metrics_use_full_relevant_sets(self):
        self.assertEqual(recall_at_k([1, 2], {2, 3}, 2), 0.5)
        self.assertGreater(ndcg_at_k([2, 1], {2, 3}, 2), 0.5)
        metrics = evaluate_recommendations(
            {1: [1, 2], 2: [3, 4]},
            {1: {2}, 2: {3}},
            {
                1: np.array([1.0, 0.0]),
                2: np.array([0.0, 1.0]),
                3: np.array([1.0, 0.0]),
                4: np.array([0.0, 1.0]),
            },
            catalog_size=8,
            recall_k=2,
            ndcg_k=2,
        )

        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertEqual(metrics["coverage"], 0.5)

    def test_pipeline_reports_stage_latency_and_model_version(self):
        vectors = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.8, 0.2]),
            3: np.array([0.0, 1.0]),
        }
        retriever = MultiRouteRetriever(
            user_vectors={1: np.array([1.0, 0.0])},
            collaborative_item_vectors=vectors,
            semantic_item_vectors=vectors,
            popularity=[3, 2, 1],
        )
        pipeline = RecommendationPipeline(
            retriever=retriever,
            item_features={},
            item_vectors=vectors,
            model_version="2026-09-13-deadbeef",
        )
        result = pipeline.recommend(1, [1], {}, limit=2)

        self.assertEqual(len(result.item_ids), 2)
        self.assertEqual(result.model_version, "2026-09-13-deadbeef")
        self.assertIn("retrieval", result.stage_latency_ms)
        self.assertGreater(result.route_candidate_counts["popularity"], 0)
        self.assertNotIn(1, result.item_ids)


class TrainingAndArtifactTests(unittest.TestCase):
    def test_content_vectors_combine_genres_and_genome_scores(self):
        movies = pd.DataFrame(
            [
                {"movieId": 1, "genres": "Action|Drama"},
                {"movieId": 2, "genres": "Drama"},
            ]
        )
        genome_scores = pd.DataFrame(
            [
                {"movieId": 1, "tagId": 10, "relevance": 0.8},
                {"movieId": 2, "tagId": 10, "relevance": 0.2},
            ]
        )
        vectors = build_content_vectors(movies, genome_scores)

        self.assertEqual(set(vectors), {1, 2})
        self.assertAlmostEqual(float(np.linalg.norm(vectors[1])), 1.0, places=6)
        self.assertEqual(vectors[1].shape, vectors[2].shape)

    def test_training_features_only_use_supplied_ratings(self):
        ratings = pd.DataFrame(
            [
                {"userId": 1, "movieId": 10, "rating": 5.0},
                {"userId": 1, "movieId": 11, "rating": 3.0},
                {"userId": 2, "movieId": 10, "rating": 4.0},
            ]
        )
        user_features, item_features, popularity = build_training_features(ratings)

        self.assertEqual(user_features[1]["interaction_count"], 2.0)
        self.assertAlmostEqual(user_features[1]["rating_std"], np.sqrt(2.0))
        self.assertEqual(item_features[10]["interaction_count"], 2.0)
        self.assertEqual(popularity[0], 10)

    def test_artifacts_are_versioned_and_quality_gate_is_relative(self):
        version = build_model_version(123456, "deadbeefcafebabefeed")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            publish_artifact(
                root,
                "lambdarank",
                version,
                {"weights": [1, 2]},
                {"recall@20": 0.3},
            )
            payload, metadata = load_artifact(root, "lambdarank", version)

        self.assertEqual(payload, {"weights": [1, 2]})
        self.assertEqual(metadata["version"], version)
        self.assertTrue(
            passes_quality_gate(
                {
                    "recall@20": 0.294,
                    "ndcg@10": 0.196,
                    "p99_ms": 99.0,
                    "feature_consistency": 1.0,
                },
                {"recall@20": 0.3, "ndcg@10": 0.2},
            )
        )

    def test_s3_artifact_paths_include_model_type_and_version(self):
        class FakeS3Client:
            def __init__(self):
                self.uploads = []
                self.downloads = []

            def upload_file(self, source, bucket, key):
                self.uploads.append((source, bucket, key))

            def download_file(self, bucket, key, destination):
                self.downloads.append((bucket, key, destination))
                Path(destination).write_bytes(b"artifact")

        client = FakeS3Client()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "lambdarank" / "123-deadbeef"
            artifact.mkdir(parents=True)
            (artifact / "model.pkl").write_bytes(b"model")
            (artifact / "metadata.json").write_text("{}", encoding="utf-8")
            upload_artifact_to_s3(artifact, "models-bucket", client=client)
            download_artifact_from_s3(
                root / "downloaded",
                "lambdarank",
                "123-deadbeef",
                "models-bucket",
                client=client,
            )

        uploaded_keys = {upload[2] for upload in client.uploads}
        downloaded_keys = {download[1] for download in client.downloads}
        expected_keys = {
            "models/lambdarank/123-deadbeef/model.pkl",
            "models/lambdarank/123-deadbeef/metadata.json",
        }
        self.assertEqual(uploaded_keys, expected_keys)
        self.assertEqual(downloaded_keys, expected_keys)


if __name__ == "__main__":
    unittest.main()
