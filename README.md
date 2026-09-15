# PreferredTime

An offline-first movie recommendation system for measuring the trade-offs among retrieval, ranking, diversity, and CPU latency on MovieLens-25M.

## Experiment matrix

Measured on the full MovieLens-25M global temporal test split with 3,946 eligible users. The run used commit `1c96ad1`, 32 ALS factors, 10 ALS iterations, 10,000 LambdaRank training users, and HNSW retrieval.

| Variant | Recall@20 | NDCG@10 | Coverage | Diversity | P99 (ms) |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.0369 | 0.0647 | 0.0075 | 0.3116 | 5.51 |
| ALS + ANN | 0.0628 | 0.0826 | 0.1360 | 0.3850 | 13.81 |
| + semantic retrieval | 0.0832 | 0.1121 | 0.1193 | 0.3071 | 26.74 |
| + LightGBM LambdaRank | 0.1181 | 0.1610 | 0.0337 | 0.2543 | 24.73 |
| + MMR | 0.1177 | 0.1608 | 0.0343 | 0.2560 | 19.18 |

The final system improves Recall@20 by 218.7% and NDCG@10 by 148.6% over popularity. MMR trades 0.3% relative Recall@20 for a 0.7% relative diversity gain compared with LambdaRank alone. `scripts/run_offline_experiment.py` writes unrounded values to `experiment_matrix.json` inside the versioned artifact directory.

Measure ANN approximation loss and latency against exact cosine search with:

```bash
python3 -m scripts.benchmark_ann --model-version <training_cutoff>-<git_sha>
```

For the published run, HNSW at `ef_search=200` achieved Recall@100 of 0.9885 over 200 queries. Its P50/P99 latency was 0.22/0.38 ms versus 2.31/2.63 ms for exact cosine search, with zero exact-search fallbacks.

## Implemented pipeline

1. Ratings are converted to implicit positives at a configurable threshold and split by global timestamps. The validation and test windows are never included in training aggregates.
2. Implicit ALS produces user and item embeddings. Exact cosine or hnswlib indexes support user-to-item and item-to-item retrieval.
3. Genre and MovieLens genome vectors form an independent semantic route. Popularity provides a cold-user fallback.
4. Reciprocal-rank fusion deduplicates all routes and caps the ranker input at 500 candidates.
5. LightGBM LambdaRank scores candidates by user, item, and retrieval-route features. Negatives are sampled from retrieved but unobserved items.
6. MMR reranks the top 100 into a top-20 list while explicitly trading relevance for intra-list diversity.
7. Offline evaluation ranks over the retrievable catalog after removing training history; it does not use a sampled 100-negative test set.

The service loads one immutable `{model_type}/{training_cutoff}-{git_sha}` artifact at startup. Requests report the model version, candidate count, per-route candidate counts, and retrieval/ranking/reranking latency. Switching versions requires an instance restart; there is no process-local `/reload` endpoint.

## Reproduce the experiment

Download MovieLens-25M and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m scripts.run_offline_experiment \
  --ratings data/ml-25m/ratings.csv \
  --movies data/ml-25m/movies.csv \
  --genome-scores data/ml-25m/genome-scores.csv \
  --output artifacts \
  --backend hnsw \
  --als-factors 32 \
  --als-iterations 10 \
  --als-threads 4 \
  --ranker-training-users 10000
```

This command produced `lambdarank/1515033900-1c96ad1185f4`; the validation and test cutoffs were 2016-06-25 and 2018-01-04 UTC. Add `--s3-model-bucket preferredtime-models` when S3 publication is configured.

For a quick pipeline check, add `--limit-users 100`. That option is for development only and must not be used for the final reported matrix.

LambdaRank training uses up to 10,000 validation users by default so the sampled hard-negative matrix stays within a single-machine memory budget. Change it with `--ranker-training-users`; evaluation still uses every eligible test user unless `--limit-users` is explicitly supplied.

To compare a candidate with production, pass a JSON file containing the production metrics:

```bash
python3 -m scripts.run_offline_experiment \
  --ratings data/ml-25m/ratings.csv \
  --movies data/ml-25m/movies.csv \
  --genome-scores data/ml-25m/genome-scores.csv \
  --production-metrics artifacts/production-metrics.json
```

Publication is blocked if Recall@20 or NDCG@10 regresses by more than 2%, end-to-end P99 exceeds 100 ms, or the shared feature transform is not covered by the consistency check.

## Serve a version

```bash
export MODEL_ROOT=artifacts
export MODEL_TYPE=lambdarank
export MODEL_VERSION=<training_cutoff>-<git_sha>
export MODEL_S3_BUCKET=preferredtime-models
uvicorn api.main:app --host 0.0.0.0 --port 8082
```

Endpoints:

- `GET /health/live`: process health
- `GET /health/ready`: artifact readiness and active model version
- `GET /v1/recommendations?user_id=1&limit=20`: synchronous recommendation
- `POST /v1/events`: optional Kafka event ingestion

Set `KAFKA_PRODUCER_ENABLED=true` to enable the event endpoint. It is disabled by default so a missing broker cannot delay model serving startup.

## Batch and event path

The Airflow DAG validates the MovieLens inputs and runs the deterministic train → evaluate → publish command daily. Model artifacts are versioned; the DAG does not mutate a running process.

The optional event path is FastAPI → Kafka → consumer → date-partitioned S3 JSON. Kafka auto-commit is disabled. The consumer writes `event_id` as the S3 object key and commits the offset only after the write succeeds, giving at-least-once delivery with idempotent downstream storage. `config/config.yaml` specifies three partitions as the intended upper bound for consumer parallelism; throughput, lag, rebalance recovery, and partition selection still require a recorded load test before this path should be described as production-ready.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

CI checks temporal isolation, full-catalog metrics, route fusion, hard-negative sampling, feature transforms, MMR behavior, versioned artifacts, and relative quality gates.

## Non-goals

- Online A/B testing: there is no real user traffic.
- Second-level online feature refresh: MovieLens is an offline dataset; batch refresh is sufficient.
- Distributed or GPU training: MovieLens-25M fits on one machine.
- Multi-objective ranking: ratings provide only one meaningful target.
- Multi-region availability, failover, or speculative caching.

## Known limitations

- MovieLens has no exposure log. Ranker negatives are sampled, so the project cannot estimate or correct position and exposure bias.
- Every quality conclusion is offline until a real traffic source exists.
- Genre and genome features do not provide the full-catalog semantic coverage planned for TMDB overview tagging.
- The controlled-vocabulary LLM tagger, genome-agreement study, sequential-ranker comparison, threshold sensitivity sweep, MMR lambda curve, and ANN parameter sweep are specified follow-up experiments, not completed results. The reported ANN result covers one configuration.
- The optional Kafka path has not yet earned a production claim; its benchmark questions are listed above.

See [ENGINEERING_LOG.md](ENGINEERING_LOG.md) for failures and fixes observed during implementation.
