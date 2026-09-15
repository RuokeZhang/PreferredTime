# Engineering log

## 2026-09-13 — legacy architecture audit

The repository sent one interaction through overlapping SQLite, S3, DynamoDB, Redis, Kafka, Kinesis, Lambda, and an in-process `/reload` path. Training and serving computed features through different classes, while the public description called the result production-grade without tests or comparative results.

The rewrite removed Kinesis, Lambda, DynamoDB, Redis, SQLite serving state, and hot reload. Retrieval, ranking, and reranking now share one artifact and one feature-matrix transform. The remaining Kafka path is explicitly optional until it has load-test evidence.

## 2026-09-13 — normalized-vector test failure

The content-vector test originally required a float32 vector norm to equal `1.0` at seven decimal places. NumPy returned `0.99999994`, so the test failed even though normalization was correct. The assertion now uses six decimal places, matching the representation's precision instead of hiding a real ranking error behind an unrealistic tolerance.

## 2026-09-13 — serving artifact was missing user history

The first versioned artifact contained the ranker, indexes, aggregate features, and popularity order but not the interaction history needed to exclude previously seen movies at serving time. Loading that artifact would either fail or recommend seen items. The experiment publisher now stores the exact pre-test history used by evaluation, and the API requires it when loading a version.

## 2026-09-13 — semantic index excluded cold-start items

The first retrieval builder intersected semantic vectors with the ALS item set. That made vector dimensions easy to reason about but silently removed every movie with no positive training interaction, defeating the semantic route's cold-start purpose. The semantic index now covers every movie with content features, independently of the collaborative index, and a regression test requires retrieval of a semantic-only item.

The same audit found that item-to-item and semantic profiles averaged every rated movie, including ratings below the positive threshold. Seen-item exclusion still needs all interactions, but interest profiles do not. Retrieval now receives both histories explicitly: all seen items for filtering and positive items for the query vector.

## 2026-09-15 — HNSW failed for users with long histories

The first MovieLens-25M smoke experiment failed when retrieval requested `per_route + seen_items` neighbors greater than the index's fixed `efSearch=200`. hnswlib could not produce the requested contiguous result array. Query execution now raises the search depth to at least the requested `k`, capped by catalog size, and CI covers the case where `k` is larger than the configured search depth.

Raising `efSearch` alone did not solve every query because the content index contains many duplicate and near-duplicate vectors. The index now falls back to exact cosine search only when hnswlib reports that it cannot return `k`; the fallback is counted and reported by the ANN benchmark so its latency cost cannot be hidden.

The next smoke attempt also showed that requesting `per_route + all_seen_items` neighbors makes ANN work grow with a user's lifetime history. Retrieval now caps oversampling at three times the per-route target; route shortfalls are filled by the other routes and popularity instead of allowing a heavy user to force a multi-thousand-neighbor query.

## 2026-09-15 — LightGBM import passed but ranker construction failed

The smoke experiment reached `LGBMRanker` construction and failed because LightGBM's sklearn API requires scikit-learn. Earlier unit tests only exercised feature construction and a fake ranker, so importing `lightgbm` was not enough to catch the missing runtime dependency. scikit-learn is now explicit in `requirements.txt`, and CI fits and predicts with a small grouped LambdaRank dataset.

## 2026-09-15 — MMR dominated serving latency

The first successful 100-user experiment measured MMR at 53.4 ms P50 and 77.0 ms P99, pushing end-to-end P99 to 121.2 ms. The implementation recomputed pairwise cosine similarity in nested Python loops, and even the `lambda=1` ranking-only baseline paid that cost. MMR now computes one normalized similarity matrix with NumPy and updates the maximum redundancy vector incrementally; `lambda=1` bypasses similarity work entirely.

## 2026-09-15 — full MovieLens-25M experiment

The final run evaluated all 3,946 eligible test users with 32-factor ALS, 10 iterations, 10,000 LambdaRank training users, and HNSW retrieval. The MMR system reached 0.1177 Recall@20 and 0.1608 NDCG@10 at 19.18 ms end-to-end P99, compared with 0.0369 and 0.0647 for popularity. The versioned local artifact is `lambdarank/1515033900-1c96ad1185f4` and occupies 424 MB.

An independent 200-query benchmark at `k=100` and `ef_search=200` measured HNSW Recall@100 at 0.9885. HNSW P99 was 0.38 ms versus 2.63 ms for exact cosine search, and no query required the exact-search fallback.
