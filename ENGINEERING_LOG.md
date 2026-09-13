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
