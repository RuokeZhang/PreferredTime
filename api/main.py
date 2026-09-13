import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Query
from kafka import KafkaProducer
from pydantic import BaseModel, Field

from recsys.artifacts import download_artifact_from_s3, load_artifact
from recsys.pipeline import RecommendationPipeline
from recsys.ranking import FEATURE_NAMES
from recsys.retrieval import MultiRouteRetriever


app = FastAPI(title="PreferredTime Recommendation API", version="2.0.0")
pipeline: RecommendationPipeline | None = None
history_by_user: dict[int, list[int]] = {}
positive_history_by_user: dict[int, list[int]] = {}
user_features: dict[int, dict[str, float]] = {}
artifact_metadata: dict = {}
kafka_producer: KafkaProducer | None = None
kafka_topic = "user-events"


class UserEvent(BaseModel):
    event_id: str = Field(..., min_length=1)
    user_id: int = Field(..., ge=1)
    movie_id: int = Field(..., ge=1)
    rating: float = Field(..., ge=0.0, le=5.0)
    timestamp: str | None = None


def load_config() -> dict:
    config_path = Path(__file__).parents[1] / "config" / "config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def initialize_pipeline() -> None:
    global pipeline, history_by_user, positive_history_by_user, user_features
    global artifact_metadata

    model_version = os.environ.get("MODEL_VERSION")
    if not model_version:
        pipeline = None
        return
    model_root = Path(os.environ.get("MODEL_ROOT", "artifacts"))
    model_type = os.environ.get("MODEL_TYPE", "lambdarank")
    model_bucket = os.environ.get("MODEL_S3_BUCKET")
    if model_bucket:
        download_artifact_from_s3(
            model_root,
            model_type,
            model_version,
            model_bucket,
            prefix=os.environ.get("MODEL_S3_PREFIX", "models"),
        )
    payload, artifact_metadata = load_artifact(model_root, model_type, model_version)
    if tuple(payload["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("artifact feature schema does not match serving code")
    backend = os.environ.get("VECTOR_INDEX_BACKEND", "hnsw")
    model_config = load_config()["model"]
    retriever = MultiRouteRetriever(
        user_vectors=payload["user_vectors"],
        collaborative_item_vectors=payload["collaborative_item_vectors"],
        semantic_item_vectors=payload["semantic_item_vectors"],
        popularity=payload["popularity"],
        backend=backend,
    )
    pipeline = RecommendationPipeline(
        retriever=retriever,
        item_features=payload["item_features"],
        item_vectors=payload["semantic_item_vectors"],
        ranker=payload["ranker"],
        model_version=model_version,
        max_candidates=model_config["max_candidates"],
        mmr_relevance_weight=model_config["mmr_relevance_weight"],
    )
    history_by_user = payload["history_by_user"]
    positive_history_by_user = payload["positive_history_by_user"]
    user_features = payload["user_features"]


def initialize_kafka() -> None:
    global kafka_producer, kafka_topic

    if os.environ.get("KAFKA_PRODUCER_ENABLED", "false").lower() != "true":
        return
    config = load_config()["kafka"]
    kafka_topic = os.environ.get("KAFKA_TOPIC", config["topic"])
    bootstrap_servers = os.environ.get(
        "KAFKA_BOOTSTRAP_SERVERS", config["bootstrap_servers"]
    )
    try:
        kafka_producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            acks="all",
            retries=3,
            linger_ms=20,
        )
    except Exception:
        kafka_producer = None


@app.on_event("startup")
async def startup_event() -> None:
    initialize_pipeline()
    initialize_kafka()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if kafka_producer is not None:
        kafka_producer.flush(timeout=3)
        kafka_producer.close()


@app.get("/")
async def root() -> dict:
    return {
        "service": "PreferredTime Recommendation API",
        "version": "2.0.0",
        "model_version": None if pipeline is None else pipeline.model_version,
    }


@app.get("/health/live")
async def liveness() -> dict:
    return {"status": "live"}


@app.get("/health/ready")
async def readiness() -> dict:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="model artifact is not loaded")
    return {
        "status": "ready",
        "model_version": pipeline.model_version,
        "artifact_created_at": artifact_metadata.get("created_at"),
    }


@app.post("/v1/events", status_code=202)
async def ingest_event(event: UserEvent) -> dict:
    if kafka_producer is None:
        raise HTTPException(status_code=503, detail="Kafka producer is unavailable")
    payload = event.model_dump()
    payload["timestamp"] = payload["timestamp"] or datetime.now(timezone.utc).isoformat()
    try:
        metadata = kafka_producer.send(kafka_topic, payload).get(timeout=3)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Kafka publish failed") from error
    return {
        "accepted": True,
        "event_id": event.event_id,
        "partition": metadata.partition,
        "offset": metadata.offset,
    }


@app.get("/v1/recommendations")
async def recommend(
    user_id: int = Query(..., ge=1),
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="model artifact is not loaded")
    result = pipeline.recommend(
        user_id,
        history_by_user.get(user_id, []),
        user_features.get(user_id, {}),
        positive_history=positive_history_by_user.get(user_id, []),
        limit=limit,
    )
    return {
        "user_id": user_id,
        "movie_ids": result.item_ids,
        "model_version": result.model_version,
        "candidate_count": result.candidate_count,
        "route_candidate_counts": result.route_candidate_counts,
        "stage_latency_ms": result.stage_latency_ms,
    }
