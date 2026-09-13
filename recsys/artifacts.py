import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_model_version(training_cutoff: int, git_sha: str) -> str:
    short_sha = git_sha[:12]
    return f"{training_cutoff}-{short_sha}"


def publish_artifact(
    root: Path,
    model_type: str,
    version: str,
    payload: Any,
    metrics: dict[str, float],
) -> Path:
    model_root = root / model_type
    destination = model_root / version
    if destination.exists():
        raise FileExistsError(f"artifact already exists: {destination}")
    staging = model_root / f".{version}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with (staging / "model.pkl").open("wb") as file:
        pickle.dump(payload, file)
    metadata = {
        "model_type": model_type,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    (staging / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.rename(destination)
    return destination


def load_artifact(root: Path, model_type: str, version: str) -> tuple[Any, dict]:
    artifact_directory = root / model_type / version
    metadata = json.loads(
        (artifact_directory / "metadata.json").read_text(encoding="utf-8")
    )
    with (artifact_directory / "model.pkl").open("rb") as file:
        payload = pickle.load(file)
    return payload, metadata


def upload_artifact_to_s3(
    artifact_directory: Path,
    bucket: str,
    *,
    prefix: str = "models",
    client=None,
) -> None:
    if client is None:
        import boto3

        client = boto3.client("s3")
    relative_root = artifact_directory.parent.name + "/" + artifact_directory.name
    for path in artifact_directory.iterdir():
        if path.is_file():
            key = f"{prefix.rstrip('/')}/{relative_root}/{path.name}"
            client.upload_file(str(path), bucket, key)


def download_artifact_from_s3(
    root: Path,
    model_type: str,
    version: str,
    bucket: str,
    *,
    prefix: str = "models",
    client=None,
) -> Path:
    if client is None:
        import boto3

        client = boto3.client("s3")
    destination = root / model_type / version
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("model.pkl", "metadata.json"):
        key = f"{prefix.rstrip('/')}/{model_type}/{version}/{filename}"
        client.download_file(bucket, key, str(destination / filename))
    return destination


def passes_quality_gate(
    candidate_metrics: dict[str, float],
    production_metrics: dict[str, float],
    *,
    max_p99_ms: float = 100.0,
    allowed_regression: float = 0.02,
) -> bool:
    return (
        candidate_metrics["recall@20"]
        >= production_metrics["recall@20"] * (1.0 - allowed_regression)
        and candidate_metrics["ndcg@10"]
        >= production_metrics["ndcg@10"] * (1.0 - allowed_regression)
        and candidate_metrics["p99_ms"] <= max_p99_ms
        and candidate_metrics.get("feature_consistency", 0.0) == 1.0
    )
