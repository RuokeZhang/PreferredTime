import argparse
import json
from pathlib import Path

from recsys.artifacts import load_artifact
from recsys.evaluation import benchmark_ann


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare hnswlib retrieval with exact cosine search."
    )
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--model-type", default="lambdarank")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    payload, _ = load_artifact(
        Path(arguments.artifact_root),
        arguments.model_type,
        arguments.model_version,
    )
    query_vectors = list(payload["user_vectors"].values())[: arguments.queries]
    metrics = benchmark_ann(
        payload["collaborative_item_vectors"],
        query_vectors,
        k=arguments.k,
        ef_search=arguments.ef_search,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
