import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


project_root = os.environ.get("PREFERREDTIME_PROJECT_ROOT", "/opt/preferredtime")
data_root = os.environ.get("MOVIELENS_DATA_ROOT", "/data/ml-25m")
artifact_root = os.environ.get("MODEL_ARTIFACT_ROOT", "/artifacts")

with DAG(
    dag_id="preferredtime_daily_training",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    default_args={
        "owner": "preferredtime",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["recommendation", "batch", "movielens"],
) as dag:
    validate_inputs = BashOperator(
        task_id="validate_inputs",
        bash_command=(
            f"test -f {data_root}/ratings.csv "
            f"&& test -f {data_root}/movies.csv "
            f"&& test -f {data_root}/genome-scores.csv"
        ),
    )
    train_evaluate_publish = BashOperator(
        task_id="train_evaluate_publish",
        cwd=project_root,
        bash_command=(
            "python3 -m scripts.run_offline_experiment "
            f"--ratings {data_root}/ratings.csv "
            f"--movies {data_root}/movies.csv "
            f"--genome-scores {data_root}/genome-scores.csv "
            f"--output {artifact_root} --backend hnsw"
        ),
    )

    validate_inputs >> train_evaluate_publish
