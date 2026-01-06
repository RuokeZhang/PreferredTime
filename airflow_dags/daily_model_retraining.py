"""
Airflow DAG: 每日模型重训练
从S3读取数据 → 计算特征 → 训练模型 → 更新DynamoDB → 部署模型
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.http_operator import SimpleHttpOperator
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, '/path/to/Movie_Rec')  # 需要修改为实际路径

from model_training.train_model import (
    validate_data,
    extract_features_batch,
    train_hybrid_model,
    evaluate_model,
    deploy_model
)

# 默认参数
default_args = {
    'owner': 'movie-rec-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 1),
    'email': ['alerts@movie-rec.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# 创建DAG
dag = DAG(
    'daily_model_retraining',
    default_args=default_args,
    description='每日重新训练推荐模型',
    schedule_interval='0 2 * * *',  # 每天凌晨2点运行
    catchup=False,
    tags=['ml', 'recommendation', 'batch']
)

# Task 1: 验证数据质量
validate_data_task = PythonOperator(
    task_id='validate_s3_data',
    python_callable=validate_data,
    op_kwargs={'date': '{{ ds }}'},  # Airflow模板变量，当前日期
    dag=dag,
)

# Task 2: 批量特征提取
extract_features_task = PythonOperator(
    task_id='extract_features',
    python_callable=extract_features_batch,
    op_kwargs={'date': '{{ ds }}'},
    provide_context=True,
    dag=dag,
)

# Task 3: 训练模型
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_hybrid_model,
    provide_context=True,
    dag=dag,
)

# Task 4: 评估模型
evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag,
)

# Task 5: 部署模型
deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    provide_context=True,
    dag=dag,
)

# Task 6: 重新加载API模型
reload_api_task = SimpleHttpOperator(
    task_id='reload_api',
    http_conn_id='movie_rec_api',
    endpoint='/reload',
    method='POST',
    headers={'Content-Type': 'application/json'},
    response_check=lambda response: response.status_code == 200,
    dag=dag,
)

# 定义任务依赖关系
validate_data_task >> extract_features_task >> train_model_task >> evaluate_model_task >> deploy_model_task >> reload_api_task



