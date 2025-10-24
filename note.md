## Data Storage
### S3 Raw Data
从Kafka的user-events中拿到用户评分，然后调用s3_storage里的save_raw_event，把原始Json保存到S3的bronze layer
### S3 Processed
每次Airflow定时运行

- 首先extract_features_batch，计算用户特征
- 然后dynamoDB进行update_movie_feature，update_user_feature。S3同时也把特征存储到silver layer，格式为parquet
- 接下来开始train_hybrid_model。首先build_user_item_matrix，再把matrix传给HybridRecommender让它初始化，这里面分别初始化两个模型。最后把model保存。
- evaluate_model（划分训练集和测试集、在训练集上重建模型、测试）
- deploy_model，如果质量检测通过
- reload,通过API来重新initialize_recommender（从文件路径加载模型）

### Feature Store
DynamoDB存储特征，支持快速的特征检索和更新

## Airflow
AWS Managed Workflows for Apache Airflow


