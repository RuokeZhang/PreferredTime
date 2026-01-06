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


## Kafka
```yaml
kafka:
  bootstrap_servers: "localhost:9092"
  topic: "user-events"
  group_id: "movie-rec-consumer-group"
  auto_offset_reset: "earliest"
```
### 消费
1. 新建KafkaProducer
    ```py
    kafka_producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3,
            linger_ms=20
        )
    ```
2. 从API接收user events, ```ingest_user_event```
    ```py
    future = kafka_producer.send(kafka_topic, payload)
    record = future.get(timeout=2)  # 等待broker ack
    ```
3. 新建一个KafkaConsumer， 配置其订阅的topic和监听的server，以及定义它的group_id（真正的开始）
4. ```for message in self.consumer:```持续拉取消息，并对每个消息里传来的event做```process_event```
    - 一个event包含user_id, movie_id, rating, timestamp
    - save_rating, S3实现中，用save_raw_event把原始event json保存到S3，（按日期分区）
5. Airflow批处理任务 validate_data >> extract_features_task >> train_model_task >> evaluate_model_task >> deploy_model_task >> reload_api_task
    - validate_data: 从S3拿到当天的数据，进行检验
    - extract_features_batch: 从S3拿到所有rating，分别计算用户和电影的特征。update_user_feature，把每个user的特征更新到dynamoDB
    - train_hybrid_model
        - build_user_item_matrix：拿到所有的rating，做一个列为movie_id,行为user_id的matrix
        - 训练模型HybridRecommender（CollaborativeFiltering+ContentBasedRecommender）。CollaborativeFiltering其实在初始化阶段并没有生成相似度矩阵，只是存了这个rating matrix; ContentBasedRecommender在初始化阶段预计算了电影的相似度矩阵
    - evaluate_model：划分测试集，为测试用户生成推荐
    - 调用movie_rec_api中的/reload,用保存好的模型pkl文件直接替换掉recommender
6. user-based recommend
    - 计算用户相似度列表。对于每个用户，遍历其它所有用户，找到共同评分过的电影，然后算cosine similarity. 对于该用户，最后返回相似度高于某个阈值的用户列表，相似度降序排序
    - 遍历所有电影，对于每个电影，拿出相似用户给他们的评分，用similarity做weighted rating，则可以得到该用户对此电影的评分
7. content-based：用电影的metadata做相似度矩阵
8. rerank
    - quality score=流行度*0.4 + 平均评分*0.6
    - final score=recall_score*(1-quality weight) + quality score*quality weight

recommend具体推荐的时候可以指定是user-based,还是item-based