# Movie Recommendation Platform

一个基于混合推荐算法的电影推荐平台，结合协同过滤和基于内容的推荐方法。

## 项目简介

本项目实现了一个完整的**生产级**电影推荐系统，包括：
- **事件驱动架构**: Kinesis Stream + Lambda触发推荐
- **高性能缓存**: ElastiCache (Redis)
- **容器化部署**: Docker + ECR + ECS Fargate
- **实时数据流**: Kafka处理用户行为事件
- **混合推荐模型**: 协同过滤 + 基于内容推荐
- **自动化训练**: Airflow调度的每日模型更新
- **完整评估体系**: Precision/Recall/NDCG等业界标准指标

## 流程
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
2. 接收user events, ```ingest_user_event```
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


## 模型评估
### 质量阈值

模型必须满足以下最低要求才能部署到生产环境：

| 指标 | 最低要求 |
|------|---------|
| Precision@10 | ≥ 1% |
| Hit Rate@10 | ≥ 10% |
| Coverage | ≥ 5% |
| Evaluated Users | ≥ 10 |

### 评估指标

#### 准确率指标
- **Precision@K**: 推荐准确率 - 推荐列表中相关物品的比例
- **Recall@K**: 召回率 - 相关物品被推荐的比例
- **F1-Score@K**: Precision和Recall的调和平均
- **NDCG@K**: 归一化折损累积增益 - 考虑排序位置的质量指标
- **Hit Rate@K**: 命中率 - 至少推荐一个相关物品的用户比例

#### 系统级指标
- **Coverage**: 推荐覆盖率 - 推荐系统能够推荐的不同物品比例
- **Diversity**: 多样性 - 不同用户推荐列表的差异度


## 配置

编辑 `config/config.yaml` 来调整系统参数：

```yaml
model:
  collaborative_filtering:
    n_neighbors: 20              # 相似用户/电影数量
    min_common_items: 3          # 最少共同项
    similarity_threshold: 0.1    # 相似度阈值
  
  content_based:
    top_n_similar_movies: 50     # 考虑的相似电影数量
  
  hybrid:
    cf_weight: 0.6               # 协同过滤权重
    content_weight: 0.4          # 基于内容权重
  
  recommendation:
    top_n: 20                    # 推荐数量
    min_rating_threshold: 3.0    # 最低评分阈值
```

