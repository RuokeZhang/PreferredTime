# 面试问答指南

本文档帮助你准备关于这个电影推荐项目的面试问题。

## 核心问题准备

### Q1: "What is a data lake in your case?"

**完美回答**:

"在我的电影推荐项目中，数据湖是一个**分层的S3存储架构**，采用典型的Bronze-Silver-Gold三层模式：

**Bronze层（原始数据）**：
- 存储从Kafka实时摄取的原始评分事件
- 格式：JSON文件
- 路径：`s3://movie-rec-data-lake/bronze/user-events/date=YYYY-MM-DD/`
- 保留完整的原始数据，支持数据回溯和重新处理

**Silver层（清洗数据）**：
- 存储处理后的用户和电影特征
- 格式：Parquet列式存储（优化查询性能）
- 路径：`s3://movie-rec-data-lake/silver/user-features/`
- 用于批处理和模型训练

**Gold层（分析数据）**：
- 存储预计算的相似度矩阵和聚合统计
- 用于高级分析和报表

这个数据湖采用**Schema-on-Read**模式，数据存储时不强制Schema，读取时根据需求灵活解析。"

---

### Q2: "Why did you send data to S3 instead of directly to a database?"

**完美回答**:

"我采用S3作为数据湖有以下几个关键原因：

**1. 成本效益**
- S3存储成本极低（$0.023/GB/月），适合长期存储大量历史数据
- 关系型数据库存储相同数据成本高10-50倍

**2. 解耦存储和计算**
- S3作为持久化层，支持多种计算引擎读取（Spark、Athena、EMR）
- 数据库故障不会导致数据丢失
- 可以随时重新处理历史数据（reprocessing）

**3. Lambda架构**
- **Speed Layer**：Kafka → DynamoDB（实时特征，低延迟）
- **Batch Layer**：Kafka → S3 → Airflow（批处理，高吞吐）
- S3作为"source of truth"，DynamoDB作为"serving layer"

**4. 数据审计和合规**
- S3支持版本控制和不可变存储
- 满足数据治理要求
- 支持灾难恢复

**5. 灵活性**
- 原始JSON保留完整信息
- 可以用不同格式优化不同场景（Parquet、Avro）
- Schema变更成本低"

**代码示例**:
```python
# S3保存原始事件（Bronze层）
def save_raw_event(self, event):
    key = f"bronze/user-events/date={date}/event_{uuid}.json"
    s3_client.put_object(
        Bucket='movie-rec-data-lake',
        Key=key,
        Body=json.dumps(event)
    )

# DynamoDB保存实时特征（Serving层）
def update_user_feature(self, user_id, features):
    dynamodb_table.put_item(
        Item={
            'user_id': user_id,
            'avg_rating': features['avg_rating'],
            'rating_count': features['rating_count']
        }
    )
```

---

### Q3: "How does your collaborative filtering work?"

**完美回答**:

"我实现了**混合协同过滤**，结合User-based和Item-based两种方法：

**User-based CF（基于用户）**：
1. 计算用户相似度（余弦相似度）
2. 找到与目标用户最相似的K个用户（K=20）
3. 基于相似用户的评分预测目标用户对电影的评分
4. 公式：`predicted_rating = Σ(similarity * rating) / Σ(similarity)`

**Item-based CF（基于物品）**：
1. 计算电影相似度矩阵
2. 找到用户喜欢的电影的相似电影
3. 基于相似电影的评分预测
4. 对冷启动问题更鲁棒

**优化策略**：
- 相似度缓存：避免重复计算
- 最小共同项阈值：min_common_items=3
- Top-K限制：只考虑最相似的20个邻居

**代码实现**（关键部分）：
```python
# 计算用户相似度
def _cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# 生成推荐
def user_based_recommend(user_id, top_n=20):
    similar_users = get_similar_users(user_id)
    
    for movie_id in candidates:
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user, similarity in similar_users:
            if similar_user rated movie_id:
                weighted_sum += similarity * rating
                similarity_sum += similarity
        
        predicted_score = weighted_sum / similarity_sum
```

**性能**：
- 时间复杂度：O(K * M)，K是邻居数，M是候选电影数
- 空间复杂度：O(U * K)，U是用户数，缓存Top-K邻居"

---

### Q4: "How does your content-based model work?"

**完美回答**:

"我的基于内容的推荐使用**电影特征相似度**：

**核心思想**：
- 如果用户喜欢某部电影，推荐相似的电影
- 相似度基于其他用户的评分模式

**实现步骤**：

1. **构建电影相似度矩阵**：
   - 使用用户-电影评分矩阵
   - 计算电影之间的余弦相似度
   - 预计算并缓存（提升查询性能）

2. **分析用户偏好**：
   - 提取用户的高分电影（rating ≥ 4.0）
   - 构建用户画像

3. **生成推荐**：
   - 找到与用户喜欢的电影相似的电影
   - 使用加权平均计算推荐分数
   - 公式：`score = Σ(similarity * user_rating) / Σ(similarity)`

**代码实现**：
```python
# 预计算电影相似度矩阵
def _compute_movie_similarity_matrix(self):
    # 转置矩阵：行为电影，列为用户
    movie_matrix = rating_matrix.T
    
    # 归一化
    normalized = movie_matrix / np.linalg.norm(movie_matrix, axis=1)
    
    # 余弦相似度 = 点积
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix

# 生成推荐
def recommend(user_id, top_n=20):
    liked_movies = get_user_high_rated_movies(user_id)
    
    for candidate_movie in all_movies:
        score = 0
        for liked_movie, rating in liked_movies:
            similarity = similarity_matrix[liked_movie, candidate_movie]
            score += similarity * rating
        
        normalized_score = score / len(liked_movies)
```

**优势**：
- 解决协同过滤的冷启动问题
- 可以推荐新电影（只要有相似电影）
- 提供推荐解释能力"

---

### Q5: "How did you implement the hybrid model?"

**完美回答**:

"我实现了**加权混合推荐模型**，融合三种方法：

**混合策略**：
- User-based CF: 30%权重
- Item-based CF: 30%权重  
- Content-based: 40%权重

**实现逻辑**：
```python
def recommend(user_id, top_n=20):
    all_recommendations = {}
    
    # 1. User-based CF推荐
    user_cf_recs = self.cf_model.user_based_recommend(user_id)
    merge_with_weight(all_recommendations, user_cf_recs, weight=0.3)
    
    # 2. Item-based CF推荐
    item_cf_recs = self.cf_model.item_based_recommend(user_id)
    merge_with_weight(all_recommendations, item_cf_recs, weight=0.3)
    
    # 3. Content-based推荐
    content_recs = self.content_model.recommend(user_id)
    merge_with_weight(all_recommendations, content_recs, weight=0.4)
    
    # 4. 按综合分数排序
    sorted_recs = sorted(all_recommendations.items(), 
                        key=lambda x: x[1], reverse=True)
    
    return [movie_id for movie_id, score in sorted_recs[:top_n]]
```

**降级策略**：
- 新用户或数据不足时，返回热门电影
- 使用贝叶斯平均计算热门度

**为什么这样设计**：
- CF擅长发现相似用户/电影的规律
- Content-based解决冷启动问题
- 混合可以平衡两者优缺点
- 权重可配置，支持A/B测试"

---

### Q6: "How do you handle the Kafka stream?"

**完美回答**:

"我使用**kafka-python**库实现实时流处理：

**消费者架构**：
```python
class MovieRecConsumer:
    def __init__(self, config):
        # Kafka消费者
        self.consumer = KafkaConsumer(
            'user-events',
            bootstrap_servers='localhost:9092',
            group_id='movie-rec-consumer-group',
            auto_offset_reset='earliest',
            value_deserializer=lambda x: json.loads(x)
        )
        
        # 混合存储层
        self.storage = HybridStorage(config)
    
    def start(self):
        for message in self.consumer:
            event = message.value
            self.process_event(event)
    
    def process_event(self, event):
        # 1. 保存原始事件到S3 Bronze层
        self.storage.save_rating(
            user_id=event['user_id'],
            movie_id=event['movie_id'],
            rating=event['rating'],
            timestamp=event['timestamp']
        )
        
        # 2. 更新实时特征（如果需要）
        if self.mode == 'sqlite':
            self.feature_extractor.extract_features(event)
```

**事件格式**：
```json
{
  "user_id": 123,
  "movie_id": 456,
  "rating": 4.5,
  "timestamp": "2025-03-15T10:30:00"
}
```

**可靠性保证**：
- Consumer Group：支持并行消费和负载均衡
- Auto Commit：自动提交offset
- Error Handling：异常捕获和日志记录
- At-least-once语义：确保不丢失数据"

---

### Q7: "How would you use Airflow for daily retraining?"

**完美回答**:

"虽然当前实现没有Airflow，但我设计了完整的批处理架构：

**Airflow DAG设计**：
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

dag = DAG(
    'daily_feature_etl',
    schedule_interval='@daily',  # 每天凌晨运行
    start_date=datetime(2025, 3, 1)
)

# Task 1: 数据质量检查
def validate_s3_data(**context):
    date = context['ds']  # 昨天的日期
    events = read_from_s3_bronze(date)
    
    # 检查数据完整性
    assert len(events) > 0
    assert all(validate_event(e) for e in events)

# Task 2: 特征工程
def extract_features(**context):
    events = read_from_s3_bronze(context['ds'])
    
    # 计算用户特征
    user_features = compute_user_features(events)
    
    # 计算电影特征  
    movie_features = compute_movie_features(events)
    
    # 保存到S3 Silver层（Parquet）
    save_to_s3_silver(user_features, movie_features)
    
    return {'user_count': len(user_features)}

# Task 3: 更新DynamoDB
def update_feature_store(**context):
    features = read_from_s3_silver(context['ds'])
    
    # 批量更新DynamoDB
    batch_write_to_dynamodb(features)

# Task 4: 模型训练
def train_model(**context):
    # 读取历史数据
    data = read_training_data_from_s3()
    
    # 训练模型
    model = HybridRecommender(data)
    
    # 评估模型
    metrics = evaluate_model(model)
    
    # 使用MLflow跟踪
    with mlflow.start_run():
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "recommender")
    
    # 如果性能提升，部署到生产
    if metrics['hit_rate'] > current_best:
        deploy_model(model)

# Task 5: 通知
def send_notification(**context):
    metrics = context['task_instance'].xcom_pull(task_ids='train_model')
    send_email(f"Daily training completed: {metrics}")

# 定义依赖关系
validate = PythonOperator(task_id='validate', python_callable=validate_s3_data, dag=dag)
extract = PythonOperator(task_id='extract', python_callable=extract_features, dag=dag)
update = PythonOperator(task_id='update', python_callable=update_feature_store, dag=dag)
train = PythonOperator(task_id='train', python_callable=train_model, dag=dag)
notify = PythonOperator(task_id='notify', python_callable=send_notification, dag=dag)

validate >> extract >> [update, train] >> notify
```

**监控和告警**：
- 训练失败时发送告警
- 模型性能下降时通知
- 数据质量异常时暂停pipeline"

---

### Q8: "How would you track experiments with MLflow?"

**完美回答**:

"MLflow用于**模型生命周期管理**：

**实验跟踪**：
```python
import mlflow

# 开始实验
with mlflow.start_run(run_name="hybrid_model_v1"):
    # 1. 记录参数
    mlflow.log_params({
        'n_neighbors': 20,
        'cf_weight': 0.6,
        'content_weight': 0.4,
        'min_common_items': 3
    })
    
    # 2. 训练模型
    model = HybridRecommender(rating_matrix, config)
    
    # 3. 评估性能
    metrics = evaluate_model(model, test_data)
    mlflow.log_metrics({
        'rmse': 0.85,
        'mae': 0.65,
        'hit_rate@10': 0.72,
        'ndcg@10': 0.68
    })
    
    # 4. 保存模型
    mlflow.sklearn.log_model(model, "model")
    
    # 5. 记录数据版本
    mlflow.log_param('data_date', '2025-03-15')
    mlflow.log_param('training_samples', len(train_data))
```

**模型注册和部署**：
```python
# 注册最佳模型
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "HybridRecommender")

# 标记为生产版本
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="HybridRecommender",
    version=3,
    stage="Production"
)

# API中加载生产模型
model = mlflow.sklearn.load_model("models:/HybridRecommender/Production")
```

**A/B测试**：
```python
# 同时运行两个模型版本
model_a = load_model("models:/HybridRecommender/Production")
model_b = load_model("models:/HybridRecommender/Staging")

# 随机分配用户
if user_id % 2 == 0:
    recommendations = model_a.recommend(user_id)
    mlflow.log_metric(f"user_{user_id}_model", 'A')
else:
    recommendations = model_b.recommend(user_id)
    mlflow.log_metric(f"user_{user_id}_model", 'B')
```

**监控生产模型**：
- 跟踪API响应时间
- 记录推荐点击率
- 监控模型性能退化"

---

## 技术深度问题

### Q9: "What's the difference between your data lake (S3) and a data warehouse?"

| 特性 | 数据湖 (S3) | 数据仓库 (Redshift) |
|------|------------|-------------------|
| Schema | Schema-on-Read | Schema-on-Write |
| 数据类型 | 原始、非结构化 | 结构化、聚合 |
| 处理方式 | ELT | ETL |
| 成本 | 低 | 高 |
| 查询性能 | 一般（扫描） | 优秀（索引） |
| 用途 | 存储+探索 | 分析报表 |

**在我的项目中**：
- S3存储原始评分事件（灵活、低成本）
- DynamoDB作为特征仓库（快速查询）
- 未来可以添加Redshift用于分析报表

---

### Q10: "How do you ensure data quality?"

**完美回答**:

"我实现了多层数据质量保证：

**1. 输入验证**：
```python
def validate_event(event):
    # 必需字段检查
    required = ['user_id', 'movie_id', 'rating', 'timestamp']
    if not all(k in event for k in required):
        raise ValueError("Missing required fields")
    
    # 数据类型检查
    assert isinstance(event['user_id'], int)
    assert isinstance(event['rating'], (int, float))
    
    # 值范围检查
    assert 1.0 <= event['rating'] <= 5.0
    
    return True
```

**2. 数据清洗**：
- 去重：基于(user_id, movie_id, timestamp)
- 异常值处理：过滤极端评分
- 缺失值处理：填充或删除

**3. Schema验证**（Parquet）：
```python
schema = pa.schema([
    ('user_id', pa.int64()),
    ('avg_rating', pa.float64()),
    ('rating_count', pa.int64())
])

df.to_parquet('features.parquet', schema=schema)
```

**4. 监控指标**：
- 每日事件数量
- 异常率
- 数据延迟
- 特征分布"

---

## 快速参考

### 数据流总结

```
实时流: Kafka → S3 Bronze (JSON)
批处理: S3 Bronze → Airflow → S3 Silver (Parquet) + DynamoDB
推荐: DynamoDB → FastAPI → 用户
```

### 关键数字

- **评分矩阵**: 100 users × 500 movies
- **相似度邻居**: K=20
- **推荐数量**: Top-20
- **特征更新**: 每天批处理
- **API延迟**: <100ms (DynamoDB查询)
- **成本**: ~$10/月 (10万用户规模)

### 技术关键词

面试中要突出的关键词：
- Lambda Architecture
- Data Lake (Bronze-Silver-Gold)
- Feature Store
- Hybrid Recommendation
- Collaborative Filtering
- Content-based Filtering
- Real-time Stream Processing
- Batch Processing
- Schema-on-Read
- MLflow Model Tracking

### 项目亮点

1. ✅ **可扩展架构**: SQLite → AWS无缝切换
2. ✅ **Lambda架构**: 实时+批处理
3. ✅ **混合推荐**: 三种算法融合
4. ✅ **完整数据流**: Kafka → S3 → DynamoDB
5. ✅ **生产级代码**: 错误处理、日志、监控
6. ✅ **面试友好**: 清晰的架构文档

祝面试顺利！🎬🚀


