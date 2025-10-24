# 架构对比：SQLite vs AWS (S3 + DynamoDB)

本文档详细对比了两种存储架构的实现。

## 架构对比图

### SQLite架构（开发环境）

```
┌─────────────────────────────────────────────────────────┐
│                    Kafka Stream                          │
│                  (user-events topic)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
            ┌──────────────────┐
            │  Kafka Consumer  │
            │  - 实时消费      │
            │  - 特征提取      │
            └────────┬─────────┘
                     │
                     ▼
            ┌────────────────────┐
            │   SQLite Database  │
            │                    │
            │ ┌────────────────┐ │
            │ │ ratings        │ │ ← 原始评分数据
            │ └────────────────┘ │
            │ ┌────────────────┐ │
            │ │ user_features  │ │ ← 用户特征
            │ └────────────────┘ │
            │ ┌────────────────┐ │
            │ │ movie_features │ │ ← 电影特征
            │ └────────────────┘ │
            └────────┬───────────┘
                     │
                     ▼
            ┌────────────────────┐
            │   FastAPI Service  │
            │  - 加载数据        │
            │  - 混合推荐        │
            │  - HTTP API        │
            └────────────────────┘
```

### AWS架构（生产环境）

```
┌─────────────────────────────────────────────────────────┐
│                    Kafka Stream                          │
│                  (user-events topic)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   Kafka Consumer     │
            │   - 实时消费         │
            └──────┬───────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐    ┌─────────────────┐
│  S3 Data Lake │    │    DynamoDB     │
│               │    │  Feature Store  │
│ Bronze Layer  │    │                 │
│ ┌───────────┐ │    │ ┌─────────────┐ │
│ │Raw Events │ │    │ │user_features│ │
│ │  (JSON)   │ │    │ └─────────────┘ │
│ └───────────┘ │    │ ┌─────────────┐ │
│               │    │ │movie_features│ │
│ Silver Layer  │    │ └─────────────┘ │
│ ┌───────────┐ │    └────────┬────────┘
│ │  Features │ │             │
│ │ (Parquet) │ │             │
│ └───────────┘ │             │
│               │             │
│ Gold Layer    │             │
│ ┌───────────┐ │             │
│ │Similarity │ │             │
│ │ Matrices  │ │             │
│ └───────────┘ │             │
└───────┬───────┘             │
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
          ┌────────────────────┐
          │  FastAPI Service   │
          │  - DynamoDB查询    │
          │  - 混合推荐        │
          │  - HTTP API        │
          └────────────────────┘
                   ▲
                   │
          ┌────────┴─────────┐
          │  Airflow (未来)  │
          │  - 批处理特征     │
          │  - 模型训练       │
          │  - MLflow跟踪    │
          └──────────────────┘
```

## 详细对比

### 1. 数据存储层

| 特性 | SQLite | AWS (S3 + DynamoDB) |
|------|--------|---------------------|
| **原始数据** | `ratings`表 | S3 Bronze层（JSON文件） |
| **处理数据** | `*_features`表 | S3 Silver层（Parquet文件） |
| **实时特征** | `*_features`表 | DynamoDB表 |
| **查询延迟** | ~10ms | DynamoDB: <5ms, S3: 秒级 |
| **扩展性** | 单机限制（~GB级） | 无限扩展（PB级） |
| **成本** | 免费（本地） | S3: $0.023/GB, DynamoDB按请求 |

### 2. 代码实现对比

#### SQLite模式

```python
# kafka_consumer/consumer.py
class MovieRecConsumer:
    def __init__(self, config):
        # 直接使用SQLite
        self.data_storage = DataStorage(config['database']['path'])
        self.feature_extractor = FeatureExtractor(self.data_storage)
    
    def process_event(self, event):
        # 保存到SQLite
        self.data_storage.save_rating(user_id, movie_id, rating, timestamp)
        
        # 实时计算特征
        self.feature_extractor.extract_user_features(user_id)
        self.feature_extractor.extract_movie_features(movie_id)
```

#### AWS模式

```python
# kafka_consumer/consumer.py
class MovieRecConsumer:
    def __init__(self, config):
        # 使用混合存储层
        self.data_storage = HybridStorage(config)
        # storage_mode = 'aws'
    
    def process_event(self, event):
        # 保存到S3 Bronze层（JSON）
        self.data_storage.save_rating(user_id, movie_id, rating, timestamp)
        # 内部调用: s3_storage.save_raw_event(event)
        
        # 特征由批处理计算（不实时）
        # 通过Airflow定期从S3 Bronze → Silver → DynamoDB
```

### 3. 数据流对比

#### SQLite数据流

```
事件 → Kafka → Consumer → SQLite(ratings) 
                    ↓
            实时特征提取
                    ↓
        SQLite(user_features, movie_features)
                    ↓
                FastAPI查询
```

#### AWS数据流

```
事件 → Kafka → Consumer → S3 Bronze (原始JSON)
                    ↓
            Airflow批处理（每天）
                    ↓
        ┌───────────┴───────────┐
        ▼                       ▼
S3 Silver (Parquet)      DynamoDB (实时特征)
        │                       │
        └───────────┬───────────┘
                    ↓
            FastAPI查询（DynamoDB）
```

### 4. 特征计算对比

#### SQLite：实时计算

```python
# 每次收到评分事件
def process_event(event):
    # 1. 保存评分
    save_rating(event)
    
    # 2. 立即重新计算特征
    ratings = get_user_ratings(user_id)  # 查询SQLite
    avg_rating = mean(ratings)           # 计算平均分
    update_user_feature(user_id, avg_rating)  # 更新SQLite
```

**优点**: 特征始终最新  
**缺点**: 每次事件都要计算，高负载时性能差

#### AWS：批处理计算

```python
# Airflow DAG（每天凌晨运行）
def daily_feature_pipeline():
    # 1. 从S3读取昨天的事件
    events = read_from_s3_bronze(date='yesterday')
    
    # 2. 批量计算特征（使用Spark）
    user_features = compute_user_features(events)
    movie_features = compute_movie_features(events)
    
    # 3. 写入S3 Silver层（Parquet）
    save_to_s3_silver(user_features, movie_features)
    
    # 4. 更新DynamoDB（实时查询用）
    batch_update_dynamodb(user_features, movie_features)
```

**优点**: 高效批处理，适合大规模数据  
**缺点**: 特征有延迟（最多24小时）

### 5. 文件结构对比

```
data_processor/
├── data_storage.py          # SQLite存储（原有）
├── s3_storage.py            # S3存储（新增）
├── dynamodb_storage.py      # DynamoDB存储（新增）
├── hybrid_storage.py        # 混合存储（新增）
└── feature_extractor.py     # 特征提取（共用）

config/config.yaml:
  storage_mode: "sqlite"     # 或 "aws"
  database: {...}            # SQLite配置
  aws:                       # AWS配置
    s3: {...}
    dynamodb: {...}
```

### 6. 配置切换

只需修改 `config/config.yaml` 一行即可切换：

```yaml
# 开发环境：本地SQLite
storage_mode: "sqlite"

# 生产环境：AWS云存储
storage_mode: "aws"
```

所有代码自动适配！

### 7. 性能对比

#### 写入性能

| 操作 | SQLite | AWS |
|------|--------|-----|
| 单次写入 | ~1ms | S3: ~50ms |
| 批量写入(1000) | ~100ms | S3: ~500ms |
| 吞吐量 | ~10K/s | ~20K/s（多线程） |

#### 读取性能

| 操作 | SQLite | AWS |
|------|--------|-----|
| 查询单个特征 | ~1ms | DynamoDB: ~2ms |
| 查询1000个特征 | ~50ms | DynamoDB: ~100ms（批量） |
| 扫描全表 | ~秒级 | 分钟级（需Athena） |

#### 推荐生成性能

| 场景 | SQLite | AWS |
|------|--------|-----|
| 单用户推荐 | ~50ms | ~60ms（DynamoDB查询） |
| 冷启动（新用户） | ~100ms | ~100ms（相同） |
| 模型加载 | ~5秒 | ~10秒（从S3读取） |

### 8. 成本对比

#### SQLite（开发）

- **存储成本**: $0（本地磁盘）
- **计算成本**: $0（本地CPU）
- **总成本**: $0

#### AWS（生产）

**假设**: 10万用户，1000万评分，100万次API调用/月

| 服务 | 用量 | 成本 |
|------|------|------|
| S3存储 | 10GB | $0.23 |
| S3请求 | 100万写+100万读 | $5.40 |
| DynamoDB写入 | 10万次/天 | $3.75 |
| DynamoDB读取 | 100万次/月 | $0.25 |
| DynamoDB存储 | 1GB | $0.25 |
| **总计** | | **~$10/月** |

### 9. 优缺点总结

#### SQLite架构

**优点**:
- ✅ 简单易用，无需AWS账户
- ✅ 开发速度快
- ✅ 零成本
- ✅ 实时特征更新
- ✅ 适合原型和演示

**缺点**:
- ❌ 无法扩展到大规模
- ❌ 单机性能限制
- ❌ 无数据冗余
- ❌ 不适合生产环境

#### AWS架构

**优点**:
- ✅ 无限扩展性
- ✅ 高可用性（99.99%）
- ✅ 数据持久化保证
- ✅ 支持大规模生产
- ✅ Lambda架构（批处理+流处理）

**缺点**:
- ❌ 有成本
- ❌ 需要AWS知识
- ❌ 配置较复杂
- ❌ 特征更新有延迟

### 10. 使用建议

#### 使用SQLite的场景：
- 开发和测试
- 原型验证
- 数据量 < 1GB
- 用户数 < 10K
- QPS < 100

#### 使用AWS的场景：
- 生产环境
- 数据量 > 10GB
- 用户数 > 100K
- QPS > 1000
- 需要高可用性
- 需要数据分析

### 11. 迁移路径

#### 从SQLite迁移到AWS：

```bash
# 1. 安装AWS依赖
pip install boto3 pyarrow

# 2. 配置AWS凭证
aws configure

# 3. 创建AWS资源
python3 setup_aws.py

# 4. 修改配置
# 编辑 config/config.yaml
storage_mode: "aws"

# 5. 重启服务
./stop.sh
./run.sh

# 6. （可选）导出SQLite数据到S3
python3 migrate_sqlite_to_s3.py
```

#### 从AWS回退到SQLite：

```bash
# 1. 修改配置
# 编辑 config/config.yaml
storage_mode: "sqlite"

# 2. 重启服务
./stop.sh
./run.sh
```

## 总结

两种架构各有优势：

- **SQLite**: 开发快速，成本零，适合学习和原型
- **AWS**: 可扩展，高可用，适合生产和面试展示

**推荐策略**: 
- 开发阶段用SQLite
- 面试展示时强调AWS架构
- 实际面试时可以演示代码如何轻松切换

