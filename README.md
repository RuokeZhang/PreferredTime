# Movie Recommendation Platform

一个基于混合推荐算法的电影推荐平台，结合协同过滤和基于内容的推荐方法。

## 项目简介

本项目实现了一个完整的**生产级**电影推荐系统，包括：
- **事件驱动架构**: Kinesis Stream + Lambda触发推荐
- **高性能缓存**: ElastiCache (Redis) 实现 <100ms P99 延迟
- **容器化部署**: Docker + ECR + ECS Fargate
- **实时数据流**: Kafka处理用户行为事件
- **混合推荐模型**: 协同过滤 + 基于内容推荐
- **自动化训练**: Airflow调度的每日模型更新
- **完整评估体系**: Precision/Recall/NDCG等业界标准指标

## 技术栈

### 核心服务
- **Python 3.11**: 主要开发语言
- **FastAPI**: 高性能Web框架
- **Redis**: ElastiCache缓存层，实现低延迟响应
- **Docker**: 容器化部署

### AWS云服务
- **Kinesis Stream**: 事件流处理
- **Lambda**: 无服务器计算，处理推荐请求
- **ECS Fargate**: 容器编排，运行FastAPI服务
- **ECR**: 容器镜像仓库
- **ElastiCache**: 托管Redis缓存
- **S3**: 数据湖存储
- **DynamoDB**: Feature Store
- **CloudWatch**: 监控和日志

### 数据处理
- **Kafka**: 实时事件流（用户行为）
- **Airflow**: 工作流调度（模型训练）
- **Pandas & NumPy**: 数据处理
- **Scikit-learn**: 机器学习工具

## 架构模式

### 🚀 生产架构（Kinesis-Lambda事件驱动）

```
用户请求 → API Gateway → Kinesis Stream
                              ↓
                         Lambda Function
                              ↓
                    FastAPI Service (ECS)
                              ↓
                    ElastiCache Redis ← 缓存层
                              ↓
                         推荐结果 (<100ms P99)
```

### 🔄 数据处理流程

```
用户行为 → Kafka → Consumer → S3 (数据湖) + DynamoDB (Feature Store)
                                    ↓
                              Airflow (每日2am)
                                    ↓
                           模型训练 + 评估 + 部署
```

**详细文档**:
- [Kinesis-Lambda架构](KINESIS_LAMBDA_ARCHITECTURE.md) ⭐ 事件驱动推荐
- [AWS部署指南](AWS_SETUP.md)
- [架构对比](ARCHITECTURE_COMPARISON.md)

## 项目结构

```
Movie_Rec/
├── api/                        # FastAPI服务
│   ├── __init__.py
│   └── main.py                 # API主文件（支持Redis缓存+内部端点）
├── lambda/                     # ⭐ Lambda函数（Kinesis触发器）
│   ├── recommendation_handler.py
│   ├── requirements.txt
│   └── Dockerfile
├── config/                     # 配置文件
│   └── config.yaml
├── database/                   # 数据库模块
│   ├── __init__.py
│   ├── models.py               # 数据库模型
│   └── init_db.py              # 数据库初始化
├── data_processor/             # 数据处理
│   ├── __init__.py
│   ├── data_storage.py         # 数据存储
│   └── feature_extractor.py   # 特征提取
├── kafka_consumer/             # Kafka消费者
│   ├── __init__.py
│   └── consumer.py
├── models/                     # 推荐模型
│   ├── __init__.py
│   ├── collaborative_filtering.py  # 协同过滤
│   ├── content_based.py            # 基于内容
│   └── hybrid_model.py             # 混合模型
├── utils/                      # 工具类
│   ├── __init__.py
│   └── logger.py
├── docker-compose.yml          # Kafka环境配置
├── requirements.txt            # Python依赖
├── run.sh                      # 启动脚本
└── README.md
```

## 快速开始

### 前置要求

- Python 3.9+
- Docker & Docker Compose
- pip

### 安装步骤

1. **克隆项目**（如果适用）或进入项目目录：
```bash
cd Movie_Rec
```

2. **使用启动脚本（推荐）**：
```bash
chmod +x run.sh
./run.sh
```

启动脚本会自动完成以下操作：
- 检查环境依赖
- 安装Python包
- 启动Kafka和Zookeeper
- 初始化数据库
- 启动Kafka消费者
- 启动FastAPI服务

### 手动启动步骤

如果您想手动启动各个组件：

1. **安装Python依赖**：
```bash
pip install -r requirements.txt
```

2. **启动Kafka环境**：
```bash
docker-compose up -d
```

等待30秒让Kafka完全启动。

3. **初始化数据库**：
```bash
python database/init_db.py
```

4. **启动Kafka消费者**（在新的终端窗口）：
```bash
python -m kafka_consumer.consumer
```

5. **启动FastAPI服务**（在新的终端窗口）：
```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8082
```

## API使用

### 主要端点

1. **获取推荐** - `GET /recommend/{user_id}`
   
   返回指定用户的电影推荐（逗号分隔的电影ID列表）
   
   ```bash
   curl http://localhost:8082/recommend/123
   ```
   
   响应示例：
   ```
   456,789,101,202,303,404,505,606,707,808,909,1010,1111,1212,1313,1414,1515,1616,1717,1818
   ```

2. **健康检查** - `GET /health`
   
   检查服务状态
   
   ```bash
   curl http://localhost:8082/health
   ```

3. **重新加载模型** - `POST /reload`
   
   在数据更新后重新训练和加载模型
   
   ```bash
   curl -X POST http://localhost:8082/reload
   ```

4. **用户画像** - `GET /user/{user_id}/profile`
   
   获取用户的评分历史和偏好信息
   
   ```bash
   curl http://localhost:8082/user/123/profile
   ```

5. **相似电影** - `GET /movie/{movie_id}/similar?top_n=10`
   
   获取与指定电影相似的电影
   
   ```bash
   curl http://localhost:8082/movie/456/similar?top_n=10
   ```

## Kafka数据格式

向Kafka topic `user-events` 发送的用户事件数据格式：

```json
{
    "user_id": 123,
    "movie_id": 456,
    "rating": 4.5,
    "timestamp": "2025-03-15T10:30:00"
}
```

字段说明：
- `user_id`: 用户ID（整数）
- `movie_id`: 电影ID（整数）
- `rating`: 评分（浮点数，通常1-5）
- `timestamp`: 时间戳（ISO格式字符串，可选）

### 发送测试数据

您可以使用以下Python脚本向Kafka发送测试数据：

```python
from kafka import KafkaProducer
import json
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# 发送测试事件
event = {
    'user_id': 1,
    'movie_id': 100,
    'rating': 4.5,
    'timestamp': datetime.utcnow().isoformat()
}

producer.send('user-events', event)
producer.flush()
```

## 推荐算法

本系统采用混合推荐策略，结合以下方法：

### 1. 协同过滤 (Collaborative Filtering)

- **User-based CF**: 基于相似用户的评分进行推荐
- **Item-based CF**: 基于相似电影进行推荐
- 使用余弦相似度计算相似性

### 2. 基于内容 (Content-Based)

- 基于电影的特征和用户的历史偏好
- 计算电影之间的相似度矩阵
- 推荐与用户喜欢的电影相似的内容

### 3. 混合策略 (Hybrid)

- 默认权重：协同过滤 60%，基于内容 40%
- 可在 `config/config.yaml` 中调整权重
- 自动降级到热门推荐（针对新用户或冷启动情况）

## 模型评估

本系统使用业界标准的推荐系统评估指标，全面衡量模型性能：

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

### 评估流程

```
1. 数据划分 (80% 训练, 20% 测试)
   ↓
2. 在训练集上构建模型
   ↓
3. 为测试用户生成推荐
   ↓
4. 与测试集对比计算指标
   ↓
5. 决定是否部署新模型
```

### 质量阈值

模型必须满足以下最低要求才能部署到生产环境：

| 指标 | 最低要求 |
|------|---------|
| Precision@10 | ≥ 1% |
| Hit Rate@10 | ≥ 10% |
| Coverage | ≥ 5% |
| Evaluated Users | ≥ 10 |

### 查看评估结果

评估会在模型训练流程中自动执行：

```bash
# 本地测试
python model_training/train_model.py

# 通过Airflow（每天02:00自动运行）
# 查看日志: ~/airflow/logs/daily_model_retraining/evaluate_model/
```

### 详细文档

- 📊 [评估指标详解](EVALUATION_METRICS.md) - 每个指标的详细说明和计算方法
- 📖 [评估系统使用指南](EVALUATION_USAGE.md) - 如何使用和改进评估系统
- 🧪 [测试评估功能](test_evaluation.py) - 单元测试脚本

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

## 停止服务

1. **停止FastAPI服务**: Ctrl+C

2. **停止Kafka消费者**: 
```bash
kill $(cat logs/consumer.pid)
```

3. **停止Kafka和Zookeeper**:
```bash
docker-compose down
```

## 开发和扩展

### 添加新的推荐算法

1. 在 `models/` 目录创建新的模型类
2. 在 `hybrid_model.py` 中集成新算法
3. 更新配置文件添加新参数

### 数据库扩展

目前使用SQLite，如需扩展到其他数据库：
1. 修改 `database/models.py` 中的连接字符串
2. 保持SQLAlchemy ORM接口不变

### 特征工程

在 `data_processor/feature_extractor.py` 中添加新的特征提取逻辑。

## 注意事项

- 首次启动时，如果没有数据，系统会返回默认推荐
- 推荐质量依赖于数据量，建议先导入足够的历史数据
- Kafka消费者会实时处理新的评分事件并更新特征
- 可以通过 `/reload` 端点重新训练模型

## 故障排除

### Kafka连接失败
- 确保Docker正在运行
- 等待Kafka完全启动（约30秒）
- 检查端口9092是否被占用

### 推荐结果为空
- 检查数据库是否有数据
- 使用 `/health` 端点检查服务状态
- 查看日志文件 `logs/consumer.log`

### 数据库错误
- 删除旧的数据库文件
- 重新运行 `python database/init_db.py`

## 许可证

本项目仅用于学习和研究目的。

## 联系方式

如有问题或建议，请联系项目维护者。

