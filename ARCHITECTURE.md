# 架构设计文档

## 系统概述

Movie Recommendation Platform是一个基于混合推荐算法的实时电影推荐系统。系统通过Kafka接收用户评分事件，实时更新特征，并提供基于HTTP的推荐服务。

## 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    用户评分事件源                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Kafka Topic   │
         │ (user-events)  │
         └────────┬───────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │   Kafka Consumer        │
    │  (实时事件处理)          │
    └──────────┬──────────────┘
               │
               ▼
    ┌─────────────────────────┐
    │  Feature Extractor      │
    │  (特征提取)              │
    └──────────┬──────────────┘
               │
               ▼
    ┌─────────────────────────┐
    │   SQLite Database       │
    │  - ratings              │
    │  - user_features        │
    │  - movie_features       │
    └──────────┬──────────────┘
               │
               ▼
    ┌─────────────────────────┐
    │   Hybrid Recommender    │
    │  - Collaborative Filter │
    │  - Content Based        │
    └──────────┬──────────────┘
               │
               ▼
    ┌─────────────────────────┐
    │    FastAPI Service      │
    │  (HTTP推荐接口)          │
    └─────────────────────────┘
```

## 核心模块

### 1. Kafka Consumer (kafka_consumer/)

**职责**: 实时接收和处理用户评分事件

**主要功能**:
- 订阅Kafka topic (`user-events`)
- 解析JSON格式的评分事件
- 调用数据存储层保存评分
- 触发特征更新

**数据格式**:
```json
{
    "user_id": 123,
    "movie_id": 456,
    "rating": 4.5,
    "timestamp": "2025-03-15T10:30:00"
}
```

**技术选择**: kafka-python库，异步处理

### 2. Data Processor (data_processor/)

#### 2.1 Data Storage (data_storage.py)

**职责**: 数据持久化和查询

**主要功能**:
- 保存评分记录
- 查询用户/电影的评分历史
- 更新用户和电影特征
- 获取评分矩阵数据

**数据库表**:
- `ratings`: 原始评分记录
- `user_features`: 用户特征（平均分、评分数、标准差）
- `movie_features`: 电影特征（平均分、评分数、流行度）
- `user_similarity`: 用户相似度（预留）
- `movie_similarity`: 电影相似度（预留）

#### 2.2 Feature Extractor (feature_extractor.py)

**职责**: 从原始数据中提取推荐所需的特征

**主要功能**:
- 提取用户特征（评分统计、偏好分析）
- 提取电影特征（评分统计、流行度）
- 构建用户-电影评分矩阵
- 计算用户/电影相似度

**核心算法**:
- 余弦相似度计算
- 矩阵构建与转换
- 特征归一化

### 3. Recommendation Models (models/)

#### 3.1 Collaborative Filtering (collaborative_filtering.py)

**方法1: User-based CF**
- 找到与目标用户相似的用户
- 基于相似用户的评分预测目标用户对电影的评分
- 使用余弦相似度计算用户相似性

**方法2: Item-based CF**
- 找到与用户喜欢的电影相似的电影
- 基于相似电影的评分预测
- 使用余弦相似度计算电影相似性

**优化**:
- 相似度缓存机制
- 最小共同项阈值
- 可配置的邻居数量

#### 3.2 Content-Based (content_based.py)

**核心思想**: 基于电影内容特征和用户历史偏好

**主要步骤**:
1. 预计算电影相似度矩阵（基于评分模式）
2. 分析用户的高分电影
3. 推荐与用户喜欢的电影相似的电影
4. 使用加权平均计算推荐分数

**特点**:
- 解决协同过滤的冷启动问题
- 基于用户历史偏好
- 提供推荐解释能力

#### 3.3 Hybrid Model (hybrid_model.py)

**混合策略**: 加权组合协同过滤和基于内容的推荐

**默认权重**:
- User-based CF: 30%
- Item-based CF: 30%
- Content-based: 40%

**降级策略**:
- 新用户或数据不足时，返回热门电影
- 使用贝叶斯平均计算热门度（平衡评分和评分数量）

**推荐流程**:
```
1. 生成User-based CF推荐
2. 生成Item-based CF推荐
3. 生成Content-based推荐
4. 加权合并所有推荐结果
5. 按综合分数排序
6. 返回Top-N推荐
```

### 4. API Service (api/)

**框架**: FastAPI

**核心端点**:

| 端点 | 方法 | 功能 | 响应格式 |
|------|------|------|----------|
| `/recommend/{user_id}` | GET | 获取推荐 | 逗号分隔的电影ID |
| `/health` | GET | 健康检查 | JSON状态信息 |
| `/reload` | POST | 重新加载模型 | JSON成功/失败 |
| `/user/{user_id}/profile` | GET | 用户画像 | JSON用户信息 |
| `/movie/{movie_id}/similar` | GET | 相似电影 | JSON相似电影列表 |

**启动流程**:
1. 加载配置文件
2. 连接数据库
3. 构建评分矩阵
4. 初始化推荐模型
5. 启动HTTP服务

**特点**:
- 异步处理
- 自动初始化
- 错误处理和日志
- 模型热重载

### 5. Database (database/)

**技术选择**: SQLite + SQLAlchemy ORM

**优点**:
- 轻量级，无需额外安装
- 足够支持中小规模数据
- ORM易于维护和扩展

**扩展性**: 可通过修改连接字符串轻松切换到PostgreSQL/MySQL

**表设计**:

```sql
-- 评分表
CREATE TABLE ratings (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    rating FLOAT NOT NULL,
    timestamp DATETIME,
    INDEX(user_id),
    INDEX(movie_id)
);

-- 用户特征表
CREATE TABLE user_features (
    user_id INTEGER PRIMARY KEY,
    avg_rating FLOAT,
    rating_count INTEGER,
    std_dev FLOAT,
    last_update DATETIME
);

-- 电影特征表
CREATE TABLE movie_features (
    movie_id INTEGER PRIMARY KEY,
    avg_rating FLOAT,
    rating_count INTEGER,
    popularity FLOAT,
    last_update DATETIME
);
```

## 数据流

### 评分事件流

```
用户评分 → Kafka Topic → Consumer → 保存到数据库 → 更新特征 → 可用于推荐
```

### 推荐生成流

```
HTTP请求 → 加载用户数据 → 生成多种推荐 → 混合策略合并 → 返回Top-N
```

## 配置管理

所有配置集中在 `config/config.yaml`:

```yaml
kafka:
  bootstrap_servers: "localhost:9092"
  topic: "user-events"

database:
  path: "database/movie_rec.db"

model:
  collaborative_filtering:
    n_neighbors: 20
    min_common_items: 3
  
  content_based:
    top_n_similar_movies: 50
  
  hybrid:
    cf_weight: 0.6
    content_weight: 0.4

api:
  host: "0.0.0.0"
  port: 8082
```

## 性能考虑

### 时间复杂度

- **相似度计算**: O(n²) 其中n是用户/电影数量
  - 优化: 缓存机制，增量更新
  
- **推荐生成**: O(k * m) 其中k是邻居数，m是候选电影数
  - 优化: Top-K堆，早停策略

### 空间复杂度

- **评分矩阵**: O(u * m) 其中u是用户数，m是电影数
  - 稀疏矩阵优化: 使用Pandas DataFrame自动处理稀疏性
  
- **相似度缓存**: O(n * k) 其中k是缓存的邻居数

### 扩展性建议

**当前适用规模**:
- 用户: < 10,000
- 电影: < 10,000
- 评分: < 1,000,000

**超出规模后的优化方案**:
1. 使用PostgreSQL替代SQLite
2. 实现分布式计算（Spark）
3. 引入Redis缓存推荐结果
4. 使用矩阵分解（SVD）替代协同过滤
5. 批量更新特征而非实时更新

## 可扩展性

### 添加新的推荐算法

1. 在 `models/` 目录创建新模型类
2. 实现 `recommend()` 方法
3. 在 `hybrid_model.py` 中集成
4. 在配置文件中添加参数

### 切换数据库

1. 修改 `database/models.py` 中的 `get_engine()` 函数
2. 更新配置文件中的数据库路径
3. SQLAlchemy ORM代码无需修改

### 添加新特征

1. 在 `database/models.py` 添加新表
2. 在 `feature_extractor.py` 添加提取逻辑
3. 在推荐模型中使用新特征

## 监控和日志

**日志级别**: INFO（可配置）

**日志位置**:
- FastAPI服务: 标准输出
- Kafka消费者: `logs/consumer.log`

**监控指标**:
- 推荐请求量
- 推荐生成时间
- 评分事件处理速度
- 数据库查询性能

## 安全考虑

**当前实现**: 开发环境，无认证

**生产环境建议**:
1. 添加API认证（JWT Token）
2. 限流和请求频率限制
3. 数据加密（传输和存储）
4. 输入验证和SQL注入防护（已使用ORM）

## 测试策略

**单元测试**: 每个模块独立测试
**集成测试**: 端到端流程测试
**性能测试**: 负载测试和压力测试

**测试工具**:
- `test_producer.py`: 数据生成
- `test_api.py`: API测试

## 部署建议

**开发环境**: 使用 `run.sh` 一键启动

**生产环境**:
1. 使用Docker容器化所有服务
2. Kubernetes编排（可选）
3. 使用Nginx作为反向代理
4. 配置日志聚合（ELK Stack）
5. 设置监控告警（Prometheus + Grafana）

## 总结

这个架构设计平衡了以下因素：
- ✅ 简单性：易于理解和维护
- ✅ 可扩展性：模块化设计，易于扩展
- ✅ 性能：缓存和优化策略
- ✅ 实用性：满足中小规模应用需求

适合作为学习项目或原型系统，也可以作为生产系统的起点。

