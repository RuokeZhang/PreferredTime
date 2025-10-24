# Kinesis-Lambda 事件驱动推荐架构

本文档详细说明基于AWS Kinesis Stream和Lambda的事件驱动推荐系统架构。

## 🏗️ 架构概览

```
用户请求
    ↓
API Gateway / 应用
    ↓
Kinesis Stream (movie-rec-requests)
    ↓
Lambda Function (批处理)
    ↓
FastAPI Service (ECS/Fargate)
    ↓
ElastiCache Redis (缓存层)
    ↓
推荐结果
```

### 核心组件

1. **Kinesis Stream**: 接收推荐请求事件
2. **Lambda Function**: 处理事件并调用推荐服务
3. **FastAPI Service**: 部署在ECS的推荐引擎
4. **ElastiCache Redis**: 缓存层，加速响应

## 🎯 性能目标

**P99 延迟 < 100ms**

- **缓存命中**: 5-20ms (从Redis读取)
- **缓存未命中**: 50-100ms (调用推荐服务)
- **批处理吞吐量**: 100+ 请求/秒

---

## 📦 组件详解

### 1. Kinesis Stream 配置

```json
{
  "StreamName": "movie-rec-requests",
  "ShardCount": 2,
  "RetentionPeriod": 24
}
```

**事件格式**:
```json
{
  "user_id": 123,
  "top_n": 20,
  "request_id": "uuid-xxxx-xxxx",
  "timestamp": "2025-10-24T10:30:00Z"
}
```

**分区策略**: 按 `user_id` 分区，确保同一用户的请求顺序处理

---

### 2. Lambda Function

**文件**: `lambda/recommendation_handler.py`

**触发器配置**:
- 批量大小: 100条记录
- 批处理窗口: 1秒
- 最大重试: 3次

**环境变量**:
```bash
FASTAPI_ENDPOINT=http://internal-alb.xxx.elb.amazonaws.com
REDIS_ENDPOINT=your-cache.xxx.cache.amazonaws.com
REDIS_PORT=6379
CACHE_TTL=3600
```

**执行流程**:
```python
for event in kinesis_events:
    1. 解码Base64数据
    2. 解析JSON请求
    3. 检查Redis缓存
    4. 如果缓存未命中，调用FastAPI
    5. 缓存结果到Redis
    6. 返回推荐
```

**容器镜像**: 使用 `lambda/Dockerfile` 构建并推送到ECR

---

### 3. FastAPI Service (ECS)

**文件**: `api/main.py`

**新增端点**:

#### `/internal/recommend/{user_id}` - Lambda专用端点
```python
GET /internal/recommend/123?top_n=20

Response:
{
  "user_id": 123,
  "recommendations": [1, 2, 3, ...],
  "count": 20
}
```

- ✅ 优化的JSON响应
- ✅ 无额外业务逻辑
- ✅ 低延迟设计

#### `/recommend/cached/{user_id}` - 带缓存的公开端点
```python
GET /recommend/cached/123?top_n=20

Response:
{
  "user_id": 123,
  "recommendations": [1, 2, 3, ...],
  "count": 20,
  "from_cache": true,
  "latency_ms": 12
}
```

**部署配置**:
- **运行时**: ECS Fargate
- **Workers**: 4个uvicorn workers
- **内存**: 1024 MB
- **CPU**: 512 (0.5 vCPU)

**Dockerfile**: 见 `Dockerfile`

---

### 4. ElastiCache Redis

**配置**:
- 节点类型: `cache.t3.micro` (开发) / `cache.r6g.large` (生产)
- 引擎版本: Redis 7.0
- 集群模式: 单节点 (开发) / 多节点 (生产)

**缓存策略**:
- **键格式**: `rec:user:{user_id}:top{top_n}`
- **TTL**: 3600秒 (1小时)
- **淘汰策略**: LRU (Least Recently Used)

**缓存效果**:
- 命中率目标: >70%
- 延迟降低: 80-90%

---

## 🚀 部署流程

### 步骤 1: 构建和推送Docker镜像

#### FastAPI服务镜像 (ECR)
```bash
# 登录ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# 构建镜像
docker build -t movie-rec-api .

# 打标签
docker tag movie-rec-api:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/movie-rec-api:latest

# 推送
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/movie-rec-api:latest
```

#### Lambda函数镜像
```bash
cd lambda

# 构建Lambda镜像
docker build -t movie-rec-lambda .

# 打标签
docker tag movie-rec-lambda:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/movie-rec-lambda:latest

# 推送
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/movie-rec-lambda:latest
```

### 步骤 2: 创建AWS资源

使用AWS Console或Terraform创建:
1. Kinesis Stream: `movie-rec-requests`
2. ElastiCache Redis集群
3. ECS集群和服务
4. Lambda函数
5. IAM角色和策略

### 步骤 3: 配置Lambda触发器

```bash
aws lambda create-event-source-mapping \
  --function-name movie-rec-kinesis-handler \
  --event-source-arn arn:aws:kinesis:us-east-1:<account>:stream/movie-rec-requests \
  --starting-position LATEST \
  --batch-size 100 \
  --maximum-batching-window-in-seconds 1
```

### 步骤 4: 验证部署

```bash
# 测试FastAPI健康检查
curl http://<alb-dns>/health

# 发送测试事件到Kinesis
aws kinesis put-record \
  --stream-name movie-rec-requests \
  --partition-key user-123 \
  --data '{"user_id": 123, "top_n": 20, "request_id": "test-1", "timestamp": "2025-10-24T10:00:00Z"}'

# 查看Lambda日志
aws logs tail /aws/lambda/movie-rec-kinesis-handler --follow
```

---

## 🧪 本地测试

### 1. 启动FastAPI服务

```bash
# 设置环境变量
export REDIS_ENABLED=false  # 本地测试不需要Redis

# 启动服务
python -m uvicorn api.main:app --host 0.0.0.0 --port 8082
```

### 2. 测试Lambda函数

```bash
# 运行测试脚本
python test_kinesis_lambda.py
```

测试脚本会:
- ✅ 模拟Kinesis事件
- ✅ 调用Lambda处理器
- ✅ 测试单用户和批量场景
- ✅ 测试缓存性能
- ✅ 测试错误处理

### 3. 测试内部端点

```bash
# 测试内部推荐端点
curl "http://localhost:8082/internal/recommend/1?top_n=10"

# 测试带缓存的端点（需要Redis）
curl "http://localhost:8082/recommend/cached/1?top_n=10"
```

---

## 📊 监控和性能

### CloudWatch指标

**Lambda指标**:
- `Duration`: 执行时间
- `Invocations`: 调用次数
- `Errors`: 错误数
- `Throttles`: 限流次数
- `IteratorAge`: Kinesis记录延迟

**自定义指标**:
```python
import boto3
cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='MovieRecommendation',
    MetricData=[
        {
            'MetricName': 'RecommendationLatency',
            'Value': latency_ms,
            'Unit': 'Milliseconds',
            'Dimensions': [
                {'Name': 'CacheStatus', 'Value': 'Hit' if from_cache else 'Miss'}
            ]
        }
    ]
)
```

### 日志查询

```bash
# 查询高延迟请求
aws logs filter-log-events \
  --log-group-name /aws/lambda/movie-rec-kinesis-handler \
  --filter-pattern "[time, request_id, level, msg = *latency*, latency > 100, ...]"

# 查询错误
aws logs filter-log-events \
  --log-group-name /aws/lambda/movie-rec-kinesis-handler \
  --filter-pattern "ERROR"
```

---

## 💰 成本估算

### 月度成本（假设100万请求）

| 服务 | 配置 | 月成本 (USD) |
|------|------|-------------|
| Kinesis Stream | 2 shards | $22 |
| Lambda | 512MB, 100ms/req | $10 |
| ECS Fargate | 1 task, 0.5vCPU/1GB | $15 |
| ElastiCache | cache.t3.micro | $12 |
| **总计** | | **~$59** |

### 优化建议

1. **使用Reserved Capacity** (Kinesis): 节省 30%
2. **Spot Instances** (ECS): 节省 70%
3. **Lambda预留并发**: 稳定性能
4. **CloudFront CDN**: 减少API调用

---

## 🔧 故障排查

### 问题 1: Lambda超时

**症状**: Lambda执行时间 > 30秒

**原因**:
- FastAPI服务响应慢
- Redis连接超时
- 批量大小过大

**解决**:
```bash
# 增加Lambda超时
aws lambda update-function-configuration \
  --function-name movie-rec-kinesis-handler \
  --timeout 60

# 减少批量大小
aws lambda update-event-source-mapping \
  --uuid <mapping-uuid> \
  --batch-size 50
```

### 问题 2: 高延迟

**症状**: P99延迟 > 100ms

**检查清单**:
- [ ] Redis连接正常
- [ ] FastAPI service健康
- [ ] ECS task资源充足
- [ ] 无网络瓶颈

**优化**:
```python
# 增加FastAPI workers
CMD ["uvicorn", "api.main:app", "--workers", "8"]

# 调整Redis连接池
redis_client = redis.Redis(max_connections=100)
```

### 问题 3: Kinesis Iterator Age增加

**症状**: 记录处理延迟增加

**原因**:
- Lambda处理速度不足
- 下游服务瓶颈

**解决**:
```bash
# 增加Kinesis分片
aws kinesis update-shard-count \
  --stream-name movie-rec-requests \
  --target-shard-count 4 \
  --scaling-type UNIFORM_SCALING

# 增加Lambda并发
aws lambda put-function-concurrency \
  --function-name movie-rec-kinesis-handler \
  --reserved-concurrent-executions 100
```

---

## 📚 相关文档

- [AWS Lambda开发指南](https://docs.aws.amazon.com/lambda/)
- [Kinesis Data Streams](https://docs.aws.amazon.com/kinesis/)
- [ECS Fargate最佳实践](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)
- [ElastiCache for Redis](https://docs.aws.amazon.com/elasticache/)

---

## ✅ 检查清单

部署前确认:

- [ ] Docker镜像已构建并推送到ECR
- [ ] Kinesis Stream已创建
- [ ] ElastiCache Redis集群运行中
- [ ] ECS服务健康
- [ ] Lambda函数已创建并配置触发器
- [ ] IAM角色权限正确
- [ ] 环境变量已配置
- [ ] 本地测试通过
- [ ] CloudWatch告警已设置

---

**架构状态**: ✅ 代码就绪，等待AWS部署

**性能目标**: 🎯 P99 < 100ms

**下一步**: 部署到AWS并进行负载测试


