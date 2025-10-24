# AWS部署指南

本指南说明如何将Movie Recommendation Platform从本地SQLite迁移到AWS S3 + DynamoDB架构。

## 架构概述

```
本地开发（SQLite）:
Kafka → Consumer → SQLite (ratings + features)

AWS生产（S3 + DynamoDB）:
Kafka → Consumer → {
    ├─ S3 Bronze层 (原始JSON事件)
    └─ DynamoDB (实时特征)
}
```

## 前置要求

### 1. AWS账户和凭证

确保你有：
- AWS账户
- IAM用户（具有S3和DynamoDB权限）
- AWS凭证配置

```bash
# 方法1: 使用AWS CLI配置
aws configure

# 方法2: 设置环境变量
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

新增的依赖：
- `boto3`: AWS SDK for Python
- `pyarrow`: Parquet文件支持

## 部署步骤

### 步骤1: 配置AWS参数

编辑 `config/config.yaml`:

```yaml
# 修改存储模式
storage_mode: "aws"  # 从 "sqlite" 改为 "aws"

aws:
  region: "us-east-1"  # 你的AWS区域
  
  s3:
    bucket: "movie-rec-data-lake"  # 你的S3 bucket名称
    bronze_prefix: "bronze/user-events/"
    silver_prefix: "silver/"
    gold_prefix: "gold/"
  
  dynamodb:
    user_features_table: "movie-rec-user-features"
    movie_features_table: "movie-rec-movie-features"
```

### 步骤2: 创建AWS资源

运行初始化脚本创建S3 bucket和DynamoDB表：

```bash
python3 setup_aws.py
```

这个脚本会：
- 创建S3 bucket
- 创建Bronze/Silver/Gold目录结构
- 创建DynamoDB表（用户特征和电影特征）

**检查资源创建：**

```bash
# 检查S3 bucket
aws s3 ls s3://movie-rec-data-lake/

# 检查DynamoDB表
aws dynamodb list-tables
```

### 步骤3: 启动服务

```bash
# 停止现有服务
./stop.sh

# 启动AWS模式的服务
./run.sh
```

### 步骤4: 验证部署

**1. 检查健康状态：**
```bash
curl http://localhost:8082/health
```

应该看到：
```json
{
  "status": "healthy",
  "storage_mode": "aws",
  "storage_info": {
    "mode": "aws",
    "s3_bucket": "movie-rec-data-lake",
    "dynamodb_tables": {...}
  }
}
```

**2. 发送测试数据：**
```bash
python3 test_producer.py
# 选择选项1发送100个事件
```

**3. 验证S3数据：**
```bash
# 查看S3中的原始事件
aws s3 ls s3://movie-rec-data-lake/bronze/user-events/ --recursive

# 下载一个事件查看
aws s3 cp s3://movie-rec-data-lake/bronze/user-events/date=2025-03-15/event_xxx.json -
```

**4. 验证DynamoDB数据：**
```bash
# 查询用户特征
aws dynamodb get-item \
  --table-name movie-rec-user-features \
  --key '{"user_id": {"N": "1"}}'

# 扫描表（查看所有记录）
aws dynamodb scan --table-name movie-rec-user-features --max-items 5
```

## 本地开发使用LocalStack

如果想在本地模拟AWS环境，可以使用LocalStack：

### 1. 启动LocalStack

```bash
# 使用Docker启动LocalStack
docker run -d \
  --name localstack \
  -p 4566:4566 \
  -e SERVICES=s3,dynamodb \
  localstack/localstack
```

### 2. 配置使用LocalStack

编辑 `config/config.yaml`:

```yaml
storage_mode: "aws"

aws:
  region: "us-east-1"
  endpoint_url: "http://localhost:4566"  # 取消注释这行
  
  s3:
    bucket: "movie-rec-data-lake"
  ...
```

### 3. 创建资源

```bash
# 使用setup_aws.py创建资源
python3 setup_aws.py

# 或手动创建
aws --endpoint-url=http://localhost:4566 s3 mb s3://movie-rec-data-lake
aws --endpoint-url=http://localhost:4566 dynamodb create-table \
  --table-name movie-rec-user-features \
  --attribute-definitions AttributeName=user_id,AttributeType=N \
  --key-schema AttributeName=user_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

### 4. 测试LocalStack

```bash
# 列出S3 buckets
aws --endpoint-url=http://localhost:4566 s3 ls

# 列出DynamoDB表
aws --endpoint-url=http://localhost:4566 dynamodb list-tables
```

## 数据流和架构

### 实时数据流

```
用户评分事件 
  → Kafka (user-events topic)
  → Consumer
  → {
      S3 Bronze层: 保存原始JSON
      (DynamoDB暂时不实时更新，由批处理更新)
    }
```

### 批处理流程（未来扩展）

```
S3 Bronze层
  → Airflow DAG (每天运行)
  → 读取昨天的事件
  → 计算用户和电影特征
  → {
      S3 Silver层: 保存Parquet格式特征
      DynamoDB: 更新实时特征
    }
  → 模型重训练
```

## 数据格式

### S3 Bronze层（原始事件）

**路径**: `s3://movie-rec-data-lake/bronze/user-events/date=YYYY-MM-DD/event_xxx.json`

**格式**: JSON
```json
{
  "user_id": 123,
  "movie_id": 456,
  "rating": 4.5,
  "timestamp": "2025-03-15T10:30:00"
}
```

### S3 Silver层（处理后特征）

**路径**: `s3://movie-rec-data-lake/silver/user-features/user_123.parquet`

**格式**: Parquet
```
user_id | avg_rating | rating_count | std_dev | last_update
--------|------------|--------------|---------|-------------
123     | 4.2        | 15           | 0.8     | 2025-03-15T...
```

### DynamoDB（实时特征）

**表**: `movie-rec-user-features`
```json
{
  "user_id": 123,
  "avg_rating": 4.2,
  "rating_count": 15,
  "std_dev": 0.8,
  "last_update": "2025-03-15T10:30:00"
}
```

## 成本估算

### S3成本
- 存储: $0.023/GB/月
- PUT请求: $0.005/1000请求
- GET请求: $0.0004/1000请求

**示例**: 100万条评分记录（~100MB）
- 存储: $0.002/月
- 写入: $5（一次性）
- 可忽略不计

### DynamoDB成本
- 按需模式（推荐）:
  - 写入: $1.25/百万次
  - 读取: $0.25/百万次
- 存储: $0.25/GB/月

**示例**: 10万用户，100万次读取/月
- 写入: $0.10
- 读取: $0.25
- 存储: ~$0.25
- **总计**: ~$0.60/月

## 故障排查

### 问题1: AWS凭证错误

```
botocore.exceptions.NoCredentialsError
```

**解决**:
```bash
aws configure
# 或
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

### 问题2: S3权限不足

```
AccessDenied when calling PutObject
```

**解决**: 确保IAM用户有以下权限：
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:PutObject",
    "s3:GetObject",
    "s3:ListBucket"
  ],
  "Resource": [
    "arn:aws:s3:::movie-rec-data-lake",
    "arn:aws:s3:::movie-rec-data-lake/*"
  ]
}
```

### 问题3: DynamoDB表已存在但配置不同

```
ResourceInUseException: Table already exists
```

**解决**: 删除并重新创建表
```bash
aws dynamodb delete-table --table-name movie-rec-user-features
python3 setup_aws.py
```

### 问题4: LocalStack连接失败

```
Could not connect to the endpoint URL
```

**解决**:
```bash
# 检查LocalStack是否运行
docker ps | grep localstack

# 重启LocalStack
docker restart localstack
```

## 切换回SQLite

如果需要切回本地SQLite模式：

1. 编辑 `config/config.yaml`:
```yaml
storage_mode: "sqlite"
```

2. 重启服务:
```bash
./stop.sh
./run.sh
```

## 下一步

1. **实现Airflow批处理管道** - 自动化特征计算
2. **添加MLflow跟踪** - 模型版本管理
3. **优化查询性能** - 添加缓存层（Redis）
4. **监控和告警** - CloudWatch集成
5. **数据备份** - S3版本控制和跨区域复制

## 参考资料

- [AWS SDK for Python (Boto3)](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [LocalStack文档](https://docs.localstack.cloud/)
- [S3最佳实践](https://docs.aws.amazon.com/AmazonS3/latest/userguide/best-practices.html)
- [DynamoDB最佳实践](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/best-practices.html)


