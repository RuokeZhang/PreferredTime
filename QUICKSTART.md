# 快速入门指南

这是一个快速开始使用Movie Recommendation Platform的指南。

## 第一步：环境准备

确保您已安装：
- Python 3.9+
- Docker 和 Docker Compose
- pip

## 第二步：一键启动

使用启动脚本快速启动所有服务：

```bash
chmod +x run.sh
./run.sh
```

脚本会自动：
1. 检查环境依赖
2. 安装Python包
3. 启动Kafka和Zookeeper
4. 初始化数据库
5. 启动Kafka消费者（后台）
6. 启动FastAPI服务

等待服务启动完成（约1-2分钟）。

## 第三步：生成测试数据

在**新的终端窗口**中运行测试数据生成器：

```bash
python3 test_producer.py
```

选择选项 **3** 生成真实数据集：
```
请选择操作 (1-5): 3
这将生成大量数据，是否继续？(y/n): y
```

这会生成：
- 100个用户
- 500部电影
- 约2000-3000条评分记录
- 包含活跃用户、普通用户、新用户和热门电影

等待数据生成完成（约2-5分钟）。

## 第四步：测试推荐API

### 1. 检查服务健康状态

```bash
curl http://localhost:8082/health
```

预期响应：
```json
{
  "status": "healthy",
  "recommender_loaded": true,
  "database_connected": true,
  "rating_matrix_shape": [100, 500]
}
```

### 2. 重新加载模型（基于新数据）

```bash
curl -X POST http://localhost:8082/reload
```

### 3. 获取用户推荐

为用户1获取推荐：
```bash
curl http://localhost:8082/recommend/1
```

预期响应（逗号分隔的电影ID）：
```
234,156,423,89,312,445,78,901,234,567,123,890,456,789,234,567,890,123,456,789
```

### 4. 获取用户画像

```bash
curl http://localhost:8082/user/1/profile
```

### 5. 获取相似电影

获取与电影100相似的电影：
```bash
curl http://localhost:8082/movie/100/similar?top_n=10
```

## 第五步：实时测试

### 发送新的评分事件

使用测试生成器为特定用户添加评分：

```bash
python3 test_producer.py
```

选择选项 **4**：
```
请选择操作 (1-5): 4
请输入用户ID: 999
请输入电影ID列表（逗号分隔）: 1,2,3,5,10
请输入对应的评分列表（逗号分隔）: 5.0,4.5,4.0,5.0,4.5
```

### 重新加载模型并查看新推荐

```bash
curl -X POST http://localhost:8082/reload
curl http://localhost:8082/recommend/999
```

## 常用命令备忘

### 启动服务
```bash
./run.sh
```

### 停止服务
```bash
./stop.sh
# 或按 Ctrl+C 停止FastAPI，然后运行：
# docker-compose down
```

### 仅启动API服务（假设Kafka已运行）
```bash
./start_api.sh
```

### 查看消费者日志
```bash
tail -f logs/consumer.log
```

### 重启Kafka（清空数据）
```bash
docker-compose down -v
docker-compose up -d
```

### 重新初始化数据库
```bash
rm database/movie_rec.db
python3 database/init_db.py
```

## API端点速查

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | API信息 |
| `/health` | GET | 健康检查 |
| `/recommend/{user_id}` | GET | 获取推荐（主要接口） |
| `/reload` | POST | 重新加载模型 |
| `/user/{user_id}/profile` | GET | 用户画像 |
| `/movie/{movie_id}/similar` | GET | 相似电影 |

## 故障排查

### Kafka连接失败
```bash
# 检查Kafka是否运行
docker ps

# 查看Kafka日志
docker logs kafka

# 重启Kafka
docker-compose restart kafka
```

### 推荐结果为空
```bash
# 检查是否有数据
python3 -c "from data_processor.data_storage import DataStorage; ds = DataStorage('database/movie_rec.db'); print(len(ds.get_all_ratings()))"

# 如果没有数据，运行测试生成器
python3 test_producer.py
```

### 端口被占用
如果8082端口被占用，修改 `config/config.yaml` 中的端口：
```yaml
api:
  port: 8083  # 改为其他端口
```

## 下一步

- 查看 `README.md` 了解更多详细信息
- 修改 `config/config.yaml` 调整推荐参数
- 探索 `models/` 目录了解推荐算法实现
- 集成到您的应用程序中

## 性能优化建议

1. **首次使用**：先生成足够的测试数据（建议1000+用户，1000+电影）
2. **定期重训练**：新数据较多时，调用 `/reload` 端点
3. **调整权重**：根据实际效果调整 `config.yaml` 中的协同过滤和基于内容的权重
4. **缓存优化**：生产环境中可以添加Redis缓存推荐结果

祝使用愉快！🎬


