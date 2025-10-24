#!/bin/bash

# 仅启动FastAPI服务的脚本（假设Kafka和数据库已配置）

echo "启动FastAPI推荐服务..."
echo "API端点: http://localhost:8082"
echo "推荐接口: http://localhost:8082/recommend/<user_id>"
echo "按Ctrl+C停止服务"
echo "=========================================="

# 启动FastAPI
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8082 --reload

