#!/bin/bash

# Movie Recommendation Platform 启动脚本

echo "=========================================="
echo "Movie Recommendation Platform"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查Python
echo -e "${YELLOW}检查Python环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到Python3${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3已安装${NC}"

# 检查Docker
echo -e "${YELLOW}检查Docker环境...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: 未找到Docker${NC}"
    echo "请先安装Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker已安装${NC}"

# 检查docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}错误: 未找到docker-compose${NC}"
    echo "请先安装docker-compose"
    exit 1
fi
echo -e "${GREEN}✓ docker-compose已安装${NC}"

# 安装Python依赖
echo -e "${YELLOW}安装Python依赖...${NC}"
pip3 install -r requirements.txt --quiet
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python依赖安装完成${NC}"
else
    echo -e "${RED}错误: Python依赖安装失败${NC}"
    exit 1
fi

# 启动Kafka和Zookeeper
echo -e "${YELLOW}启动Kafka和Zookeeper...${NC}"
docker-compose up -d
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Kafka和Zookeeper已启动${NC}"
else
    echo -e "${RED}错误: Kafka启动失败${NC}"
    exit 1
fi

# 等待Kafka启动
echo -e "${YELLOW}等待Kafka完全启动（30秒）...${NC}"
sleep 30
echo -e "${GREEN}✓ Kafka准备就绪${NC}"

# 初始化数据库
echo -e "${YELLOW}初始化数据库...${NC}"
python3 database/init_db.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 数据库初始化完成${NC}"
else
    echo -e "${RED}错误: 数据库初始化失败${NC}"
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 启动Kafka消费者（后台运行）
echo -e "${YELLOW}启动Kafka消费者...${NC}"
nohup python3 -m kafka_consumer.consumer > logs/consumer.log 2>&1 &
CONSUMER_PID=$!
echo $CONSUMER_PID > logs/consumer.pid
echo -e "${GREEN}✓ Kafka消费者已启动 (PID: $CONSUMER_PID)${NC}"

# 启动FastAPI服务
echo -e "${YELLOW}启动FastAPI推荐服务...${NC}"
echo -e "${GREEN}=========================================="
echo -e "推荐服务将在端口8082上运行"
echo -e "API端点: http://localhost:8082"
echo -e "推荐接口: http://localhost:8082/recommend/<user_id>"
echo -e "健康检查: http://localhost:8082/health"
echo -e "==========================================${NC}"

# 启动FastAPI
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8082 --log-level info

