#!/bin/bash

# Movie Recommendation Platform 停止脚本

echo "=========================================="
echo "停止Movie Recommendation Platform"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 停止Kafka消费者
if [ -f logs/consumer.pid ]; then
    echo -e "${YELLOW}停止Kafka消费者...${NC}"
    CONSUMER_PID=$(cat logs/consumer.pid)
    kill $CONSUMER_PID 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Kafka消费者已停止${NC}"
        rm logs/consumer.pid
    else
        echo -e "${RED}Kafka消费者进程不存在或已停止${NC}"
    fi
else
    echo -e "${YELLOW}未找到Kafka消费者PID文件${NC}"
fi

# 停止Kafka和Zookeeper
echo -e "${YELLOW}停止Kafka和Zookeeper...${NC}"
docker-compose down
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Kafka和Zookeeper已停止${NC}"
else
    echo -e "${RED}停止Kafka失败${NC}"
fi

echo -e "${GREEN}=========================================="
echo -e "所有服务已停止"
echo -e "==========================================${NC}"



