#!/bin/bash

# 检查并清理占用8082端口的进程

echo "检查端口8082的使用情况..."

# 查找占用8082端口的进程
PID=$(lsof -ti:8082)

if [ -z "$PID" ]; then
    echo "✓ 端口8082未被占用"
else
    echo "端口8082被进程 $PID 占用"
    echo "进程详情:"
    ps -p $PID -o pid,command
    echo ""
    read -p "是否要杀掉该进程? (y/n): " answer
    
    if [ "$answer" = "y" ]; then
        kill -9 $PID
        echo "✓ 进程已终止"
    else
        echo "操作取消"
    fi
fi


