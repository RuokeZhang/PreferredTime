# Dockerfile for FastAPI Recommendation Service
# 用于部署到AWS ECS/Fargate (ECR镜像)

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p models/saved_models logs database

# 暴露端口
EXPOSE 8082

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8082/health').raise_for_status()" || exit 1

# 启动命令 - 使用多worker提高并发性能
# Workers数量 = (2 x CPU cores) + 1
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8082", \
     "--workers", "4", \
     "--log-level", "info"]


