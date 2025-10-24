# Lambda容器镜像
# 用于部署到AWS Lambda（Container Image支持）
FROM public.ecr.aws/lambda/python:3.11

# 复制依赖文件并安装
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# 复制Lambda函数代码
COPY recommendation_handler.py ${LAMBDA_TASK_ROOT}/

# 设置Lambda处理器
CMD ["recommendation_handler.lambda_handler"]


