"""
AWS资源初始化脚本
创建S3 bucket和DynamoDB表
"""
import yaml
import boto3
import sys
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_config():
    """加载配置文件"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def setup_s3(config):
    """创建S3 bucket"""
    logger.info("设置S3资源...")
    
    region = config['aws']['region']
    bucket = config['aws']['s3']['bucket']
    
    # 如果配置了endpoint_url（LocalStack），使用它
    endpoint_url = config['aws'].get('endpoint_url')
    if endpoint_url:
        s3_client = boto3.client('s3', region_name=region, endpoint_url=endpoint_url)
    else:
        s3_client = boto3.client('s3', region_name=region)
    
    try:
        # 检查bucket是否存在
        s3_client.head_bucket(Bucket=bucket)
        logger.info(f"✓ S3 bucket已存在: {bucket}")
    except:
        try:
            # 创建bucket
            if region == 'us-east-1':
                s3_client.create_bucket(Bucket=bucket)
            else:
                s3_client.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )
            logger.info(f"✓ 创建S3 bucket成功: {bucket}")
        except Exception as e:
            logger.error(f"✗ 创建S3 bucket失败: {e}")
            return False
    
    # 创建目录结构（通过放置空对象）
    prefixes = [
        config['aws']['s3']['bronze_prefix'],
        config['aws']['s3']['silver_prefix'] + 'user-features/',
        config['aws']['s3']['silver_prefix'] + 'movie-features/',
        config['aws']['s3']['gold_prefix']
    ]
    
    for prefix in prefixes:
        try:
            s3_client.put_object(Bucket=bucket, Key=prefix + '.keep', Body=b'')
            logger.info(f"✓ 创建S3目录: s3://{bucket}/{prefix}")
        except Exception as e:
            logger.warning(f"创建目录失败 {prefix}: {e}")
    
    return True


def setup_dynamodb(config):
    """创建DynamoDB表"""
    logger.info("设置DynamoDB资源...")
    
    region = config['aws']['region']
    user_table_name = config['aws']['dynamodb']['user_features_table']
    movie_table_name = config['aws']['dynamodb']['movie_features_table']
    
    # 如果配置了endpoint_url（LocalStack），使用它
    endpoint_url = config['aws'].get('endpoint_url')
    if endpoint_url:
        dynamodb = boto3.resource('dynamodb', region_name=region, endpoint_url=endpoint_url)
    else:
        dynamodb = boto3.resource('dynamodb', region_name=region)
    
    # 创建用户特征表
    try:
        user_table = dynamodb.Table(user_table_name)
        user_table.table_status
        logger.info(f"✓ DynamoDB表已存在: {user_table_name}")
    except:
        try:
            table = dynamodb.create_table(
                TableName=user_table_name,
                KeySchema=[
                    {'AttributeName': 'user_id', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'user_id', 'AttributeType': 'N'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            table.wait_until_exists()
            logger.info(f"✓ 创建DynamoDB表成功: {user_table_name}")
        except Exception as e:
            logger.error(f"✗ 创建DynamoDB表失败 {user_table_name}: {e}")
            return False
    
    # 创建电影特征表
    try:
        movie_table = dynamodb.Table(movie_table_name)
        movie_table.table_status
        logger.info(f"✓ DynamoDB表已存在: {movie_table_name}")
    except:
        try:
            table = dynamodb.create_table(
                TableName=movie_table_name,
                KeySchema=[
                    {'AttributeName': 'movie_id', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'movie_id', 'AttributeType': 'N'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            table.wait_until_exists()
            logger.info(f"✓ 创建DynamoDB表成功: {movie_table_name}")
        except Exception as e:
            logger.error(f"✗ 创建DynamoDB表失败 {movie_table_name}: {e}")
            return False
    
    return True


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("AWS资源初始化")
    logger.info("=" * 60)
    
    # 加载配置
    config = load_config()
    
    if config.get('storage_mode') != 'aws':
        logger.warning(f"当前storage_mode是'{config.get('storage_mode')}'，不是'aws'")
        response = input("是否继续创建AWS资源？(y/n): ")
        if response.lower() != 'y':
            logger.info("取消操作")
            return
    
    # 显示配置信息
    logger.info("\nAWS配置:")
    logger.info(f"  Region: {config['aws']['region']}")
    logger.info(f"  S3 Bucket: {config['aws']['s3']['bucket']}")
    logger.info(f"  DynamoDB User Table: {config['aws']['dynamodb']['user_features_table']}")
    logger.info(f"  DynamoDB Movie Table: {config['aws']['dynamodb']['movie_features_table']}")
    
    if config['aws'].get('endpoint_url'):
        logger.info(f"  Endpoint URL: {config['aws']['endpoint_url']} (LocalStack模式)")
    
    print()
    
    # 设置S3
    if not setup_s3(config):
        logger.error("S3设置失败")
        sys.exit(1)
    
    print()
    
    # 设置DynamoDB
    if not setup_dynamodb(config):
        logger.error("DynamoDB设置失败")
        sys.exit(1)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ AWS资源初始化完成！")
    logger.info("=" * 60)
    
    logger.info("\n下一步:")
    logger.info("1. 修改 config/config.yaml 中的 storage_mode 为 'aws'")
    logger.info("2. 配置AWS凭证 (aws configure 或环境变量)")
    logger.info("3. 重启服务: ./run.sh")


if __name__ == "__main__":
    main()



