"""
S3数据存储层 - 处理原始数据和批处理特征
"""
import json
import boto3
import pandas as pd
from datetime import datetime
from io import BytesIO
from typing import List, Tuple, Dict, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class S3Storage:
    """S3存储管理类 - 数据湖实现"""
    
    def __init__(self, config: dict):
        """
        初始化S3客户端
        
        Args:
            config: AWS配置字典
        """
        self.region = config.get('region', 'us-east-1')
        self.bucket = config['s3']['bucket']
        self.bronze_prefix = config['s3']['bronze_prefix']
        self.silver_prefix = config['s3']['silver_prefix']
        self.gold_prefix = config['s3']['gold_prefix']
        
        # 支持LocalStack用于本地开发
        endpoint_url = config.get('endpoint_url')
        if endpoint_url:
            self.s3_client = boto3.client('s3', 
                                         region_name=self.region,
                                         endpoint_url=endpoint_url)
            logger.info(f"S3客户端初始化完成（LocalStack模式）: {endpoint_url}")
        else:
            self.s3_client = boto3.client('s3', region_name=self.region)
            logger.info(f"S3客户端初始化完成: {self.bucket}")
    
    def save_raw_event(self, event: dict) -> bool:
        """
        保存原始评分事件到S3 Bronze层
        
        Args:
            event: 评分事件字典
        
        Returns:
            是否保存成功
        """
        try:
            # 构建S3路径：按日期分区
            timestamp = event.get('timestamp', datetime.utcnow().isoformat())
            if isinstance(timestamp, str):
                date = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d')
            else:
                date = timestamp.strftime('%Y-%m-%d')
            
            # 生成唯一的文件名
            import uuid
            key = f"{self.bronze_prefix}date={date}/event_{uuid.uuid4()}.json"
            
            # 写入S3
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(event, default=str),
                ContentType='application/json'
            )
            
            logger.debug(f"保存原始事件到S3: s3://{self.bucket}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"保存原始事件到S3失败: {e}")
            return False
    
    def save_user_features_silver(self, user_id: int, features: dict) -> bool:
        """
        保存用户特征到S3 Silver层（Parquet格式）
        
        Args:
            user_id: 用户ID
            features: 特征字典
        
        Returns:
            是否保存成功
        """
        try:
            # 构建DataFrame
            df = pd.DataFrame([{
                'user_id': user_id,
                'avg_rating': features.get('avg_rating', 0.0),
                'rating_count': features.get('rating_count', 0),
                'std_dev': features.get('std_dev', 0.0),
                'last_update': datetime.utcnow().isoformat()
            }])
            
            # 转换为Parquet
            parquet_buffer = BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            
            # 写入S3
            key = f"{self.silver_prefix}user-features/user_{user_id}.parquet"
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=parquet_buffer.getvalue(),
                ContentType='application/octet-stream'
            )
            
            logger.debug(f"保存用户特征到S3 Silver层: user_id={user_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存用户特征到S3失败: {e}")
            return False
    
    def save_movie_features_silver(self, movie_id: int, features: dict) -> bool:
        """
        保存电影特征到S3 Silver层（Parquet格式）
        
        Args:
            movie_id: 电影ID
            features: 特征字典
        
        Returns:
            是否保存成功
        """
        try:
            # 构建DataFrame
            df = pd.DataFrame([{
                'movie_id': movie_id,
                'avg_rating': features.get('avg_rating', 0.0),
                'rating_count': features.get('rating_count', 0),
                'popularity': features.get('popularity', 0.0),
                'last_update': datetime.utcnow().isoformat()
            }])
            
            # 转换为Parquet
            parquet_buffer = BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            
            # 写入S3
            key = f"{self.silver_prefix}movie-features/movie_{movie_id}.parquet"
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=parquet_buffer.getvalue(),
                ContentType='application/octet-stream'
            )
            
            logger.debug(f"保存电影特征到S3 Silver层: movie_id={movie_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存电影特征到S3失败: {e}")
            return False
    
    def read_raw_events_by_date(self, date: str) -> List[dict]:
        """
        读取指定日期的所有原始事件
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
        
        Returns:
            事件列表
        """
        try:
            prefix = f"{self.bronze_prefix}date={date}/"
            
            # 列出所有对象
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            events = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # 读取每个文件
                    obj_response = self.s3_client.get_object(
                        Bucket=self.bucket,
                        Key=obj['Key']
                    )
                    content = obj_response['Body'].read().decode('utf-8')
                    event = json.loads(content)
                    events.append(event)
            
            logger.info(f"从S3读取 {len(events)} 个事件: date={date}")
            return events
            
        except Exception as e:
            logger.error(f"从S3读取事件失败: {e}")
            return []
    
    def list_dates_in_bronze(self) -> List[str]:
        """
        列出Bronze层所有的日期分区
        
        Returns:
            日期列表
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.bronze_prefix,
                Delimiter='/'
            )
            
            dates = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    # 提取日期：bronze/user-events/date=2025-03-15/
                    date_str = prefix['Prefix'].split('date=')[1].rstrip('/')
                    dates.append(date_str)
            
            return sorted(dates)
            
        except Exception as e:
            logger.error(f"列出日期分区失败: {e}")
            return []
    
    def create_bucket_if_not_exists(self):
        """创建S3 bucket（如果不存在）"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            logger.info(f"S3 bucket已存在: {self.bucket}")
        except:
            try:
                if self.region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.bucket)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )
                logger.info(f"创建S3 bucket成功: {self.bucket}")
            except Exception as e:
                logger.error(f"创建S3 bucket失败: {e}")



