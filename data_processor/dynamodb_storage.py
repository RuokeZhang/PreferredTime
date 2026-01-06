"""
DynamoDB特征存储层 - 处理实时特征读写
"""
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DynamoDBStorage:
    """DynamoDB存储管理类 - Feature Store实现"""
    
    def __init__(self, config: dict):
        """
        初始化DynamoDB客户端
        
        Args:
            config: AWS配置字典
        """
        self.region = config.get('region', 'us-east-1')
        self.user_features_table_name = config['dynamodb']['user_features_table']
        self.movie_features_table_name = config['dynamodb']['movie_features_table']
        
        # 支持LocalStack用于本地开发
        endpoint_url = config.get('endpoint_url')
        if endpoint_url:
            self.dynamodb = boto3.resource('dynamodb',
                                          region_name=self.region,
                                          endpoint_url=endpoint_url)
            logger.info(f"DynamoDB客户端初始化完成（LocalStack模式）: {endpoint_url}")
        else:
            self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
            logger.info("DynamoDB客户端初始化完成")
        
        self.user_features_table = self.dynamodb.Table(self.user_features_table_name)
        self.movie_features_table = self.dynamodb.Table(self.movie_features_table_name)
    
    def update_user_feature(self, user_id: int, avg_rating: float, 
                           rating_count: int, std_dev: float) -> bool:
        """
        更新用户特征到DynamoDB
        
        Args:
            user_id: 用户ID
            avg_rating: 平均评分
            rating_count: 评分数量
            std_dev: 标准差
        
        Returns:
            是否更新成功
        """
        try:
            self.user_features_table.put_item(
                Item={
                    'user_id': user_id,
                    'avg_rating': Decimal(str(round(avg_rating, 2))),
                    'rating_count': rating_count,
                    'std_dev': Decimal(str(round(std_dev, 2))),
                    'last_update': datetime.utcnow().isoformat()
                }
            )
            logger.debug(f"更新用户特征到DynamoDB: user_id={user_id}")
            return True
            
        except Exception as e:
            logger.error(f"更新用户特征到DynamoDB失败: {e}")
            return False
    
    def update_movie_feature(self, movie_id: int, avg_rating: float,
                            rating_count: int, popularity: float) -> bool:
        """
        更新电影特征到DynamoDB
        
        Args:
            movie_id: 电影ID
            avg_rating: 平均评分
            rating_count: 评分数量
            popularity: 流行度
        
        Returns:
            是否更新成功
        """
        try:
            self.movie_features_table.put_item(
                Item={
                    'movie_id': movie_id,
                    'avg_rating': Decimal(str(round(avg_rating, 2))),
                    'rating_count': rating_count,
                    'popularity': Decimal(str(round(popularity, 2))),
                    'last_update': datetime.utcnow().isoformat()
                }
            )
            logger.debug(f"更新电影特征到DynamoDB: movie_id={movie_id}")
            return True
            
        except Exception as e:
            logger.error(f"更新电影特征到DynamoDB失败: {e}")
            return False
    
    def get_user_feature(self, user_id: int) -> Optional[Dict]:
        """
        获取用户特征
        
        Args:
            user_id: 用户ID
        
        Returns:
            特征字典或None
        """
        try:
            response = self.user_features_table.get_item(
                Key={'user_id': user_id}
            )
            
            if 'Item' in response:
                item = response['Item']
                # 将Decimal转换为float
                return {
                    'user_id': item['user_id'],
                    'avg_rating': float(item.get('avg_rating', 0)),
                    'rating_count': item.get('rating_count', 0),
                    'std_dev': float(item.get('std_dev', 0)),
                    'last_update': item.get('last_update')
                }
            return None
            
        except Exception as e:
            logger.error(f"从DynamoDB获取用户特征失败: {e}")
            return None
    
    def get_movie_feature(self, movie_id: int) -> Optional[Dict]:
        """
        获取电影特征
        
        Args:
            movie_id: 电影ID
        
        Returns:
            特征字典或None
        """
        try:
            response = self.movie_features_table.get_item(
                Key={'movie_id': movie_id}
            )
            
            if 'Item' in response:
                item = response['Item']
                # 将Decimal转换为float
                return {
                    'movie_id': item['movie_id'],
                    'avg_rating': float(item.get('avg_rating', 0)),
                    'rating_count': item.get('rating_count', 0),
                    'popularity': float(item.get('popularity', 0)),
                    'last_update': item.get('last_update')
                }
            return None
            
        except Exception as e:
            logger.error(f"从DynamoDB获取电影特征失败: {e}")
            return None
    
    def batch_get_user_features(self, user_ids: List[int]) -> Dict[int, Dict]:
        """
        批量获取用户特征
        
        Args:
            user_ids: 用户ID列表
        
        Returns:
            {user_id: features}字典
        """
        try:
            keys = [{'user_id': uid} for uid in user_ids]
            
            response = self.dynamodb.batch_get_item(
                RequestItems={
                    self.user_features_table_name: {
                        'Keys': keys
                    }
                }
            )
            
            results = {}
            if 'Responses' in response and self.user_features_table_name in response['Responses']:
                for item in response['Responses'][self.user_features_table_name]:
                    user_id = item['user_id']
                    results[user_id] = {
                        'avg_rating': float(item.get('avg_rating', 0)),
                        'rating_count': item.get('rating_count', 0),
                        'std_dev': float(item.get('std_dev', 0))
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"批量获取用户特征失败: {e}")
            return {}
    
    def get_all_user_ids(self) -> List[int]:
        """
        获取所有用户ID（用于批处理）
        
        Returns:
            用户ID列表
        """
        try:
            user_ids = []
            
            # 扫描表（注意：生产环境中应该使用更高效的方法）
            response = self.user_features_table.scan(
                ProjectionExpression='user_id'
            )
            
            user_ids.extend([item['user_id'] for item in response.get('Items', [])])
            
            # 处理分页
            while 'LastEvaluatedKey' in response:
                response = self.user_features_table.scan(
                    ProjectionExpression='user_id',
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                user_ids.extend([item['user_id'] for item in response.get('Items', [])])
            
            return user_ids
            
        except Exception as e:
            logger.error(f"获取所有用户ID失败: {e}")
            return []
    
    def get_all_movie_ids(self) -> List[int]:
        """
        获取所有电影ID（用于批处理）
        
        Returns:
            电影ID列表
        """
        try:
            movie_ids = []
            
            response = self.movie_features_table.scan(
                ProjectionExpression='movie_id'
            )
            
            movie_ids.extend([item['movie_id'] for item in response.get('Items', [])])
            
            # 处理分页
            while 'LastEvaluatedKey' in response:
                response = self.movie_features_table.scan(
                    ProjectionExpression='movie_id',
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                movie_ids.extend([item['movie_id'] for item in response.get('Items', [])])
            
            return movie_ids
            
        except Exception as e:
            logger.error(f"获取所有电影ID失败: {e}")
            return []
    
    def create_tables_if_not_exist(self):
        """创建DynamoDB表（如果不存在）"""
        try:
            # 创建用户特征表
            try:
                self.user_features_table.table_status
                logger.info(f"DynamoDB表已存在: {self.user_features_table_name}")
            except:
                table = self.dynamodb.create_table(
                    TableName=self.user_features_table_name,
                    KeySchema=[
                        {'AttributeName': 'user_id', 'KeyType': 'HASH'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'user_id', 'AttributeType': 'N'}
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )
                table.wait_until_exists()
                logger.info(f"创建DynamoDB表成功: {self.user_features_table_name}")
            
            # 创建电影特征表
            try:
                self.movie_features_table.table_status
                logger.info(f"DynamoDB表已存在: {self.movie_features_table_name}")
            except:
                table = self.dynamodb.create_table(
                    TableName=self.movie_features_table_name,
                    KeySchema=[
                        {'AttributeName': 'movie_id', 'KeyType': 'HASH'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'movie_id', 'AttributeType': 'N'}
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )
                table.wait_until_exists()
                logger.info(f"创建DynamoDB表成功: {self.movie_features_table_name}")
                
        except Exception as e:
            logger.error(f"创建DynamoDB表失败: {e}")



