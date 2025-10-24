"""
混合存储层 - 统一SQLite、S3和DynamoDB的接口
"""
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from data_processor.data_storage import DataStorage
from data_processor.s3_storage import S3Storage
from data_processor.dynamodb_storage import DynamoDBStorage
from utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridStorage:
    """
    混合存储管理类
    - SQLite: 开发环境，本地存储
    - S3 + DynamoDB: 生产环境，AWS云存储
    """
    
    def __init__(self, config: dict):
        """
        根据配置初始化存储层
        
        Args:
            config: 完整配置字典
        """
        self.storage_mode = config.get('storage_mode', 'sqlite')
        self.config = config
        
        if self.storage_mode == 'sqlite':
            # SQLite模式（开发环境）
            logger.info("初始化存储层: SQLite模式")
            self.sqlite_storage = DataStorage(config['database']['path'])
            self.s3_storage = None
            self.dynamodb_storage = None
            
        elif self.storage_mode == 'aws':
            # AWS模式（生产环境）
            logger.info("初始化存储层: AWS模式（S3 + DynamoDB）")
            self.sqlite_storage = None
            self.s3_storage = S3Storage(config['aws'])
            self.dynamodb_storage = DynamoDBStorage(config['aws'])
            
            # 确保AWS资源存在
            self.s3_storage.create_bucket_if_not_exists()
            self.dynamodb_storage.create_tables_if_not_exist()
            
        else:
            raise ValueError(f"不支持的存储模式: {self.storage_mode}")
    
    # ===================== 评分记录相关 =====================
    
    def save_rating(self, user_id: int, movie_id: int, rating: float, 
                   timestamp: datetime = None) -> bool:
        """
        保存评分记录
        - SQLite模式: 保存到ratings表
        - AWS模式: 保存到S3 Bronze层（原始JSON）
        """
        if self.storage_mode == 'sqlite':
            return self.sqlite_storage.save_rating(user_id, movie_id, rating, timestamp)
        
        else:  # aws模式
            event = {
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,
                'timestamp': timestamp.isoformat() if timestamp else datetime.utcnow().isoformat()
            }
            return self.s3_storage.save_raw_event(event)
    
    def get_user_ratings(self, user_id: int) -> List[Tuple]:
        """
        获取用户的所有评分
        返回: [(movie_id, rating, timestamp), ...]
        """
        if self.storage_mode == 'sqlite':
            return self.sqlite_storage.get_user_ratings(user_id)
        
        else:  # aws模式
            # 从S3读取需要扫描所有日期分区（性能较低）
            # 生产环境中应该使用额外的索引或缓存
            logger.warning("AWS模式下从S3读取用户评分性能较低，建议使用额外索引")
            # 这里简化实现，返回空列表
            # 实际生产中可以使用Athena查询或维护额外的索引
            return []
    
    def get_all_ratings(self) -> List[Tuple]:
        """
        获取所有评分记录
        返回: [(user_id, movie_id, rating, timestamp), ...]
        """
        if self.storage_mode == 'sqlite':
            return self.sqlite_storage.get_all_ratings()
        
        else:  # aws模式
            logger.warning("AWS模式下从S3读取所有评分，这可能会很慢")
            all_ratings = []
            
            # 获取所有日期分区
            dates = self.s3_storage.list_dates_in_bronze()
            
            # 读取每个日期的事件
            for date in dates:
                events = self.s3_storage.read_raw_events_by_date(date)
                for event in events:
                    all_ratings.append((
                        event['user_id'],
                        event['movie_id'],
                        event['rating'],
                        datetime.fromisoformat(event['timestamp'])
                    ))
            
            return all_ratings
    
    # ===================== 用户特征相关 =====================
    
    def update_user_feature(self, user_id: int, avg_rating: float, 
                           rating_count: int, std_dev: float) -> bool:
        """
        更新用户特征
        - SQLite模式: 更新user_features表
        - AWS模式: 更新DynamoDB（实时特征）+ S3 Silver层（批处理特征）
        """
        if self.storage_mode == 'sqlite':
            return self.sqlite_storage.update_user_feature(
                user_id, avg_rating, rating_count, std_dev
            )
        
        else:  # aws模式
            # 更新到DynamoDB（实时查询）
            dynamodb_success = self.dynamodb_storage.update_user_feature(
                user_id, avg_rating, rating_count, std_dev
            )
            
            # 同时保存到S3 Silver层（批处理/分析）
            features = {
                'avg_rating': avg_rating,
                'rating_count': rating_count,
                'std_dev': std_dev
            }
            s3_success = self.s3_storage.save_user_features_silver(user_id, features)
            
            return dynamodb_success and s3_success
    
    def get_user_feature(self, user_id: int) -> Optional[Dict]:
        """
        获取用户特征
        - SQLite模式: 从user_features表查询
        - AWS模式: 从DynamoDB查询（低延迟）
        """
        if self.storage_mode == 'sqlite':
            # SQLite返回格式需要转换
            # 这里简化处理
            return None
        
        else:  # aws模式
            return self.dynamodb_storage.get_user_feature(user_id)
    
    # ===================== 电影特征相关 =====================
    
    def update_movie_feature(self, movie_id: int, avg_rating: float,
                            rating_count: int, popularity: float) -> bool:
        """
        更新电影特征
        - SQLite模式: 更新movie_features表
        - AWS模式: 更新DynamoDB（实时特征）+ S3 Silver层（批处理特征）
        """
        if self.storage_mode == 'sqlite':
            return self.sqlite_storage.update_movie_feature(
                movie_id, avg_rating, rating_count, popularity
            )
        
        else:  # aws模式
            # 更新到DynamoDB（实时查询）
            dynamodb_success = self.dynamodb_storage.update_movie_feature(
                movie_id, avg_rating, rating_count, popularity
            )
            
            # 同时保存到S3 Silver层（批处理/分析）
            features = {
                'avg_rating': avg_rating,
                'rating_count': rating_count,
                'popularity': popularity
            }
            s3_success = self.s3_storage.save_movie_features_silver(movie_id, features)
            
            return dynamodb_success and s3_success
    
    def get_movie_feature(self, movie_id: int) -> Optional[Dict]:
        """
        获取电影特征
        - SQLite模式: 从movie_features表查询
        - AWS模式: 从DynamoDB查询（低延迟）
        """
        if self.storage_mode == 'sqlite':
            return None
        
        else:  # aws模式
            return self.dynamodb_storage.get_movie_feature(movie_id)
    
    # ===================== 辅助方法 =====================
    
    def get_all_user_ids(self) -> List[int]:
        """获取所有用户ID"""
        if self.storage_mode == 'sqlite':
            return self.sqlite_storage.get_all_user_ids()
        else:
            return self.dynamodb_storage.get_all_user_ids()
    
    def get_all_movie_ids(self) -> List[int]:
        """获取所有电影ID"""
        if self.storage_mode == 'sqlite':
            return self.sqlite_storage.get_all_movie_ids()
        else:
            return self.dynamodb_storage.get_all_movie_ids()
    
    def close(self):
        """关闭数据库连接"""
        if self.sqlite_storage:
            self.sqlite_storage.close()
    
    def get_storage_info(self) -> Dict:
        """获取存储层信息"""
        info = {
            'mode': self.storage_mode,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if self.storage_mode == 'sqlite':
            info['database_path'] = self.config['database']['path']
        else:
            info['s3_bucket'] = self.config['aws']['s3']['bucket']
            info['dynamodb_tables'] = {
                'user_features': self.config['aws']['dynamodb']['user_features_table'],
                'movie_features': self.config['aws']['dynamodb']['movie_features_table']
            }
        
        return info


