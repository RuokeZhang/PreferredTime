import json
import yaml
import os
import sys
from kafka import KafkaConsumer
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor.data_storage import DataStorage
from data_processor.hybrid_storage import HybridStorage
from data_processor.feature_extractor import FeatureExtractor
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MovieRecConsumer:
    """Kafka消费者，处理用户评分事件"""
    
    def __init__(self, config: dict):
        self.config = config
        self.kafka_config = config['kafka']
        self.storage_mode = config.get('storage_mode', 'sqlite')
        
        # 初始化混合存储层（支持SQLite或AWS）
        self.data_storage = HybridStorage(config)
        
        # 初始化特征提取器（仍使用原有的DataStorage用于特征计算）
        # 注意：AWS模式下，特征提取会比较慢，因为需要从S3读取
        if self.storage_mode == 'sqlite':
            self.feature_extractor = FeatureExtractor(self.data_storage.sqlite_storage)
        else:
            # AWS模式下，特征提取需要特殊处理
            logger.warning("AWS模式下，实时特征提取性能较低")
        
        # 初始化Kafka消费者
        self.consumer = None
        self._init_consumer()
    
    def _init_consumer(self):
        """初始化Kafka消费者"""
        try:
            self.consumer = KafkaConsumer(
                self.kafka_config['topic'],
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                group_id=self.kafka_config['group_id'],
                auto_offset_reset=self.kafka_config['auto_offset_reset'],
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            logger.info(f"Kafka消费者已初始化，订阅topic: {self.kafka_config['topic']}")
        except Exception as e:
            logger.error(f"初始化Kafka消费者失败: {e}")
            raise
    
    def process_event(self, event: dict):
        """
        处理单个用户事件
        事件格式: {
            'user_id': int,
            'movie_id': int,
            'rating': float,
            'timestamp': str (可选)
        }
        """
        try:
            user_id = event.get('user_id')
            movie_id = event.get('movie_id')
            rating = event.get('rating')
            timestamp_str = event.get('timestamp')
            
            # 验证必需字段
            if user_id is None or movie_id is None or rating is None:
                logger.warning(f"事件缺少必需字段: {event}")
                return
            
            # 解析时间戳
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except:
                    timestamp = datetime.utcnow()
            else:
                timestamp = datetime.utcnow()
            
            # 保存评分记录
            success = self.data_storage.save_rating(user_id, movie_id, rating, timestamp)
            
            if success:
                # 更新用户和电影特征
                if self.storage_mode == 'sqlite':
                    # SQLite模式：实时计算特征
                    self.feature_extractor.extract_user_features(user_id)
                    self.feature_extractor.extract_movie_features(movie_id)
                else:
                    # AWS模式：特征由批处理计算（Airflow），这里只保存原始数据
                    logger.debug(f"AWS模式：原始数据已保存到S3，特征将由批处理更新")
                
                logger.info(f"处理事件成功: user_id={user_id}, movie_id={movie_id}, rating={rating}")
            else:
                logger.warning(f"处理事件失败: {event}")
        
        except Exception as e:
            logger.error(f"处理事件时出错: {e}, 事件: {event}")
    
    def start(self):
        """开始消费Kafka消息"""
        logger.info("开始消费Kafka消息...")
        
        try:
            for message in self.consumer:
                event = message.value
                logger.debug(f"收到消息: {event}")
                self.process_event(event)
        
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止消费...")
        except Exception as e:
            logger.error(f"消费消息时出错: {e}")
        finally:
            self.close()
    
    def close(self):
        """关闭消费者和数据库连接"""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka消费者已关闭")
        
        if self.data_storage:
            self.data_storage.close()
            logger.info("数据库连接已关闭")


def main():
    """主函数"""
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建并启动消费者
    consumer = MovieRecConsumer(config)
    consumer.start()


if __name__ == "__main__":
    main()

