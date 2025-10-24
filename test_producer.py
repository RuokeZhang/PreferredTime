"""
Kafka测试数据生成器
用于生成模拟的用户评分事件并发送到Kafka
"""

import json
import random
import time
from datetime import datetime, timedelta
from kafka import KafkaProducer
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TestDataProducer:
    """测试数据生成器"""
    
    def __init__(self, bootstrap_servers='localhost:9092', topic='user-events'):
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info(f"Kafka生产者已初始化，连接到: {bootstrap_servers}")
    
    def generate_random_event(self, user_id_range=(1, 100), movie_id_range=(1, 500)):
        """生成随机的评分事件"""
        user_id = random.randint(*user_id_range)
        movie_id = random.randint(*movie_id_range)
        
        # 生成符合正态分布的评分（均值3.5，标准差1.0）
        rating = random.gauss(3.5, 1.0)
        rating = max(1.0, min(5.0, rating))  # 限制在1-5之间
        rating = round(rating, 1)
        
        # 生成过去30天内的随机时间戳
        days_ago = random.randint(0, 30)
        timestamp = datetime.utcnow() - timedelta(days=days_ago)
        
        event = {
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': timestamp.isoformat()
        }
        
        return event
    
    def send_event(self, event):
        """发送单个事件到Kafka"""
        try:
            future = self.producer.send(self.topic, event)
            future.get(timeout=10)
            logger.info(f"发送事件: user_id={event['user_id']}, movie_id={event['movie_id']}, rating={event['rating']}")
            return True
        except Exception as e:
            logger.error(f"发送事件失败: {e}")
            return False
    
    def send_batch_events(self, count=100, delay=0.1):
        """批量发送事件"""
        logger.info(f"开始发送 {count} 个测试事件...")
        
        success_count = 0
        for i in range(count):
            event = self.generate_random_event()
            if self.send_event(event):
                success_count += 1
            
            if delay > 0:
                time.sleep(delay)
            
            if (i + 1) % 10 == 0:
                logger.info(f"进度: {i + 1}/{count}")
        
        self.producer.flush()
        logger.info(f"批量发送完成: {success_count}/{count} 成功")
    
    def send_user_specific_events(self, user_id, movie_ids, ratings):
        """为特定用户发送指定的评分事件"""
        if len(movie_ids) != len(ratings):
            logger.error("movie_ids和ratings长度不匹配")
            return
        
        for movie_id, rating in zip(movie_ids, ratings):
            event = {
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.send_event(event)
        
        self.producer.flush()
    
    def generate_realistic_dataset(self):
        """生成更真实的数据集"""
        logger.info("生成真实数据集...")
        
        # 1. 生成一些活跃用户（评分很多）
        logger.info("生成活跃用户数据...")
        for user_id in range(1, 11):  # 10个活跃用户
            num_ratings = random.randint(50, 100)
            for _ in range(num_ratings):
                event = self.generate_random_event(
                    user_id_range=(user_id, user_id),
                    movie_id_range=(1, 500)
                )
                self.send_event(event)
                time.sleep(0.05)
        
        # 2. 生成一些普通用户（评分适中）
        logger.info("生成普通用户数据...")
        for user_id in range(11, 51):  # 40个普通用户
            num_ratings = random.randint(10, 30)
            for _ in range(num_ratings):
                event = self.generate_random_event(
                    user_id_range=(user_id, user_id),
                    movie_id_range=(1, 500)
                )
                self.send_event(event)
                time.sleep(0.05)
        
        # 3. 生成一些新用户（评分很少）
        logger.info("生成新用户数据...")
        for user_id in range(51, 101):  # 50个新用户
            num_ratings = random.randint(1, 5)
            for _ in range(num_ratings):
                event = self.generate_random_event(
                    user_id_range=(user_id, user_id),
                    movie_id_range=(1, 500)
                )
                self.send_event(event)
                time.sleep(0.05)
        
        # 4. 创建一些热门电影（被很多人评分）
        logger.info("生成热门电影数据...")
        popular_movies = [1, 2, 3, 5, 10, 20, 50, 100]
        for movie_id in popular_movies:
            num_ratings = random.randint(30, 50)
            for _ in range(num_ratings):
                user_id = random.randint(1, 100)
                rating = random.gauss(4.0, 0.5)  # 热门电影评分偏高
                rating = max(1.0, min(5.0, round(rating, 1)))
                
                event = {
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'timestamp': datetime.utcnow().isoformat()
                }
                self.send_event(event)
                time.sleep(0.05)
        
        self.producer.flush()
        logger.info("真实数据集生成完成！")
    
    def close(self):
        """关闭生产者"""
        if self.producer:
            self.producer.close()
            logger.info("Kafka生产者已关闭")


def main():
    """主函数"""
    print("=" * 50)
    print("Kafka测试数据生成器")
    print("=" * 50)
    print()
    print("选择操作:")
    print("1. 发送100个随机事件")
    print("2. 发送自定义数量的随机事件")
    print("3. 生成真实数据集（推荐用于首次初始化）")
    print("4. 为特定用户发送评分")
    print("5. 退出")
    print()
    
    try:
        producer = TestDataProducer()
        
        while True:
            choice = input("请选择操作 (1-5): ").strip()
            
            if choice == '1':
                producer.send_batch_events(count=100, delay=0.1)
            
            elif choice == '2':
                count = int(input("请输入要发送的事件数量: "))
                delay = float(input("请输入事件间隔（秒，0表示无延迟）: "))
                producer.send_batch_events(count=count, delay=delay)
            
            elif choice == '3':
                confirm = input("这将生成大量数据，是否继续？(y/n): ").strip().lower()
                if confirm == 'y':
                    producer.generate_realistic_dataset()
            
            elif choice == '4':
                user_id = int(input("请输入用户ID: "))
                movie_ids_str = input("请输入电影ID列表（逗号分隔）: ")
                ratings_str = input("请输入对应的评分列表（逗号分隔）: ")
                
                movie_ids = [int(x.strip()) for x in movie_ids_str.split(',')]
                ratings = [float(x.strip()) for x in ratings_str.split(',')]
                
                producer.send_user_specific_events(user_id, movie_ids, ratings)
            
            elif choice == '5':
                print("退出...")
                break
            
            else:
                print("无效的选择，请重试")
            
            print()
        
        producer.close()
    
    except KeyboardInterrupt:
        print("\n\n收到中断信号，退出...")
    except Exception as e:
        logger.error(f"发生错误: {e}")


if __name__ == "__main__":
    main()


