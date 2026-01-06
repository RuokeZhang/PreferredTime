"""
检查数据库中是否有数据
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor.data_storage import DataStorage
import yaml

def check_database():
    # 加载配置
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_path = config['database']['path']
    print(f"数据库路径: {db_path}")
    
    # 检查文件是否存在
    if not os.path.exists(db_path):
        print("❌ 数据库文件不存在！")
        return
    
    print("✓ 数据库文件存在")
    
    # 连接数据库
    ds = DataStorage(db_path)
    
    # 检查评分数据
    ratings = ds.get_all_ratings()
    print(f"\n评分记录数量: {len(ratings)}")
    
    if len(ratings) > 0:
        print(f"  示例评分: user_id={ratings[0][0]}, movie_id={ratings[0][1]}, rating={ratings[0][2]}")
    
    # 检查用户数量
    users = ds.get_all_user_ids()
    print(f"用户数量: {len(users)}")
    if users:
        print(f"  示例用户ID: {users[:5]}")
    
    # 检查电影数量
    movies = ds.get_all_movie_ids()
    print(f"电影数量: {len(movies)}")
    if movies:
        print(f"  示例电影ID: {movies[:5]}")
    
    ds.close()
    
    if len(ratings) == 0:
        print("\n❌ 数据库中没有数据！")
        print("\n可能的原因：")
        print("1. Kafka消费者没有运行")
        print("2. test_producer.py 没有成功发送数据到Kafka")
        print("3. Kafka连接有问题")
        print("\n建议操作：")
        print("1. 检查Kafka消费者日志: tail -f logs/consumer.log")
        print("2. 检查Kafka消费者是否运行: ps aux | grep kafka_consumer")
        print("3. 重新运行test_producer.py发送数据")
    else:
        print("\n✓ 数据库有数据！")
        print("请调用 /reload 端点重新加载模型:")
        print("  curl -X POST http://localhost:8082/reload")

if __name__ == "__main__":
    check_database()



