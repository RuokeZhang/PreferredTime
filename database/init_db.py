import os
import sys
import yaml

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import Base, get_engine
from utils.logger import setup_logger

logger = setup_logger(__name__)


def init_database(db_path: str):
    """初始化数据库，创建所有表"""
    # 确保数据库目录存在
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # 创建引擎并建表
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    logger.info(f"数据库初始化完成: {db_path}")
    logger.info(f"创建的表: {list(Base.metadata.tables.keys())}")


if __name__ == "__main__":
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    db_path = config['database']['path']
    init_database(db_path)


