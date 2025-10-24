import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
from data_processor.data_storage import DataStorage
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, data_storage: DataStorage):
        self.data_storage = data_storage
    
    def extract_user_features(self, user_id: int) -> Dict:
        """
        提取用户特征
        返回: {
            'avg_rating': 平均评分,
            'rating_count': 评分数量,
            'std_dev': 评分标准差,
            'rated_movies': 已评分的电影ID列表
        }
        """
        ratings = self.data_storage.get_user_ratings(user_id)
        
        if not ratings:
            return {
                'avg_rating': 0.0,
                'rating_count': 0,
                'std_dev': 0.0,
                'rated_movies': []
            }
        
        rating_values = [r[1] for r in ratings]  # r[1]是rating值
        movie_ids = [r[0] for r in ratings]  # r[0]是movie_id
        
        features = {
            'avg_rating': float(np.mean(rating_values)),
            'rating_count': len(rating_values),
            'std_dev': float(np.std(rating_values)),
            'rated_movies': movie_ids
        }
        
        # 更新到数据库
        self.data_storage.update_user_feature(
            user_id,
            features['avg_rating'],
            features['rating_count'],
            features['std_dev']
        )
        
        return features
    
    def extract_movie_features(self, movie_id: int) -> Dict:
        """
        提取电影特征
        返回: {
            'avg_rating': 平均评分,
            'rating_count': 评分数量,
            'popularity': 流行度（基于评分数量）
        }
        """
        ratings = self.data_storage.get_movie_ratings(movie_id)
        
        if not ratings:
            return {
                'avg_rating': 0.0,
                'rating_count': 0,
                'popularity': 0.0
            }
        
        rating_values = [r[1] for r in ratings]  # r[1]是rating值
        
        # 计算流行度：评分数量的对数归一化
        popularity = np.log1p(len(rating_values))
        
        features = {
            'avg_rating': float(np.mean(rating_values)),
            'rating_count': len(rating_values),
            'popularity': float(popularity)
        }
        
        # 更新到数据库
        self.data_storage.update_movie_feature(
            movie_id,
            features['avg_rating'],
            features['rating_count'],
            features['popularity']
        )
        
        return features
    
    def build_user_item_matrix(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        构建用户-电影评分矩阵
        返回: (评分矩阵DataFrame, user_id到索引的映射, movie_id到索引的映射)
        """
        ratings = self.data_storage.get_all_ratings()
        
        if not ratings:
            logger.warning("没有评分数据，返回空矩阵")
            return pd.DataFrame(), {}, {}
        
        # 构建DataFrame
        df = pd.DataFrame(ratings, columns=['user_id', 'movie_id', 'rating', 'timestamp'])
        
        # 创建数据透视表（用户-电影评分矩阵）
        rating_matrix = df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        # 创建ID到索引的映射
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(rating_matrix.index)}
        movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(rating_matrix.columns)}
        
        logger.info(f"构建评分矩阵: {rating_matrix.shape[0]} 用户 x {rating_matrix.shape[1]} 电影")
        
        return rating_matrix, user_id_to_idx, movie_id_to_idx
    
    def compute_user_similarity(self, rating_matrix: pd.DataFrame, user_id: int, 
                                 min_common_items: int = 3) -> List[Tuple[int, float]]:
        """
        计算用户相似度（使用余弦相似度）
        返回: [(相似用户ID, 相似度分数), ...]，按相似度降序排列
        """
        if user_id not in rating_matrix.index:
            logger.warning(f"用户 {user_id} 不在评分矩阵中")
            return []
        
        user_ratings = rating_matrix.loc[user_id].values
        similarities = []
        
        for other_user_id in rating_matrix.index:
            if other_user_id == user_id:
                continue
            
            other_ratings = rating_matrix.loc[other_user_id].values
            
            # 找到两个用户都评分过的电影
            common_mask = (user_ratings != 0) & (other_ratings != 0)
            
            if np.sum(common_mask) < min_common_items:
                continue
            
            # 计算余弦相似度
            user_common = user_ratings[common_mask]
            other_common = other_ratings[common_mask]
            
            similarity = self._cosine_similarity(user_common, other_common)
            
            if similarity > 0:
                similarities.append((other_user_id, similarity))
        
        # 按相似度降序排列
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def compute_movie_similarity(self, rating_matrix: pd.DataFrame, movie_id: int,
                                  min_common_users: int = 3) -> List[Tuple[int, float]]:
        """
        计算电影相似度（使用余弦相似度）
        返回: [(相似电影ID, 相似度分数), ...]，按相似度降序排列
        """
        if movie_id not in rating_matrix.columns:
            logger.warning(f"电影 {movie_id} 不在评分矩阵中")
            return []
        
        movie_ratings = rating_matrix[movie_id].values
        similarities = []
        
        for other_movie_id in rating_matrix.columns:
            if other_movie_id == movie_id:
                continue
            
            other_ratings = rating_matrix[other_movie_id].values
            
            # 找到两部电影都被评分过的用户
            common_mask = (movie_ratings != 0) & (other_ratings != 0)
            
            if np.sum(common_mask) < min_common_users:
                continue
            
            # 计算余弦相似度
            movie_common = movie_ratings[common_mask]
            other_common = other_ratings[common_mask]
            
            similarity = self._cosine_similarity(movie_common, other_common)
            
            if similarity > 0:
                similarities.append((other_movie_id, similarity))
        
        # 按相似度降序排列
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def update_all_features(self):
        """更新所有用户和电影的特征"""
        logger.info("开始更新所有特征...")
        
        # 更新用户特征
        user_ids = self.data_storage.get_all_user_ids()
        for user_id in user_ids:
            self.extract_user_features(user_id)
        
        # 更新电影特征
        movie_ids = self.data_storage.get_all_movie_ids()
        for movie_id in movie_ids:
            self.extract_movie_features(movie_id)
        
        logger.info(f"特征更新完成: {len(user_ids)} 用户, {len(movie_ids)} 电影")


