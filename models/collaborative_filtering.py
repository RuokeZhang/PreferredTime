import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


class CollaborativeFiltering:
    """协同过滤推荐模型"""
    
    def __init__(self, rating_matrix: pd.DataFrame, n_neighbors: int = 20,
                 min_common_items: int = 3, similarity_threshold: float = 0.1):
        """
        初始化协同过滤模型
        
        Args:
            rating_matrix: 用户-电影评分矩阵
            n_neighbors: 考虑的相似用户/电影数量
            min_common_items: 计算相似度所需的最少共同项
            similarity_threshold: 相似度阈值
        """
        self.rating_matrix = rating_matrix
        self.n_neighbors = n_neighbors
        self.min_common_items = min_common_items
        self.similarity_threshold = similarity_threshold
        
        # 缓存相似度矩阵
        self.user_similarity_cache = {}
        self.item_similarity_cache = {}
    
    def user_based_recommend(self, user_id: int, top_n: int = 20,
                             exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        基于用户的协同过滤推荐
        
        Args:
            user_id: 目标用户ID
            top_n: 返回的推荐数量
            exclude_rated: 是否排除用户已评分的电影
        
        Returns:
            [(movie_id, predicted_score), ...]，按预测分数降序排列
        """
        if user_id not in self.rating_matrix.index:
            logger.warning(f"用户 {user_id} 不在评分矩阵中")
            return []
        
        # 获取相似用户
        similar_users = self._get_similar_users(user_id)
        
        if not similar_users:
            logger.warning(f"用户 {user_id} 没有找到相似用户")
            return []
        
        # 获取用户已评分的电影
        user_ratings = self.rating_matrix.loc[user_id]
        rated_movies = set(user_ratings[user_ratings > 0].index)
        
        # 计算预测评分
        predictions = {}
        
        for movie_id in self.rating_matrix.columns:
            # 如果需要排除已评分的电影
            if exclude_rated and movie_id in rated_movies:
                continue
            
            # 计算加权平均评分
            weighted_sum = 0.0
            similarity_sum = 0.0
            
            for similar_user_id, similarity in similar_users[:self.n_neighbors]:
                rating = self.rating_matrix.loc[similar_user_id, movie_id]
                
                if rating > 0:  # 相似用户评分过这部电影
                    weighted_sum += similarity * rating
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                predicted_score = weighted_sum / similarity_sum
                predictions[movie_id] = predicted_score
        
        # 按预测分数降序排列
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_predictions[:top_n]
    
    def item_based_recommend(self, user_id: int, top_n: int = 20,
                            exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        基于物品的协同过滤推荐
        
        Args:
            user_id: 目标用户ID
            top_n: 返回的推荐数量
            exclude_rated: 是否排除用户已评分的电影
        
        Returns:
            [(movie_id, predicted_score), ...]，按预测分数降序排列
        """
        if user_id not in self.rating_matrix.index:
            logger.warning(f"用户 {user_id} 不在评分矩阵中")
            return []
        
        # 获取用户已评分的电影
        user_ratings = self.rating_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        
        if len(rated_movies) == 0:
            logger.warning(f"用户 {user_id} 没有评分记录")
            return []
        
        # 计算预测评分
        predictions = {}
        
        for movie_id in self.rating_matrix.columns:
            # 如果需要排除已评分的电影
            if exclude_rated and movie_id in rated_movies.index:
                continue
            
            # 获取相似电影
            similar_movies = self._get_similar_movies(movie_id)
            
            if not similar_movies:
                continue
            
            # 计算加权平均评分
            weighted_sum = 0.0
            similarity_sum = 0.0
            
            for similar_movie_id, similarity in similar_movies[:self.n_neighbors]:
                if similar_movie_id in rated_movies.index:
                    rating = rated_movies[similar_movie_id]
                    weighted_sum += similarity * rating
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                predicted_score = weighted_sum / similarity_sum
                predictions[movie_id] = predicted_score
        
        # 按预测分数降序排列
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_predictions[:top_n]
    
    def _get_similar_users(self, user_id: int) -> List[Tuple[int, float]]:
        """获取相似用户列表（带缓存）"""
        if user_id in self.user_similarity_cache:
            return self.user_similarity_cache[user_id]
        
        user_ratings = self.rating_matrix.loc[user_id].values
        similarities = []
        
        for other_user_id in self.rating_matrix.index:
            if other_user_id == user_id:
                continue
            
            other_ratings = self.rating_matrix.loc[other_user_id].values
            
            # 找到两个用户都评分过的电影
            common_mask = (user_ratings != 0) & (other_ratings != 0)
            
            if np.sum(common_mask) < self.min_common_items:
                continue
            
            # 计算余弦相似度
            user_common = user_ratings[common_mask]
            other_common = other_ratings[common_mask]
            
            similarity = self._cosine_similarity(user_common, other_common)
            
            if similarity > self.similarity_threshold:
                similarities.append((other_user_id, similarity))
        
        # 按相似度降序排列
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 缓存结果
        self.user_similarity_cache[user_id] = similarities
        
        return similarities
    
    def _get_similar_movies(self, movie_id: int) -> List[Tuple[int, float]]:
        """获取相似电影列表（带缓存）"""
        if movie_id in self.item_similarity_cache:
            return self.item_similarity_cache[movie_id]
        
        movie_ratings = self.rating_matrix[movie_id].values
        similarities = []
        
        for other_movie_id in self.rating_matrix.columns:
            if other_movie_id == movie_id:
                continue
            
            other_ratings = self.rating_matrix[other_movie_id].values
            
            # 找到两部电影都被评分过的用户
            common_mask = (movie_ratings != 0) & (other_ratings != 0)
            
            if np.sum(common_mask) < self.min_common_items:
                continue
            
            # 计算余弦相似度
            movie_common = movie_ratings[common_mask]
            other_common = other_ratings[common_mask]
            
            similarity = self._cosine_similarity(movie_common, other_common)
            
            if similarity > self.similarity_threshold:
                similarities.append((other_movie_id, similarity))
        
        # 按相似度降序排列
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 缓存结果
        self.item_similarity_cache[movie_id] = similarities
        
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

