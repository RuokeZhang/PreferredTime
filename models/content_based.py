import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ContentBasedRecommender:
    """基于内容的推荐模型"""
    
    def __init__(self, rating_matrix: pd.DataFrame, top_n_similar_movies: int = 50):
        """
        初始化基于内容的推荐模型
        
        Args:
            rating_matrix: 用户-电影评分矩阵
            top_n_similar_movies: 为每部电影考虑的相似电影数量
        """
        self.rating_matrix = rating_matrix
        self.top_n_similar_movies = top_n_similar_movies
        
        # 预计算电影相似度
        self.movie_similarity_matrix = self._compute_movie_similarity_matrix()
    
    def _compute_movie_similarity_matrix(self) -> pd.DataFrame:
        """
        计算电影相似度矩阵（基于用户评分模式）
        使用余弦相似度
        """
        logger.info("计算电影相似度矩阵...")
        
        # 转置矩阵，行为电影，列为用户
        movie_matrix = self.rating_matrix.T
        
        # 计算余弦相似度
        # 归一化每个电影向量
        movie_norms = np.linalg.norm(movie_matrix.values, axis=1, keepdims=True)
        movie_norms[movie_norms == 0] = 1  # 避免除以零
        
        normalized_matrix = movie_matrix.values / movie_norms
        
        # 计算相似度矩阵
        similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)
        
        # 创建DataFrame
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=movie_matrix.index,
            columns=movie_matrix.index
        )
        
        logger.info(f"电影相似度矩阵计算完成: {similarity_df.shape}")
        
        return similarity_df
    
    def recommend(self, user_id: int, top_n: int = 20,
                  min_rating_threshold: float = 3.0,
                  exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        为用户生成基于内容的推荐
        
        Args:
            user_id: 目标用户ID
            top_n: 返回的推荐数量
            min_rating_threshold: 仅考虑高于此评分的电影
            exclude_rated: 是否排除用户已评分的电影
        
        Returns:
            [(movie_id, predicted_score), ...]，按预测分数降序排列
        """
        if user_id not in self.rating_matrix.index:
            logger.warning(f"用户 {user_id} 不在评分矩阵中")
            return []
        
        # 获取用户的评分
        user_ratings = self.rating_matrix.loc[user_id]
        
        # 获取用户喜欢的电影（高评分电影）
        liked_movies = user_ratings[user_ratings >= min_rating_threshold]
        
        if len(liked_movies) == 0:
            logger.warning(f"用户 {user_id} 没有高评分电影")
            return []
        
        # 获取用户已评分的电影
        rated_movies = set(user_ratings[user_ratings > 0].index)
        
        # 为每部候选电影计算分数
        candidate_scores = {}
        
        for movie_id in self.rating_matrix.columns:
            # 如果需要排除已评分的电影
            if exclude_rated and movie_id in rated_movies:
                continue
            
            if movie_id not in self.movie_similarity_matrix.columns:
                continue
            
            # 基于用户喜欢的电影，计算候选电影的分数
            score = 0.0
            weight_sum = 0.0
            
            for liked_movie_id, rating in liked_movies.items():
                if liked_movie_id in self.movie_similarity_matrix.index:
                    similarity = self.movie_similarity_matrix.loc[liked_movie_id, movie_id]
                    
                    if similarity > 0:
                        # 分数 = 相似度 * 用户评分
                        score += similarity * rating
                        weight_sum += similarity
            
            if weight_sum > 0:
                # 归一化分数
                normalized_score = score / weight_sum
                candidate_scores[movie_id] = normalized_score
        
        # 按分数降序排列
        sorted_recommendations = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_recommendations[:top_n]
    
    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        获取与指定电影最相似的电影
        
        Args:
            movie_id: 目标电影ID
            top_n: 返回的相似电影数量
        
        Returns:
            [(similar_movie_id, similarity_score), ...]，按相似度降序排列
        """
        if movie_id not in self.movie_similarity_matrix.index:
            logger.warning(f"电影 {movie_id} 不在相似度矩阵中")
            return []
        
        # 获取相似度分数
        similarities = self.movie_similarity_matrix.loc[movie_id]
        
        # 排除自己
        similarities = similarities[similarities.index != movie_id]
        
        # 按相似度降序排列
        sorted_similarities = similarities.sort_values(ascending=False)
        
        # 返回前N个
        top_similar = [(int(mid), float(score)) 
                      for mid, score in sorted_similarities.head(top_n).items()]
        
        return top_similar
    
    def get_user_profile(self, user_id: int) -> Dict:
        """
        构建用户画像（基于用户的评分历史）
        
        Args:
            user_id: 用户ID
        
        Returns:
            用户画像字典，包含偏好电影列表和平均评分等信息
        """
        if user_id not in self.rating_matrix.index:
            logger.warning(f"用户 {user_id} 不在评分矩阵中")
            return {}
        
        user_ratings = self.rating_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        
        if len(rated_movies) == 0:
            return {
                'rated_movies': [],
                'avg_rating': 0.0,
                'high_rated_movies': [],
                'low_rated_movies': []
            }
        
        # 分类高分和低分电影
        high_rated = rated_movies[rated_movies >= 4.0]
        low_rated = rated_movies[rated_movies < 3.0]
        
        profile = {
            'rated_movies': list(rated_movies.index),
            'avg_rating': float(rated_movies.mean()),
            'high_rated_movies': list(high_rated.index),
            'low_rated_movies': list(low_rated.index),
            'rating_count': len(rated_movies)
        }
        
        return profile


