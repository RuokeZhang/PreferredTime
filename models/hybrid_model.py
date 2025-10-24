import pandas as pd
from typing import List, Tuple, Dict
from models.collaborative_filtering import CollaborativeFiltering
from models.content_based import ContentBasedRecommender
from utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridRecommender:
    """混合推荐模型，结合协同过滤和基于内容的推荐"""
    
    def __init__(self, rating_matrix: pd.DataFrame, config: Dict):
        """
        初始化混合推荐模型
        
        Args:
            rating_matrix: 用户-电影评分矩阵
            config: 配置字典，包含各模型的参数
        """
        self.rating_matrix = rating_matrix
        self.config = config
        
        # 初始化协同过滤模型
        cf_config = config.get('collaborative_filtering', {})
        self.cf_model = CollaborativeFiltering(
            rating_matrix=rating_matrix,
            n_neighbors=cf_config.get('n_neighbors', 20),
            min_common_items=cf_config.get('min_common_items', 3),
            similarity_threshold=cf_config.get('similarity_threshold', 0.1)
        )
        
        # 初始化基于内容的推荐模型
        content_config = config.get('content_based', {})
        self.content_model = ContentBasedRecommender(
            rating_matrix=rating_matrix,
            top_n_similar_movies=content_config.get('top_n_similar_movies', 50)
        )
        
        # 混合权重
        hybrid_config = config.get('hybrid', {})
        self.cf_weight = hybrid_config.get('cf_weight', 0.6)
        self.content_weight = hybrid_config.get('content_weight', 0.4)
        
        # 推荐参数
        rec_config = config.get('recommendation', {})
        self.top_n = rec_config.get('top_n', 20)
        self.min_rating_threshold = rec_config.get('min_rating_threshold', 3.0)
        
        logger.info(f"混合推荐模型初始化完成 - CF权重: {self.cf_weight}, 内容权重: {self.content_weight}")
    
    def recommend(self, user_id: int, top_n: int = None,
                  use_user_based: bool = True,
                  use_item_based: bool = True) -> List[int]:
        """
        为用户生成混合推荐
        
        Args:
            user_id: 目标用户ID
            top_n: 返回的推荐数量（如果为None，使用配置中的默认值）
            use_user_based: 是否使用基于用户的协同过滤
            use_item_based: 是否使用基于物品的协同过滤
        
        Returns:
            推荐的电影ID列表，按推荐优先级排序
        """
        if top_n is None:
            top_n = self.top_n
        
        if user_id not in self.rating_matrix.index:
            logger.warning(f"用户 {user_id} 不在评分矩阵中，返回热门电影")
            return self._get_popular_movies(top_n)
        
        # 收集不同方法的推荐结果
        all_recommendations = {}
        
        # 1. 基于用户的协同过滤
        if use_user_based:
            try:
                user_based_recs = self.cf_model.user_based_recommend(
                    user_id, top_n=top_n * 2
                )
                self._merge_recommendations(
                    all_recommendations,
                    user_based_recs,
                    weight=self.cf_weight * 0.5
                )
                logger.debug(f"用户基于CF推荐: {len(user_based_recs)} 部电影")
            except Exception as e:
                logger.error(f"用户基于CF推荐失败: {e}")
        
        # 2. 基于物品的协同过滤
        if use_item_based:
            try:
                item_based_recs = self.cf_model.item_based_recommend(
                    user_id, top_n=top_n * 2
                )
                self._merge_recommendations(
                    all_recommendations,
                    item_based_recs,
                    weight=self.cf_weight * 0.5
                )
                logger.debug(f"物品基于CF推荐: {len(item_based_recs)} 部电影")
            except Exception as e:
                logger.error(f"物品基于CF推荐失败: {e}")
        
        # 3. 基于内容的推荐
        try:
            content_recs = self.content_model.recommend(
                user_id,
                top_n=top_n * 2,
                min_rating_threshold=self.min_rating_threshold
            )
            self._merge_recommendations(
                all_recommendations,
                content_recs,
                weight=self.content_weight
            )
            logger.debug(f"基于内容推荐: {len(content_recs)} 部电影")
        except Exception as e:
            logger.error(f"基于内容推荐失败: {e}")
        
        # 如果没有推荐结果，返回热门电影
        if not all_recommendations:
            logger.warning(f"用户 {user_id} 无推荐结果，返回热门电影")
            return self._get_popular_movies(top_n)
        
        # 按综合分数排序
        sorted_recommendations = sorted(
            all_recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 返回前N个电影ID
        recommended_movie_ids = [int(movie_id) for movie_id, _ in sorted_recommendations[:top_n]]
        
        logger.info(f"为用户 {user_id} 生成 {len(recommended_movie_ids)} 个推荐")
        
        return recommended_movie_ids
    
    def _merge_recommendations(self, all_recs: Dict[int, float],
                              new_recs: List[Tuple[int, float]],
                              weight: float):
        """
        合并推荐结果，使用加权平均
        
        Args:
            all_recs: 现有的推荐字典 {movie_id: score}
            new_recs: 新的推荐列表 [(movie_id, score), ...]
            weight: 新推荐的权重
        """
        for movie_id, score in new_recs:
            if movie_id in all_recs:
                # 如果电影已经在推荐中，累加分数
                all_recs[movie_id] += score * weight
            else:
                all_recs[movie_id] = score * weight
    
    def _get_popular_movies(self, top_n: int) -> List[int]:
        """
        获取热门电影（基于评分数量和平均评分）
        
        Args:
            top_n: 返回的电影数量
        
        Returns:
            热门电影ID列表
        """
        # 计算每部电影的评分数量和平均评分
        movie_stats = []
        
        for movie_id in self.rating_matrix.columns:
            ratings = self.rating_matrix[movie_id]
            rated_by = ratings[ratings > 0]
            
            if len(rated_by) > 0:
                avg_rating = rated_by.mean()
                rating_count = len(rated_by)
                
                # 使用贝叶斯平均来平衡评分和评分数量
                # score = (avg_rating * rating_count + global_avg * min_count) / (rating_count + min_count)
                global_avg = 3.0
                min_count = 10
                bayesian_avg = (avg_rating * rating_count + global_avg * min_count) / (rating_count + min_count)
                
                movie_stats.append((movie_id, bayesian_avg, rating_count))
        
        # 按贝叶斯平均分数排序
        movie_stats.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前N个电影ID
        popular_movies = [int(movie_id) for movie_id, _, _ in movie_stats[:top_n]]
        
        logger.info(f"返回 {len(popular_movies)} 部热门电影")
        
        return popular_movies
    
    def get_recommendation_explanation(self, user_id: int, movie_id: int) -> Dict:
        """
        获取推荐解释（为什么推荐这部电影）
        
        Args:
            user_id: 用户ID
            movie_id: 电影ID
        
        Returns:
            推荐解释字典
        """
        explanation = {
            'user_id': user_id,
            'movie_id': movie_id,
            'reasons': []
        }
        
        if user_id not in self.rating_matrix.index:
            explanation['reasons'].append("用户数据不足，基于热门推荐")
            return explanation
        
        # 检查相似用户
        similar_users = self.cf_model._get_similar_users(user_id)
        if similar_users:
            top_similar_user = similar_users[0]
            explanation['reasons'].append(
                f"与您相似的用户 {top_similar_user[0]} 也喜欢这部电影"
            )
        
        # 检查相似电影
        user_ratings = self.rating_matrix.loc[user_id]
        liked_movies = user_ratings[user_ratings >= 4.0].index.tolist()
        
        if liked_movies and movie_id in self.content_model.movie_similarity_matrix.columns:
            for liked_movie in liked_movies[:3]:
                if liked_movie in self.content_model.movie_similarity_matrix.index:
                    similarity = self.content_model.movie_similarity_matrix.loc[liked_movie, movie_id]
                    if similarity > 0.3:
                        explanation['reasons'].append(
                            f"与您喜欢的电影 {liked_movie} 相似（相似度: {similarity:.2f}）"
                        )
        
        return explanation

