import pandas as pd
from typing import List, Tuple, Dict, Optional
from models.collaborative_filtering import CollaborativeFiltering
from models.content_based import ContentBasedRecommender
from utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridRecommender:
    """
    混合推荐模型，结合协同过滤和基于内容的推荐
    
    企业级特性：
    - 支持 Feature Store (DynamoDB) 实时特征
    - 用户冷启动检测与降级策略
    - 精排阶段融合电影质量分 (popularity + avg_rating)
    - 用户评分倾向校正
    """
    
    def __init__(self, rating_matrix: pd.DataFrame, config: Dict, 
                 feature_store=None):
        """
        初始化混合推荐模型
        
        Args:
            rating_matrix: 用户-电影评分矩阵
            config: 配置字典，包含各模型的参数
            feature_store: 特征存储层 (HybridStorage)，用于读取 DynamoDB 实时特征
        """
        self.rating_matrix = rating_matrix
        self.config = config
        self.feature_store = feature_store  # DynamoDB/S3 特征存储
        
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
        self.quality_weight = hybrid_config.get('quality_weight', 0.2)  # 质量分权重
        
        # 推荐参数
        rec_config = config.get('recommendation', {})
        self.top_n = rec_config.get('top_n', 20)
        self.min_rating_threshold = rec_config.get('min_rating_threshold', 3.0)
        self.cold_start_threshold = rec_config.get('cold_start_threshold', 5)  # 冷启动阈值
        
        # 缓存电影特征（批量读取后缓存，避免重复查询）
        self._movie_features_cache: Dict[int, Dict] = {}
        
        logger.info(f"混合推荐模型初始化完成 - CF权重: {self.cf_weight}, 内容权重: {self.content_weight}, 质量分权重: {self.quality_weight}")
        if feature_store:
            logger.info("✓ Feature Store (DynamoDB) 已接入，支持实时特征")
    
    def recommend(self, user_id: int, top_n: int = None,
                  use_user_based: bool = True,
                  use_item_based: bool = True) -> List[int]:
        """
        为用户生成混合推荐（企业级流程）
        
        流程：
        1. 从 Feature Store 读取用户特征，判断冷启动
        2. 多路召回：User-CF + Item-CF + Content-Based
        3. 精排：融合电影质量分（从 DynamoDB 读取）
        4. 用户评分倾向校正
        
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
        
        # ========== Step 1: 用户特征 & 冷启动检测 ==========
        user_feature = self._get_user_feature(user_id)
        is_cold_start = self._is_cold_start_user(user_id, user_feature)
        
        if is_cold_start:
            logger.info(f"用户 {user_id} 为冷启动用户，使用热门推荐 + 探索策略")
            return self._get_popular_movies_with_exploration(top_n)
        
        if user_id not in self.rating_matrix.index:
            logger.warning(f"用户 {user_id} 不在评分矩阵中，返回热门电影")
            return self._get_popular_movies(top_n)
        
        # ========== Step 2: 多路召回 ==========
        all_recommendations = {}
        
        # 2.1 基于用户的协同过滤
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
                logger.debug(f"User-CF 召回: {len(user_based_recs)} 部电影")
            except Exception as e:
                logger.error(f"User-CF 召回失败: {e}")
        
        # 2.2 基于物品的协同过滤
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
                logger.debug(f"Item-CF 召回: {len(item_based_recs)} 部电影")
            except Exception as e:
                logger.error(f"Item-CF 召回失败: {e}")
        
        # 2.3 基于内容的推荐
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
            logger.debug(f"Content-Based 召回: {len(content_recs)} 部电影")
        except Exception as e:
            logger.error(f"Content-Based 召回失败: {e}")
        
        # 如果没有召回结果，返回热门电影
        if not all_recommendations:
            logger.warning(f"用户 {user_id} 无召回结果，返回热门电影")
            return self._get_popular_movies(top_n)
        
        # ========== Step 3: 精排 - 融合电影质量分 ==========
        ranked_recommendations = self._rerank_with_quality_score(
            all_recommendations, user_feature
        )
        
        # ========== Step 4: 返回 Top-N ==========
        recommended_movie_ids = [int(movie_id) for movie_id, _ in ranked_recommendations[:top_n]]
        
        logger.info(f"为用户 {user_id} 生成 {len(recommended_movie_ids)} 个推荐 (冷启动={is_cold_start})")
        
        return recommended_movie_ids
    
    def _get_user_feature(self, user_id: int) -> Optional[Dict]:
        """
        从 Feature Store (DynamoDB) 获取用户特征
        
        Returns:
            用户特征字典: {avg_rating, rating_count, std_dev} 或 None
        """
        if self.feature_store is None:
            return None
        
        try:
            return self.feature_store.get_user_feature(user_id)
        except Exception as e:
            logger.warning(f"从 Feature Store 获取用户特征失败: {e}")
            return None
    
    def _get_movie_feature(self, movie_id: int) -> Optional[Dict]:
        """
        从 Feature Store (DynamoDB) 获取电影特征（带缓存）
        
        Returns:
            电影特征字典: {avg_rating, rating_count, popularity} 或 None
        """
        # 先查缓存
        if movie_id in self._movie_features_cache:
            return self._movie_features_cache[movie_id]
        
        if self.feature_store is None:
            return None
        
        try:
            feature = self.feature_store.get_movie_feature(movie_id)
            if feature:
                self._movie_features_cache[movie_id] = feature
            return feature
        except Exception as e:
            logger.debug(f"获取电影 {movie_id} 特征失败: {e}")
            return None
    
    def _is_cold_start_user(self, user_id: int, user_feature: Optional[Dict]) -> bool:
        """
        判断用户是否为冷启动用户
        
        策略：
        1. 如果有 Feature Store 特征，用 rating_count 判断
        2. 否则检查评分矩阵
        """
        # 优先使用 DynamoDB 特征
        if user_feature and 'rating_count' in user_feature:
            return user_feature['rating_count'] < self.cold_start_threshold
        
        # 回退到评分矩阵
        if user_id in self.rating_matrix.index:
            user_ratings = self.rating_matrix.loc[user_id]
            rating_count = (user_ratings > 0).sum()
            return rating_count < self.cold_start_threshold
        
        return True  # 完全没有数据，视为冷启动
    
    def _rerank_with_quality_score(self, candidates: Dict[int, float],
                                   user_feature: Optional[Dict]) -> List[Tuple[int, float]]:
        """
        精排：融合电影质量分 + 用户活跃度自适应权重
        
        质量分 = popularity * 0.4 + avg_rating * 0.6（归一化后）
        
        用户活跃度策略：
        - 新用户（rating_count < 10）：更依赖 quality_score（热门高分电影）
        - 中等用户（10-50）：平衡 recall 和 quality
        - 活跃用户（rating_count > 50）：更依赖 recall_score（个性化推荐）
        
        最终分 = 召回分 * (1 - effective_quality_weight) + 质量分 * effective_quality_weight
        """
        reranked = []
        
        # ========== 根据用户活跃度动态调整 quality_weight ==========
        user_rating_count = 0
        if user_feature and 'rating_count' in user_feature:
            user_rating_count = user_feature['rating_count']
        
        # 活跃度越高，quality_weight 越低（更信任个性化召回）
        # rating_count: 0-10 → quality_weight: 0.5
        # rating_count: 10-50 → quality_weight: 0.3
        # rating_count: 50+ → quality_weight: 0.1
        if user_rating_count < 10:
            effective_quality_weight = 0.5  # 新用户：质量分占 50%
        elif user_rating_count < 50:
            # 线性插值：10→0.5, 50→0.1
            effective_quality_weight = 0.5 - (user_rating_count - 10) * (0.4 / 40)
        else:
            effective_quality_weight = 0.1  # 活跃用户：质量分只占 10%
        
        logger.debug(f"用户活跃度调整: rating_count={user_rating_count}, "
                    f"effective_quality_weight={effective_quality_weight:.2f}")
        
        for movie_id, recall_score in candidates.items():
            final_score = recall_score
            
            # 尝试获取电影特征（从 DynamoDB）
            movie_feature = self._get_movie_feature(movie_id)
            
            if movie_feature and effective_quality_weight > 0:
                # 计算质量分（归一化到 0-1）
                popularity = movie_feature.get('popularity', 0)
                avg_rating = movie_feature.get('avg_rating', 3.0)
                
                # popularity 通常是 log1p(rating_count)，假设范围 0-10
                norm_popularity = min(popularity / 10.0, 1.0)
                # avg_rating 范围 1-5，归一化到 0-1
                norm_avg_rating = (avg_rating - 1) / 4.0
                
                quality_score = norm_popularity * 0.4 + norm_avg_rating * 0.6
                
                # 融合召回分和质量分（使用动态权重）
                final_score = (
                    recall_score * (1 - effective_quality_weight) +
                    quality_score * effective_quality_weight * 5  # 放大到同一量级
                )
            
            reranked.append((movie_id, final_score))
        
        # 按最终分数降序排序
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    def _get_popular_movies_with_exploration(self, top_n: int) -> List[int]:
        """
        冷启动用户推荐策略：热门 + 探索
        
        80% 热门电影 + 20% 随机电影（增加多样性）
        """
        import random
        
        popular = self._get_popular_movies(int(top_n * 0.8))
        
        # 随机选择一些电影用于探索
        all_movies = list(self.rating_matrix.columns)
        explore_candidates = [m for m in all_movies if m not in popular]
        
        n_explore = top_n - len(popular)
        if explore_candidates and n_explore > 0:
            explore = random.sample(explore_candidates, min(n_explore, len(explore_candidates)))
        else:
            explore = []
        
        result = popular + explore
        logger.info(f"冷启动推荐: {len(popular)} 热门 + {len(explore)} 探索")
        
        return result[:top_n]
    
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

