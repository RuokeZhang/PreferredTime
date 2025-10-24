"""
模型训练模块
包含数据验证、特征提取、模型训练、评估和部署的完整流程
"""
import os
import sys
import yaml
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor.hybrid_storage import HybridStorage
from data_processor.feature_extractor import FeatureExtractor
from models.hybrid_model import HybridRecommender
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_data(date: str, **context) -> bool:
    """
    验证数据质量
    
    Args:
        date: 日期字符串 (YYYY-MM-DD)
    
    Returns:
        数据是否有效
    """
    logger.info(f"=" * 60)
    logger.info(f"Task 1: 验证数据质量 - {date}")
    logger.info(f"=" * 60)
    
    try:
        config = load_config()
        storage = HybridStorage(config)
        
        if config.get('storage_mode') == 'aws':
            # AWS模式：验证S3数据
            events = storage.s3_storage.read_raw_events_by_date(date)
            
            if len(events) == 0:
                logger.warning(f"日期 {date} 没有数据")
                return False
            
            # 验证数据格式
            for event in events[:100]:  # 抽样验证
                assert 'user_id' in event
                assert 'movie_id' in event
                assert 'rating' in event
                assert 1.0 <= event['rating'] <= 5.0
            
            logger.info(f"✓ 数据验证通过: {len(events)} 条记录")
        else:
            # SQLite模式：验证数据库
            ratings = storage.sqlite_storage.get_all_ratings()
            logger.info(f"✓ 数据验证通过: {len(ratings)} 条记录")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 数据验证失败: {e}")
        raise


def extract_features_batch(date: str, **context) -> Dict:
    """
    批量提取特征
    
    Args:
        date: 日期字符串
    
    Returns:
        特征统计信息
    """
    logger.info(f"=" * 60)
    logger.info(f"Task 2: 批量提取特征 - {date}")
    logger.info(f"=" * 60)
    
    try:
        config = load_config()
        storage = HybridStorage(config)
        
        # 读取所有评分数据
        all_ratings = storage.get_all_ratings()
        logger.info(f"读取评分数据: {len(all_ratings)} 条")
        
        # 构建评分DataFrame
        df = pd.DataFrame(all_ratings, columns=['user_id', 'movie_id', 'rating', 'timestamp'])
        
        # 计算用户特征
        logger.info("计算用户特征...")
        user_features = df.groupby('user_id').agg({
            'rating': ['mean', 'count', 'std']
        }).reset_index()
        user_features.columns = ['user_id', 'avg_rating', 'rating_count', 'std_dev']
        user_features['std_dev'] = user_features['std_dev'].fillna(0)
        
        # 计算电影特征
        logger.info("计算电影特征...")
        movie_features = df.groupby('movie_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_features.columns = ['movie_id', 'avg_rating', 'rating_count']
        movie_features['popularity'] = np.log1p(movie_features['rating_count'])
        
        # 更新到存储层
        logger.info("更新特征到存储层...")
        for _, row in user_features.iterrows():
            storage.update_user_feature(
                int(row['user_id']),
                float(row['avg_rating']),
                int(row['rating_count']),
                float(row['std_dev'])
            )
        
        for _, row in movie_features.iterrows():
            storage.update_movie_feature(
                int(row['movie_id']),
                float(row['avg_rating']),
                int(row['rating_count']),
                float(row['popularity'])
            )
        
        stats = {
            'user_count': len(user_features),
            'movie_count': len(movie_features),
            'rating_count': len(all_ratings),
            'date': date
        }
        
        logger.info(f"✓ 特征提取完成: {stats}")
        
        # 推送到XCom供下游任务使用
        context['task_instance'].xcom_push(key='feature_stats', value=stats)
        
        return stats
        
    except Exception as e:
        logger.error(f"✗ 特征提取失败: {e}")
        raise


def train_hybrid_model(**context) -> str:
    """
    训练混合推荐模型
    
    Returns:
        模型保存路径
    """
    logger.info(f"=" * 60)
    logger.info(f"Task 3: 训练混合推荐模型")
    logger.info(f"=" * 60)
    
    try:
        config = load_config()
        storage = HybridStorage(config)
        
        # 构建评分矩阵
        logger.info("构建用户-电影评分矩阵...")
        if config.get('storage_mode') == 'sqlite':
            feature_extractor = FeatureExtractor(storage.sqlite_storage)
        else:
            feature_extractor = FeatureExtractor(storage)
        
        rating_matrix, user_id_to_idx, movie_id_to_idx = feature_extractor.build_user_item_matrix()
        
        logger.info(f"评分矩阵形状: {rating_matrix.shape}")
        logger.info(f"用户数: {len(user_id_to_idx)}")
        logger.info(f"电影数: {len(movie_id_to_idx)}")
        
        # 训练模型
        logger.info("训练混合推荐模型...")
        model = HybridRecommender(rating_matrix, config['model'])
        
        # 保存模型
        model_dir = 'models/saved_models'
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'{model_dir}/hybrid_model_{timestamp}.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'rating_matrix': rating_matrix,
                'user_id_to_idx': user_id_to_idx,
                'movie_id_to_idx': movie_id_to_idx,
                'config': config['model'],
                'timestamp': timestamp
            }, f)
        
        logger.info(f"✓ 模型已保存: {model_path}")
        
        # 推送模型路径到XCom
        context['task_instance'].xcom_push(key='model_path', value=model_path)
        
        return model_path
        
    except Exception as e:
        logger.error(f"✗ 模型训练失败: {e}")
        raise


def split_train_test(rating_matrix: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, Dict]:
    """
    将每个用户的评分数据划分为训练集和测试集
    
    Args:
        rating_matrix: 用户-电影评分矩阵
        test_ratio: 测试集比例
    
    Returns:
        train_matrix: 训练集评分矩阵
        test_data: 测试集字典 {user_id: [(movie_id, rating), ...]}
    """
    train_matrix = rating_matrix.copy()
    test_data = {}
    
    for user_id in rating_matrix.index:
        user_ratings = rating_matrix.loc[user_id]
        # 获取用户评过分的电影
        rated_movies = user_ratings[user_ratings > 0]
        
        if len(rated_movies) < 5:  # 评分太少的用户不划分
            continue
        
        # 随机选择测试集
        n_test = max(1, int(len(rated_movies) * test_ratio))
        test_movies = rated_movies.sample(n=n_test, random_state=42)
        
        # 保存测试数据
        test_data[user_id] = [(movie_id, rating) for movie_id, rating in test_movies.items()]
        
        # 从训练集中移除测试数据
        train_matrix.loc[user_id, test_movies.index] = 0
    
    return train_matrix, test_data


def calculate_precision_recall_f1(recommended: List[int], relevant: List[int], k: int) -> Tuple[float, float, float]:
    """
    计算 Precision@K, Recall@K, F1@K
    
    Args:
        recommended: 推荐的电影ID列表
        relevant: 相关（用户喜欢）的电影ID列表
        k: Top-K
    
    Returns:
        precision, recall, f1
    """
    if not recommended or not relevant:
        return 0.0, 0.0, 0.0
    
    recommended_k = set(recommended[:k])
    relevant_set = set(relevant)
    
    # 推荐列表中相关的物品数量
    hits = len(recommended_k & relevant_set)
    
    precision = hits / len(recommended_k) if recommended_k else 0.0
    recall = hits / len(relevant_set) if relevant_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def calculate_ndcg(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    计算 NDCG@K (Normalized Discounted Cumulative Gain)
    
    Args:
        recommended: 推荐的电影ID列表
        relevant: 相关（用户喜欢）的电影ID列表
        k: Top-K
    
    Returns:
        NDCG@K 分数
    """
    if not recommended or not relevant:
        return 0.0
    
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    # DCG: 累积折扣增益
    dcg = 0.0
    for i, movie_id in enumerate(recommended_k):
        if movie_id in relevant_set:
            # 相关性为1，位置i+1
            dcg += 1.0 / np.log2(i + 2)  # i+2 因为位置从1开始
    
    # IDCG: 理想情况下的DCG（所有相关物品都在前面）
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg


def calculate_hit_rate(recommended: List[int], relevant: List[int], k: int) -> float:
    """
    计算 Hit Rate@K (命中率)
    
    Args:
        recommended: 推荐的电影ID列表
        relevant: 相关（用户喜欢）的电影ID列表
        k: Top-K
    
    Returns:
        1 如果命中，0 如果未命中
    """
    if not recommended or not relevant:
        return 0.0
    
    recommended_k = set(recommended[:k])
    relevant_set = set(relevant)
    
    # 只要推荐列表中有至少一个相关物品就算命中
    return 1.0 if len(recommended_k & relevant_set) > 0 else 0.0


def calculate_diversity(all_recommendations: List[List[int]]) -> float:
    """
    计算推荐多样性（不同推荐列表之间的差异度）
    
    Args:
        all_recommendations: 所有用户的推荐列表
    
    Returns:
        多样性分数 (0-1)
    """
    if len(all_recommendations) < 2:
        return 0.0
    
    # 计算所有推荐对之间的不相似度
    diversity_scores = []
    for i in range(len(all_recommendations)):
        for j in range(i + 1, len(all_recommendations)):
            set_i = set(all_recommendations[i])
            set_j = set(all_recommendations[j])
            
            # Jaccard距离 = 1 - Jaccard相似度
            union = len(set_i | set_j)
            if union > 0:
                jaccard_similarity = len(set_i & set_j) / union
                diversity_scores.append(1 - jaccard_similarity)
    
    return np.mean(diversity_scores) if diversity_scores else 0.0


def evaluate_model(**context) -> Dict:
    """
    评估模型性能 - 使用推荐系统标准评估指标
    
    Returns:
        评估指标字典
    """
    logger.info(f"=" * 60)
    logger.info(f"Task 4: 评估模型性能（使用标准推荐指标）")
    logger.info(f"=" * 60)
    
    try:
        # 从XCom获取模型路径
        model_path = context['task_instance'].xcom_pull(task_ids='train_model', key='model_path')
        
        # 加载模型
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        rating_matrix = model_data['rating_matrix']
        
        logger.info(f"评分矩阵大小: {rating_matrix.shape}")
        logger.info(f"用户数: {len(rating_matrix)}, 电影数: {len(rating_matrix.columns)}")
        
        # 1. 划分训练集和测试集
        logger.info("划分训练集和测试集...")
        train_matrix, test_data = split_train_test(rating_matrix, test_ratio=0.2)
        logger.info(f"测试用户数: {len(test_data)}")
        
        # 2. 在训练集上重建模型（用于评估）
        logger.info("基于训练集重建模型...")
        config = load_config()
        eval_model = HybridRecommender(train_matrix, config['model'])
        
        # 3. 评估参数
        k_values = [5, 10, 20]  # 评估不同的K值
        rating_threshold = 3.5  # 评分>=3.5认为是相关/喜欢的
        
        # 4. 收集评估数据
        metrics_by_k = {k: {
            'precision': [],
            'recall': [],
            'f1': [],
            'ndcg': [],
            'hit_rate': []
        } for k in k_values}
        
        all_recommendations = []
        all_recommended_movies = set()
        
        # 5. 对每个测试用户进行评估
        logger.info("开始评估...")
        evaluated_users = 0
        
        for user_id, test_items in test_data.items():
            try:
                # 获取推荐列表
                recommended = eval_model.recommend(user_id, top_n=max(k_values))
                
                if not recommended:
                    continue
                
                all_recommendations.append(recommended)
                all_recommended_movies.update(recommended)
                
                # 确定相关物品（测试集中评分>=阈值的电影）
                relevant = [movie_id for movie_id, rating in test_items if rating >= rating_threshold]
                
                if not relevant:
                    continue
                
                # 计算各个K值下的指标
                for k in k_values:
                    precision, recall, f1 = calculate_precision_recall_f1(recommended, relevant, k)
                    ndcg = calculate_ndcg(recommended, relevant, k)
                    hit_rate = calculate_hit_rate(recommended, relevant, k)
                    
                    metrics_by_k[k]['precision'].append(precision)
                    metrics_by_k[k]['recall'].append(recall)
                    metrics_by_k[k]['f1'].append(f1)
                    metrics_by_k[k]['ndcg'].append(ndcg)
                    metrics_by_k[k]['hit_rate'].append(hit_rate)
                
                evaluated_users += 1
                
            except Exception as e:
                logger.debug(f"评估用户 {user_id} 时出错: {e}")
                continue
        
        logger.info(f"成功评估 {evaluated_users} 个用户")
        
        # 6. 计算平均指标
        avg_metrics = {}
        for k in k_values:
            avg_metrics[f'precision@{k}'] = np.mean(metrics_by_k[k]['precision']) if metrics_by_k[k]['precision'] else 0.0
            avg_metrics[f'recall@{k}'] = np.mean(metrics_by_k[k]['recall']) if metrics_by_k[k]['recall'] else 0.0
            avg_metrics[f'f1@{k}'] = np.mean(metrics_by_k[k]['f1']) if metrics_by_k[k]['f1'] else 0.0
            avg_metrics[f'ndcg@{k}'] = np.mean(metrics_by_k[k]['ndcg']) if metrics_by_k[k]['ndcg'] else 0.0
            avg_metrics[f'hit_rate@{k}'] = np.mean(metrics_by_k[k]['hit_rate']) if metrics_by_k[k]['hit_rate'] else 0.0
        
        # 7. 计算覆盖率和多样性
        coverage = len(all_recommended_movies) / len(rating_matrix.columns)
        diversity = calculate_diversity(all_recommendations)
        
        # 8. 汇总所有指标
        metrics = {
            # 准确率指标
            **avg_metrics,
            
            # 覆盖率和多样性
            'coverage': coverage,
            'diversity': diversity,
            
            # 统计信息
            'total_users': len(rating_matrix),
            'total_movies': len(rating_matrix.columns),
            'evaluated_users': evaluated_users,
            'recommended_movies': len(all_recommended_movies),
            
            # 模型信息
            'model_path': model_path,
            'timestamp': model_data['timestamp'],
            'rating_threshold': rating_threshold
        }
        
        # 9. 打印评估结果
        logger.info(f"=" * 60)
        logger.info(f"✓ 评估完成 - 评估了 {evaluated_users} 个用户")
        logger.info(f"=" * 60)
        
        for k in k_values:
            logger.info(f"\n📊 Top-{k} 指标:")
            logger.info(f"  • Precision@{k}: {avg_metrics[f'precision@{k}']:.4f}")
            logger.info(f"  • Recall@{k}: {avg_metrics[f'recall@{k}']:.4f}")
            logger.info(f"  • F1@{k}: {avg_metrics[f'f1@{k}']:.4f}")
            logger.info(f"  • NDCG@{k}: {avg_metrics[f'ndcg@{k}']:.4f}")
            logger.info(f"  • Hit Rate@{k}: {avg_metrics[f'hit_rate@{k}']:.4f}")
        
        logger.info(f"\n📈 系统级指标:")
        logger.info(f"  • 覆盖率 (Coverage): {coverage:.4f} ({len(all_recommended_movies)}/{len(rating_matrix.columns)})")
        logger.info(f"  • 多样性 (Diversity): {diversity:.4f}")
        
        # 推送指标到XCom
        context['task_instance'].xcom_push(key='metrics', value=metrics)
        
        return metrics
        
    except Exception as e:
        logger.error(f"✗ 模型评估失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def deploy_model(**context) -> bool:
    """
    部署模型到生产环境
    
    Returns:
        是否部署成功
    """
    logger.info(f"=" * 60)
    logger.info(f"Task 5: 部署模型到生产环境")
    logger.info(f"=" * 60)
    
    try:
        # 从XCom获取模型路径和评估指标
        model_path = context['task_instance'].xcom_pull(task_ids='train_model', key='model_path')
        metrics = context['task_instance'].xcom_pull(task_ids='evaluate_model', key='metrics')
        
        logger.info(f"待部署模型: {model_path}")
        logger.info(f"\n当前模型评估指标:")
        logger.info(f"  • Precision@10: {metrics.get('precision@10', 0):.4f}")
        logger.info(f"  • Recall@10: {metrics.get('recall@10', 0):.4f}")
        logger.info(f"  • NDCG@10: {metrics.get('ndcg@10', 0):.4f}")
        logger.info(f"  • Hit Rate@10: {metrics.get('hit_rate@10', 0):.4f}")
        logger.info(f"  • Coverage: {metrics.get('coverage', 0):.4f}")
        
        # 定义模型质量阈值
        quality_checks = {
            'precision@10': (metrics.get('precision@10', 0), 0.01, "精确率过低"),
            'hit_rate@10': (metrics.get('hit_rate@10', 0), 0.1, "命中率过低"),
            'coverage': (metrics.get('coverage', 0), 0.05, "覆盖率过低"),
            'evaluated_users': (metrics.get('evaluated_users', 0), 10, "评估用户数太少")
        }
        
        # 检查每个质量指标
        failed_checks = []
        for metric_name, (value, threshold, reason) in quality_checks.items():
            if value < threshold:
                failed_checks.append(f"{metric_name} = {value:.4f} < {threshold} ({reason})")
        
        if failed_checks:
            logger.warning("⚠️  模型质量检查未通过，跳过部署:")
            for fail in failed_checks:
                logger.warning(f"  ✗ {fail}")
            return False
        
        logger.info("✓ 模型质量检查通过")
        
        # 创建生产模型路径
        production_model_path = 'models/saved_models/production_model.pkl'
        
        # 如果存在旧的生产模型，备份
        if os.path.exists(production_model_path):
            backup_path = f'{production_model_path}.backup'
            if os.path.exists(backup_path):
                os.remove(backup_path)  # 删除旧备份
            os.rename(production_model_path, backup_path)
            logger.info(f"备份旧模型: {backup_path}")
        
        # 复制新模型为生产模型
        import shutil
        shutil.copy(model_path, production_model_path)
        
        logger.info(f"=" * 60)
        logger.info(f"✓ 模型部署成功: {production_model_path}")
        logger.info(f"=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 模型部署失败: {e}")
        raise


if __name__ == "__main__":
    """本地测试"""
    print("开始本地测试模型训练流程...")
    
    # 测试数据验证
    print("\n1. 验证数据...")
    validate_data(datetime.now().strftime('%Y-%m-%d'))
    
    # 测试特征提取
    print("\n2. 提取特征...")
    context = {'task_instance': type('obj', (object,), {'xcom_push': lambda *args, **kwargs: None})}
    extract_features_batch(datetime.now().strftime('%Y-%m-%d'), **context)
    
    # 测试模型训练
    print("\n3. 训练模型...")
    model_path = train_hybrid_model(**context)
    
    # 测试模型评估
    print("\n4. 评估模型...")
    context['task_instance'].xcom_pull = lambda *args, **kwargs: model_path if kwargs.get('key') == 'model_path' else {}
    evaluate_model(**context)
    
    # 测试模型部署
    print("\n5. 部署模型...")
    deploy_model(**context)
    
    print("\n✓ 所有测试通过！")

