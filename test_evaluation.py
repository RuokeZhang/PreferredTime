#!/usr/bin/env python3
"""
测试新的评估指标
快速验证评估功能是否正常工作
"""
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_training.train_model import (
    calculate_precision_recall_f1,
    calculate_ndcg,
    calculate_hit_rate,
    calculate_diversity
)


def test_precision_recall_f1():
    """测试Precision, Recall, F1计算"""
    print("=" * 60)
    print("测试 Precision, Recall, F1 计算")
    print("=" * 60)
    
    # 测试用例1
    recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    relevant = [2, 5, 11, 15]
    k = 10
    
    precision, recall, f1 = calculate_precision_recall_f1(recommended, relevant, k)
    
    print(f"\n测试用例1:")
    print(f"推荐列表: {recommended}")
    print(f"相关物品: {relevant}")
    print(f"K = {k}")
    print(f"结果:")
    print(f"  • Precision@{k}: {precision:.4f} (期望: 0.2000)")
    print(f"  • Recall@{k}: {recall:.4f} (期望: 0.5000)")
    print(f"  • F1@{k}: {f1:.4f} (期望: 0.2857)")
    
    # 测试用例2
    recommended = [1, 2, 3, 4, 5]
    relevant = [1, 2, 3, 4, 5]
    k = 5
    
    precision, recall, f1 = calculate_precision_recall_f1(recommended, relevant, k)
    
    print(f"\n测试用例2 (完美推荐):")
    print(f"推荐列表: {recommended}")
    print(f"相关物品: {relevant}")
    print(f"K = {k}")
    print(f"结果:")
    print(f"  • Precision@{k}: {precision:.4f} (期望: 1.0000)")
    print(f"  • Recall@{k}: {recall:.4f} (期望: 1.0000)")
    print(f"  • F1@{k}: {f1:.4f} (期望: 1.0000)")


def test_ndcg():
    """测试NDCG计算"""
    print("\n" + "=" * 60)
    print("测试 NDCG 计算")
    print("=" * 60)
    
    # 测试用例1: 相关物品在前面
    recommended = [1, 2, 3, 4, 5]
    relevant = [1, 2]
    k = 5
    
    ndcg = calculate_ndcg(recommended, relevant, k)
    
    print(f"\n测试用例1 (相关物品在前面):")
    print(f"推荐列表: {recommended}")
    print(f"相关物品: {relevant}")
    print(f"K = {k}")
    print(f"NDCG@{k}: {ndcg:.4f} (期望: ~1.0000)")
    
    # 测试用例2: 相关物品在后面
    recommended = [1, 2, 3, 4, 5]
    relevant = [4, 5]
    k = 5
    
    ndcg = calculate_ndcg(recommended, relevant, k)
    
    print(f"\n测试用例2 (相关物品在后面):")
    print(f"推荐列表: {recommended}")
    print(f"相关物品: {relevant}")
    print(f"K = {k}")
    print(f"NDCG@{k}: {ndcg:.4f} (期望: < 1.0000)")


def test_hit_rate():
    """测试Hit Rate计算"""
    print("\n" + "=" * 60)
    print("测试 Hit Rate 计算")
    print("=" * 60)
    
    # 测试用例1: 命中
    recommended = [1, 2, 3, 4, 5]
    relevant = [3, 10, 11]
    k = 5
    
    hit = calculate_hit_rate(recommended, relevant, k)
    
    print(f"\n测试用例1 (命中):")
    print(f"推荐列表: {recommended}")
    print(f"相关物品: {relevant}")
    print(f"K = {k}")
    print(f"Hit Rate@{k}: {hit:.4f} (期望: 1.0000)")
    
    # 测试用例2: 未命中
    recommended = [1, 2, 3, 4, 5]
    relevant = [10, 11, 12]
    k = 5
    
    hit = calculate_hit_rate(recommended, relevant, k)
    
    print(f"\n测试用例2 (未命中):")
    print(f"推荐列表: {recommended}")
    print(f"相关物品: {relevant}")
    print(f"K = {k}")
    print(f"Hit Rate@{k}: {hit:.4f} (期望: 0.0000)")


def test_diversity():
    """测试Diversity计算"""
    print("\n" + "=" * 60)
    print("测试 Diversity 计算")
    print("=" * 60)
    
    # 测试用例1: 完全不同的推荐列表
    recommendations = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ]
    
    diversity = calculate_diversity(recommendations)
    
    print(f"\n测试用例1 (完全不同):")
    print(f"推荐列表1: {recommendations[0]}")
    print(f"推荐列表2: {recommendations[1]}")
    print(f"推荐列表3: {recommendations[2]}")
    print(f"Diversity: {diversity:.4f} (期望: 1.0000)")
    
    # 测试用例2: 完全相同的推荐列表
    recommendations = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]
    ]
    
    diversity = calculate_diversity(recommendations)
    
    print(f"\n测试用例2 (完全相同):")
    print(f"推荐列表1: {recommendations[0]}")
    print(f"推荐列表2: {recommendations[1]}")
    print(f"推荐列表3: {recommendations[2]}")
    print(f"Diversity: {diversity:.4f} (期望: 0.0000)")
    
    # 测试用例3: 部分重叠
    recommendations = [
        [1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7],
        [5, 6, 7, 8, 9]
    ]
    
    diversity = calculate_diversity(recommendations)
    
    print(f"\n测试用例3 (部分重叠):")
    print(f"推荐列表1: {recommendations[0]}")
    print(f"推荐列表2: {recommendations[1]}")
    print(f"推荐列表3: {recommendations[2]}")
    print(f"Diversity: {diversity:.4f} (期望: 0.4-0.7)")


def main():
    """运行所有测试"""
    print("\n🎯 开始测试新的评估指标\n")
    
    test_precision_recall_f1()
    test_ndcg()
    test_hit_rate()
    test_diversity()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)
    print("\n💡 提示:")
    print("  1. 查看 EVALUATION_METRICS.md 了解每个指标的详细说明")
    print("  2. 运行完整的模型训练来查看真实数据上的评估结果")
    print("  3. 使用 python model_training/train_model.py 进行本地测试")


if __name__ == "__main__":
    main()


