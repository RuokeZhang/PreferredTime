# 评估系统使用指南

本文档说明如何使用新的推荐系统评估功能。

## 🎉 更新内容

### 新增评估指标

之前的评估系统只有简单的**覆盖率(Coverage)**指标，现在已升级为业界标准的完整评估体系：

#### 准确率指标
- ✅ **Precision@K**: 推荐准确率
- ✅ **Recall@K**: 推荐召回率  
- ✅ **F1-Score@K**: 综合评分
- ✅ **NDCG@K**: 排序质量（考虑位置权重）
- ✅ **Hit Rate@K**: 命中率

#### 系统指标
- ✅ **Coverage**: 推荐覆盖率
- ✅ **Diversity**: 推荐多样性

所有指标在 **K=5, 10, 20** 三个位置上进行评估。

---

## 📖 如何使用

### 1. 运行完整的模型训练和评估

```bash
# 方式1: 本地测试
python model_training/train_model.py

# 方式2: 通过Airflow DAG（自动化）
# 每天02:00自动运行
```

### 2. 查看评估结果

评估会在日志中打印详细结果：

```
============================================================
✓ 评估完成 - 评估了 85 个用户
============================================================

📊 Top-5 指标:
  • Precision@5: 0.1520
  • Recall@5: 0.2845
  • F1@5: 0.1965
  • NDCG@5: 0.4231
  • Hit Rate@5: 0.7412

📊 Top-10 指标:
  • Precision@10: 0.1340
  • Recall@10: 0.4128
  • F1@10: 0.2015
  • NDCG@10: 0.4567
  • Hit Rate@10: 0.8235

📊 Top-20 指标:
  • Precision@20: 0.1105
  • Recall@20: 0.5621
  • F1@20: 0.1845
  • NDCG@20: 0.4789
  • Hit Rate@20: 0.8941

📈 系统级指标:
  • 覆盖率 (Coverage): 0.3254 (325/1000)
  • 多样性 (Diversity): 0.7821
```

### 3. 理解评估指标

详细的指标说明请查看：[EVALUATION_METRICS.md](EVALUATION_METRICS.md)

**快速理解**:
- **Precision高** = 推荐的都是用户喜欢的
- **Recall高** = 用户喜欢的都被推荐了
- **NDCG高** = 用户喜欢的排在前面
- **Hit Rate高** = 大多数用户至少得到1个好推荐
- **Coverage高** = 推荐内容多样化，不局限于热门
- **Diversity高** = 不同用户得到个性化推荐

---

## 🚀 模型部署条件

新模型必须满足以下**最低质量要求**才能部署：

| 指标 | 最低要求 | 说明 |
|------|---------|------|
| **Precision@10** | ≥ 1% | 至少1%的准确率 |
| **Hit Rate@10** | ≥ 10% | 至少10%的用户能得到好推荐 |
| **Coverage** | ≥ 5% | 至少推荐5%的电影库 |
| **Evaluated Users** | ≥ 10人 | 评估样本足够 |

如果任何指标不达标，系统会：
1. ⚠️ 输出警告信息
2. ❌ 跳过模型部署
3. ✅ 保留当前生产模型

---

## 🧪 测试评估功能

### 运行单元测试

```bash
# 测试各个评估指标的计算是否正确
python test_evaluation.py
```

### 测试示例输出

```
🎯 开始测试新的评估指标

============================================================
测试 Precision, Recall, F1 计算
============================================================

测试用例1:
推荐列表: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
相关物品: [2, 5, 11, 15]
K = 10
结果:
  • Precision@10: 0.2000 (期望: 0.2000)
  • Recall@10: 0.5000 (期望: 0.5000)
  • F1@10: 0.2857 (期望: 0.2857)

...
```

---

## 📊 评估流程详解

### 1. 数据划分
```
对每个用户的评分数据:
├── 训练集 (80%)  ← 用于构建推荐模型
└── 测试集 (20%)  ← 用于评估推荐效果
```

### 2. 模型重建
使用训练集数据重新训练模型，模拟真实场景（不能用未来数据）

### 3. 生成推荐
为每个测试用户生成Top-K推荐列表

### 4. 对比评估
将推荐结果与测试集中的真实偏好对比，计算各项指标

### 5. 统计汇总
计算所有用户的平均指标，得到模型整体性能

---

## 🎯 如何改进模型

### 如果Precision/Recall较低

**可能原因**:
- 数据量不足
- 特征提取不够好
- 模型权重需要调整

**改进方法**:
```yaml
# config/config.yaml
model:
  hybrid:
    cf_weight: 0.6      # 调整协同过滤权重
    content_weight: 0.4  # 调整基于内容权重
  
  collaborative_filtering:
    n_neighbors: 20            # 增加相似用户数
    similarity_threshold: 0.1   # 降低相似度阈值
```

### 如果NDCG较低

**可能原因**:
- 推荐排序不够准确
- 相关物品排在后面

**改进方法**:
- 优化推荐分数计算
- 考虑使用Learning to Rank方法
- 增加用户偏好强度特征

### 如果Coverage/Diversity较低

**可能原因**:
- 过度推荐热门物品
- 协同过滤权重太高

**改进方法**:
```yaml
# config/config.yaml
model:
  recommendation:
    min_rating_threshold: 3.0  # 降低推荐阈值
  
  hybrid:
    cf_weight: 0.5           # 降低协同过滤权重
    content_weight: 0.5       # 提高基于内容权重
  
  content_based:
    top_n_similar_movies: 100  # 增加候选电影数
```

---

## 🔄 与Airflow集成

评估模块已集成到Airflow DAG中：

```
daily_model_retraining DAG:
├── Task 1: validate_data        (验证数据)
├── Task 2: extract_features     (提取特征)
├── Task 3: train_model          (训练模型)
├── Task 4: evaluate_model       (⭐ 评估模型 - 新增)
└── Task 5: deploy_model         (部署模型 - 使用评估结果)
```

**数据流**:
```
train_model 
  ↓ (XCom: model_path)
evaluate_model
  ↓ (XCom: metrics)
deploy_model → 根据metrics决定是否部署
```

---

## 📈 监控评估趋势

### 建议监控的指标

1. **核心指标**: Precision@10, NDCG@10, Hit Rate@10
2. **多样性指标**: Coverage, Diversity
3. **趋势**: 对比每次训练的指标变化

### 日志位置

```bash
# Airflow任务日志
~/airflow/logs/daily_model_retraining/evaluate_model/...

# 本地测试日志
# 输出到终端
```

---

## 🎓 延伸阅读

- [EVALUATION_METRICS.md](EVALUATION_METRICS.md) - 详细的指标说明
- [MODEL_UPDATE_FLOW.md](MODEL_UPDATE_FLOW.md) - 模型更新流程
- [ARCHITECTURE.md](ARCHITECTURE.md) - 系统架构

---

## ❓ 常见问题

### Q1: 为什么评估用户数比总用户数少？

**A**: 评估只针对评分数≥5的用户，且需要划分出测试集。评分太少的用户无法准确评估。

### Q2: 指标都很低怎么办？

**A**: 
1. 检查数据质量（是否有足够的评分数据）
2. 查看日志中的详细错误信息
3. 调整模型参数（见上文"如何改进模型"）
4. 考虑收集更多用户行为数据

### Q3: Coverage为什么不是越高越好？

**A**: 
- Coverage过高可能说明推荐太随机，准确性不高
- 需要在准确性和多样性之间找到平衡
- 典型的好值: 20%-50%

### Q4: 如何调整部署阈值？

**A**: 
修改 `model_training/train_model.py` 中的 `deploy_model` 函数：

```python
quality_checks = {
    'precision@10': (metrics.get('precision@10', 0), 0.01, "精确率过低"),
    'hit_rate@10': (metrics.get('hit_rate@10', 0), 0.1, "命中率过低"),
    'coverage': (metrics.get('coverage', 0), 0.05, "覆盖率过低"),
    # 根据实际情况调整阈值 ↑
}
```

---

## 💡 最佳实践

1. ✅ **定期监控**: 每次训练后查看评估指标趋势
2. ✅ **A/B测试**: 线上对比新旧模型的实际效果
3. ✅ **阈值调整**: 根据业务需求调整部署阈值
4. ✅ **数据质量**: 保证充足的用户行为数据
5. ✅ **参数调优**: 基于评估结果迭代优化模型参数

---

**祝评估愉快！** 🚀

如有问题，欢迎查看其他文档或提出issue。


