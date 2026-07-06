"""
评估指标计算模块 — 召回率、精确率、MRR、NDCG
"""
from typing import List, Set
import math


def recall_at_k(
    retrieved_ids: List[str], relevant_ids: Set[str], k: int = None
) -> float:
    """
    计算 Recall@K

    Recall@K = |上K个结果中相关文档数| / |总相关文档数|
    """
    if k:
        retrieved_ids = retrieved_ids[:k]
    if len(relevant_ids) == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(
    retrieved_ids: List[str], relevant_ids: Set[str], k: int = None
) -> float:
    """
    计算 Precision@K

    Precision@K = |上K个结果中相关文档数| / K
    """
    if k:
        retrieved_ids = retrieved_ids[:k]
    if len(retrieved_ids) == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
    return hits / len(retrieved_ids)


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)

    MRR = 1 / (第一个相关文档的位置)
    """
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    relevance_scores: dict = None,
    k: int = None,
) -> float:
    """
    计算 NDCG@K (Normalized Discounted Cumulative Gain)

    使用分级相关性（默认：命中=1，未命中=0）
    """
    if k:
        retrieved_ids = retrieved_ids[:k]
    if len(retrieved_ids) == 0:
        return 0.0

    # 构建相关性分数列表
    if relevance_scores:
        scores = [relevance_scores.get(doc_id, 0) for doc_id in retrieved_ids]
    else:
        scores = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids]

    # DCG = Σ (2^reli - 1) / log2(i + 1)
    dcg = sum(
        (2 ** score - 1) / math.log2(i + 2)
        for i, score in enumerate(scores)
    )

    # IDCG (理想排序)
    ideal_scores = sorted(scores, reverse=True)
    idcg = sum(
        (2 ** score - 1) / math.log2(i + 2)
        for i, score in enumerate(ideal_scores)
    )

    if idcg == 0:
        return 0.0
    return dcg / idcg


def f1_score(precision: float, recall: float) -> float:
    """计算 F1 分数"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def hit_rate(retrieved_ids: List[str], relevant_ids: Set[str], k: int = None) -> float:
    """
    计算命中率 (Hit Rate@K)

    Hit Rate = 结果中是否至少有一个相关文档
    """
    if k:
        retrieved_ids = retrieved_ids[:k]
    if len(retrieved_ids) == 0:
        return 0.0
    hits = any(doc_id in relevant_ids for doc_id in retrieved_ids)
    return 1.0 if hits else 0.0
