"""
评估器 — 运行 RAG 评估 Pipeline
"""
from typing import List
from evaluation.metrics import recall_at_k, precision_at_k, mrr, ndcg_at_k, f1_score, hit_rate
from evaluation.benchmark import BenchmarkDataset
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger


class RAGEvaluator:
    """RAG 评估器"""

    def __init__(self):
        self.vs = VectorStoreService()
        self.benchmark = BenchmarkDataset()

    def run(
        self,
        test_cases: list[dict] = None,
        n_samples: int = 20,
        top_k: int = 5,
    ) -> dict:
        """
        运行评估 Pipeline

        :param test_cases: 测试用例列表（可选，不提供则自动生成）
        :param n_samples: 自动生成的样本数
        :param top_k: 评估 Top-K 指标
        :return: 评估报告
        """
        if test_cases is None:
            test_cases = self.benchmark.generate_from_knowledge(n_samples)

        if not test_cases:
            return {"error": "没有可用的测试数据"}

        results = []
        metrics_sum = {
            "recall": [],
            "precision": [],
            "mrr": [],
            "ndcg": [],
            "hit_rate": [],
            "f1": [],
        }
        failures = []

        for i, case in enumerate(test_cases):
            query = case["question"]
            relevant = set(case.get("ground_truth_docs", []))

            # 执行 RAG 检索
            try:
                retrieved_docs = self.vs.search(query, top_k=top_k)
                retrieved_sources = list(dict.fromkeys(
                    doc.metadata.get("source", "") for doc in retrieved_docs
                ))  # 去重（同一文件可能有多个chunk）
            except Exception as e:
                logger.error(f"[Evaluator] 检索失败 [{query[:30]}...]: {str(e)}")
                failures.append({"question": query, "error": str(e)})
                continue

            # 计算各项指标
            recall = recall_at_k(retrieved_sources, relevant, k=top_k)
            precision = precision_at_k(retrieved_sources, relevant, k=top_k)
            mr = mrr(retrieved_sources, relevant)
            nd = ndcg_at_k(retrieved_sources, relevant, k=top_k)
            hit = hit_rate(retrieved_sources, relevant, k=top_k)
            f1 = f1_score(precision, recall)

            metrics_sum["recall"].append(recall)
            metrics_sum["precision"].append(precision)
            metrics_sum["mrr"].append(mr)
            metrics_sum["ndcg"].append(nd)
            metrics_sum["hit_rate"].append(hit)
            metrics_sum["f1"].append(f1)

            results.append({
                "id": i + 1,
                "question": query,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "mrr": round(mr, 4),
                "ndcg": round(nd, 4),
                "hit_rate": hit,
                "retrieved_count": len(retrieved_sources),
                "relevant_count": len(relevant),
            })

            # 记录失败案例（recall=0）
            if recall == 0:
                failures.append({
                    "question": query,
                    "retrieved_docs": retrieved_sources[:3],
                    "expected_docs": list(relevant)[:3],
                })

        # 计算平均值
        avg_metrics = {}
        for metric, values in metrics_sum.items():
            if values:
                avg_metrics[metric] = round(sum(values) / len(values), 4)
            else:
                avg_metrics[metric] = 0.0

        # 构建报告
        report = {
            "summary": {
                "total_tests": len(results),
                "average_metrics": avg_metrics,
                "top_k": top_k,
            },
            "details": results,
            "failures": failures[:10],  # 只展示前10个失败案例
            "failure_count": len(failures),
        }

        logger.info(f"[Evaluator] 评估完成: {len(results)} 个测试, 平均 Recall={avg_metrics['recall']}")
        return report
