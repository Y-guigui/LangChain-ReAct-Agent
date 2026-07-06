"""
评估 API — 运行评估、查看结果、生成测试数据
"""
from fastapi import APIRouter
from evaluation.evaluator import RAGEvaluator

from utils.logger_handler import logger

router = APIRouter(tags=["评估"])

_evaluator: RAGEvaluator | None = None
_last_results: dict = {}


def get_evaluator() -> RAGEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator()
    return _evaluator


@router.post("/evaluation/run")
async def run_evaluation(n_samples: int = 20, top_k: int = 5):
    """
    运行 RAG 评估

    - n_samples: 自动生成的测试样本数
    - top_k: 评估 Top-K 指标
    """
    global _last_results
    logger.info(f"[评估] 开始评估: samples={n_samples}, top_k={top_k}")
    evaluator = get_evaluator()
    results = evaluator.run(n_samples=n_samples, top_k=top_k)
    _last_results = results
    return results


@router.get("/evaluation/results")
async def get_evaluation_results():
    """获取最近一次评估结果"""
    global _last_results
    if not _last_results:
        return {"error": "尚未运行评估，请先 POST /api/evaluation/run"}
    return _last_results


@router.get("/evaluation/datasets")
async def get_benchmark_datasets(n_samples: int = 20):
    """查看自动生成的测试数据集（不运行评估）"""
    evaluator = get_evaluator()
    datasets = evaluator.benchmark.generate_from_knowledge(n_samples)
    return {"total": len(datasets), "samples": datasets}
