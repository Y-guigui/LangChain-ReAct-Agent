"""
基准测试数据集 — 基于知识库自动生成问答对

策略：
1. 从向量库文档中提取关键片段
2. 使用规则/模板生成问题
3. 每个问题附带 ground_truth 文档标识
"""
import re
import random
from pathlib import Path
from typing import List
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger


class BenchmarkDataset:
    """评估数据集管理"""

    def __init__(self):
        self.vs = VectorStoreService()

    def generate_from_knowledge(self, n_samples: int = 20) -> list[dict]:
        """
        从知识库自动生成测试数据集

        方法：从向量库中提取文档片段，基于关键词生成问题

        :param n_samples: 生成的样本数
        :return: [{question, ground_truth, relevant_keywords}]
        """
        try:
            # 获取所有文档
            docs = self.vs.list_documents()
            if not docs:
                logger.warning("[Benchmark] 知识库为空，无法生成测试数据")
                return []

            # 生成代表性查询
            queries = self._generate_queries(docs, n_samples)
            test_cases = []

            for query in queries:
                # 使用向量搜索找到相关文档
                results = self.vs.search(query["question"], top_k=5)

                # 提取相关文档 ID（以 source 作为标识）
                relevant_sources = set()
                for doc in results:
                    source = doc.metadata.get("source", "")
                    if source:
                        relevant_sources.add(source)

                test_cases.append({
                    "question": query["question"],
                    "expected_keywords": query.get("keywords", []),
                    "ground_truth_docs": list(relevant_sources),
                    "category": query.get("category", "general"),
                })

            logger.info(f"[Benchmark] 已生成 {len(test_cases)} 个测试用例")
            return test_cases

        except Exception as e:
            logger.error(f"[Benchmark] 生成测试数据失败: {str(e)}")
            return []

    def _generate_queries(self, docs: list[dict], n: int) -> list[dict]:
        """基于知识库文件内容生成查询"""
        queries = []
        templates = [
            ("{topic}是什么", "explain"),
            ("{topic}怎么做", "howto"),
            ("{topic}有哪些特点", "features"),
            ("请解释{topic}", "explain"),
            ("{topic}和什么相关", "relation"),
            ("怎么理解{topic}", "understand"),
            ("{topic}的底层原理是什么", "principle"),
            ("{topic}有什么优势", "advantage"),
            ("{topic}适用于什么场景", "scenario"),
            ("{topic}和哪个技术搭配用最好", "combination"),
        ]

        for doc in docs:
            fname = doc["filename"]
            if fname.endswith(".md"):
                # 从文件名生成主题
                base = fname.replace(".md", "")
                topics = self._extract_topics_from_file(doc.get("source", ""), base)
                for topic in topics[:n // len(docs) + 1]:
                    tmpl = random.choice(templates)
                    question = tmpl[0].format(topic=topic)
                    queries.append({
                        "question": question,
                        "category": tmpl[1],
                        "keywords": [topic],
                    })

        random.shuffle(queries)
        return queries[:n]

    def _extract_topics_from_file(self, source: str, base_name: str) -> list[str]:
        """从文件中提取主题关键词"""
        try:
            with open(source, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return [base_name]

        topics = []
        # 提取 ## 标题
        headers = re.findall(r"###?\s+(.+?)(?:\n|$)", content)
        for h in headers:
            # 清理标题：去掉标点数字
            clean = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", "", h)
            if len(clean) > 2 and len(clean) < 30:
                topics.append(clean)

        # 提取代码块附近的描述性文字作为主题
        code_contexts = re.findall(r"([\u4e00-\u9fa5a-zA-Z]+)", content)
        unique = list(set(topics + [base_name]))
        random.shuffle(unique)
        return unique[:8]

    @staticmethod
    def from_file(filepath: str) -> list[dict]:
        """从 JSON/JSONL 文件加载测试数据"""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
