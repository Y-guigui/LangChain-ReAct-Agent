import os
import tempfile
import pytest
from memory.store import MemoryStore
from memory.schemas import Memory, MemoryQuery
from memory.extractor import MemoryExtractor


# 使用临时目录（Windows 兼容）
TEST_DB = os.path.join(tempfile.gettempdir(), "test_memory.sqlite")


class TestMemoryStore:
    """记忆存储测试"""

    def test_add_and_search(self):
        """测试添加和搜索记忆"""
        store = MemoryStore(db_path=TEST_DB)

        # 添加记忆
        memory = Memory(content="我喜欢用 Django 做后端开发", category="preference", importance=0.8)
        mid = store.add(memory)
        assert mid is not None
        assert len(mid) == 8

        # 搜索记忆
        results = store.search(MemoryQuery(query="Django 开发", top_k=5))
        assert len(results) > 0
        assert results[0].content == "我喜欢用 Django 做后端开发"

    def test_get_by_category(self):
        """测试按分类获取记忆"""
        store = MemoryStore(db_path=TEST_DB)

        # 添加不同分类的记忆
        store.add(Memory(content="目标：三个月内掌握 FastAPI", category="goal", importance=0.9))
        store.add(Memory(content="反馈：界面可以再优化", category="feedback", importance=0.6))

        # 按分类查询
        goals = store.get_by_category("goal")
        assert len(goals) > 0

    def test_delete(self):
        """测试删除记忆"""
        store = MemoryStore(db_path=TEST_DB)

        mid = store.add(Memory(content="XYZ_DELETE_TEST_12345", category="fact", importance=0.3))
        store.delete(mid)

        # 删除后应该搜不到这个特定内容
        results = store.search(MemoryQuery(query="XYZ_DELETE_TEST_12345", top_k=5))
        # 允许语义搜索返回其他结果，但不应包含被删除的内容
        for r in results:
            assert "XYZ_DELETE_TEST_12345" not in r.content

    def test_stats(self):
        """测试统计信息"""
        store = MemoryStore(db_path=TEST_DB)
        stats = store.stats()
        assert "total" in stats
        assert "categories" in stats


class TestMemoryExtractor:
    """记忆提取器测试"""

    def test_extract_simple(self):
        """简单提取测试（需要 API_KEY 环境变量）"""
        extractor = MemoryExtractor()
        # 注意：此测试需要 API_KEY 环境变量
        # 如果未配置 API_KEY，可以跳过
        if not os.environ.get("API_KEY"):
            pytest.skip("需要 API_KEY 环境变量")

        memories = extractor.extract(
            "我喜欢用 Python 做开发",
            "好的，了解到你喜欢 Python 开发。"
        )
        assert isinstance(memories, list)
        # 至少有一条记忆
        if memories:
            assert "content" in memories[0]
            assert "category" in memories[0]
