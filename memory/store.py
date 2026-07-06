"""记忆存储 — SQLite 元数据 + Chroma 向量存储"""
import os
import sqlite3
import uuid
from datetime import datetime
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from memory.schemas import Memory, MemoryQuery
from model.factory import embed_model
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


class MemoryStore:
    """长期记忆存储（SQLite + Chroma 双写）"""

    def __init__(self, db_path: str = None, persist_dir: str = None):
        # SQLite 路径
        self.db_path = db_path or get_abs_path("chroma_db/memory.sqlite")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Chroma 向量存储
        self.persist_dir = persist_dir or get_abs_path("chroma_db")
        self.vector_store = Chroma(
            collection_name="long_term_memory",
            embedding_function=embed_model,
            persist_directory=self.persist_dir,
        )

        # 初始化 SQLite 表
        self._init_db()

    def _init_db(self):
        """创建 SQLite 表结构"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                importance REAL NOT NULL,
                source TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON memories(category)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)
        """)
        conn.commit()
        conn.close()
        logger.info("[MemoryStore] SQLite 初始化完成")

    def add(self, memory: Memory) -> str:
        """
        添加记忆（SQLite + Chroma 双写）

        :param memory: Memory 对象
        :return: memory_id
        """
        memory_id = str(uuid.uuid4())[:8]
        memory.memory_id = memory_id

        # 1. 写入 SQLite（元数据）
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO memories (memory_id, content, category, importance, source, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (memory_id, memory.content, memory.category, memory.importance, memory.source, memory.created_at.isoformat()),
        )
        conn.commit()
        conn.close()

        # 2. 写入 Chroma（向量）
        doc = Document(
            page_content=memory.content,
            metadata={
                "memory_id": memory_id,
                "category": memory.category,
                "importance": memory.importance,
                "source": memory.source,
            },
        )
        self.vector_store.add_documents([doc])

        logger.info(f"[MemoryStore] 已添加记忆 [{memory_id}]: {memory.content[:50]}...")
        return memory_id

    def search(self, query: MemoryQuery) -> List[Memory]:
        """
        搜索记忆（语义检索 + 元数据过滤）

        :param query: MemoryQuery 对象
        :return: 记忆列表
        """
        # 1. 向量检索
        retriever = self.vector_store.as_retriever(search_kwargs={"k": query.top_k})
        docs = retriever.invoke(query.query)

        # 2. 构建结果列表
        results = []
        for doc in docs:
            meta = doc.metadata
            # 过滤
            if query.category and meta.get("category") != query.category:
                continue
            if meta.get("importance", 0) < query.min_importance:
                continue

            results.append(Memory(
                memory_id=meta.get("memory_id"),
                content=doc.page_content,
                category=meta.get("category", "fact"),
                importance=meta.get("importance", 0.5),
                source=meta.get("source", ""),
            ))

        logger.info(f"[MemoryStore] 搜索 '{query.query[:30]}...' 返回 {len(results)} 条记忆")
        return results

    def get_by_category(self, category: str, limit: int = 20) -> List[Memory]:
        """按分类获取记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT memory_id, content, category, importance, source, created_at FROM memories WHERE category = ? ORDER BY created_at DESC LIMIT ?",
            (category, limit),
        )
        rows = cursor.fetchall()
        conn.close()

        return [Memory(
            memory_id=r[0], content=r[1], category=r[2],
            importance=r[3], source=r[4], created_at=datetime.fromisoformat(r[5]),
        ) for r in rows]

    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        # 从 SQLite 删除
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        conn.commit()
        conn.close()

        # 从 Chroma 删除（通过 collection 操作）
        try:
            collection = self.vector_store._collection
            results = collection.get(include=["metadatas"])
            if results and results.get("ids"):
                ids_to_delete = []
                for i, meta in enumerate(results.get("metadatas", [])):
                    if meta and meta.get("memory_id") == memory_id:
                        ids_to_delete.append(results["ids"][i])
                if ids_to_delete:
                    collection.delete(ids=ids_to_delete)
        except Exception as e:
            logger.warning(f"[MemoryStore] 向量删除警告: {e}")

        logger.info(f"[MemoryStore] 已删除记忆 [{memory_id}]")
        return True

    def stats(self) -> dict:
        """获取统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]

        cursor = conn.execute("SELECT category, COUNT(*) FROM memories GROUP BY category")
        categories = {r[0]: r[1] for r in cursor.fetchall()}
        conn.close()

        return {"total": total, "categories": categories}
