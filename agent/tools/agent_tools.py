import os
import datetime
from utils.logger_handler import logger
from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from memory.store import MemoryStore
from memory.schemas import Memory, MemoryQuery


rag = RagSummarizeService()
memory_store = MemoryStore()


@tool(description="从个人知识库向量存储中检索与查询相关的参考资料。当用户的问题需要专业知识库内容来回答时使用此工具。入参query为检索关键词。")
def rag_summarize(query: str) -> str:
    """从向量存储中检索参考资料"""
    return rag.rag_summarize(query)


@tool(description="获取当前的日期和时间，以字符串形式返回。格式：YYYY-MM-DD HH:MM:SS。当用户询问当前时间、日期或需要时效性信息时使用。")
def get_current_time() -> str:
    """获取当前日期时间"""
    now = datetime.datetime.now()
    return now.strftime("当前时间是：%Y年%m月%d日 %H时%M分%S秒")


@tool(description='从长期记忆中搜索与用户相关的历史信息。当用户提到「之前」「记得」「上次」等词汇，或需要回忆用户偏好、历史记录时使用。入参query为搜索关键词。')
def search_memory(query: str) -> str:
    """从长期记忆中搜索信息"""
    results = memory_store.search(MemoryQuery(query=query, top_k=5))
    if not results:
        return "未找到相关记忆。"
    # 返回简洁格式，由 Agent 自行组织成自然语言
    lines = ["检索到的记忆："]
    for mem in results:
        lines.append(f"[{mem.category}] {mem.content}")
    return "\n".join(lines)


@tool(description="将关键信息存入长期记忆。当用户表达了明确偏好、目标、反馈或个人事实时使用此工具。入参content为记忆内容，category为分类（preference/fact/goal/feedback/personality），importance为重要性（0-1之间）。")
def store_memory(content: str, category: str = "fact", importance: float = 0.5) -> str:
    """将信息存入长期记忆"""
    memory = Memory(content=content, category=category, importance=importance, source="对话")
    memory_id = memory_store.add(memory)
    return f"记忆已保存 [{memory_id}]: {content[:50]}..."
