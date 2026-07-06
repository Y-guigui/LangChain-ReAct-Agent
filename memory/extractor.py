"""记忆提取器 — 从对话中提取关键信息"""
import json
from typing import List
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.logger_handler import logger


EXTRACTION_PROMPT = """请从以下对话中提取关键信息，判断是否有值得长期记忆的内容。

需要提取的信息类型：
1. 用户偏好（preference）：用户明确表达的喜好、习惯、偏好设置
2. 事实信息（fact）：用户提到的个人事实、背景、经历
3. 目标/计划（goal）：用户提到的学习目标、工作计划、待办事项
4. 用户反馈（feedback）：用户对系统、产品的评价或建议
5. 性格/习惯（personality）：用户展现出的性格特点或工作习惯

对话内容：
{dialogue}

请分析这段对话中是否包含值得记忆的信息。如果有，以 JSON 格式输出：
[
    {"content": "记忆内容", "category": "preference|fact|goal|feedback|personality", "importance": 0.8}
]

如果没有值得记忆的信息，输出空数组 []。

只输出 JSON，不要添加任何解释文字。"""


class MemoryExtractor:
    """从对话中提取记忆"""

    def __init__(self):
        self.prompt = PromptTemplate.from_template(EXTRACTION_PROMPT)
        self.chain = self.prompt | chat_model

    def extract(self, user_message: str, assistant_message: str) -> List[dict]:
        """
        从一轮对话中提取记忆

        :param user_message: 用户消息
        :param assistant_message: 助手消息
        :return: 记忆列表 [{content, category, importance}]
        """
        dialogue = f"用户：{user_message}\n\n助手：{assistant_message}"
        try:
            result = self.chain.invoke({"dialogue": dialogue})
            text = result.content if hasattr(result, "content") else str(result)

            # 提取 JSON
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                memories = json.loads(text[start:end+1])
                logger.info(f"[MemoryExtractor] 提取到 {len(memories)} 条记忆")
                return memories
            return []
        except Exception as e:
            logger.warning(f"[MemoryExtractor] 提取失败: {e}")
            return []
