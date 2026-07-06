"""
对话状态跟踪器 — 意图识别、实体提取、槽位管理
基于规则和关键词匹配，后续可升级为 LLM 驱动的意图识别
"""
import re
from typing import Dict, List, Optional, Any
from core.constants import (
    INTENT_GENERAL, INTENT_REPORT, INTENT_MEMORY_QUERY,
    INTENT_KNOWLEDGE, INTENT_CLARIFICATION,
    TOOL_RAG, TOOL_FETCH_DATA, TOOL_FILL_REPORT, TOOL_MEMORY,
    TOOL_USER_ID, TOOL_MONTH,
)
from context.schemas import DialogueState
from utils.logger_handler import logger


class DialogueTracker:
    """
    对话状态跟踪器

    负责：
    1. 意图识别 — 基于关键词和规则匹配用户意图
    2. 实体提取 — 从用户输入中提取关键实体（如城市、月份、用户ID等）
    3. 槽位填充 — 跟踪已收集和缺失的信息槽位
    4. 状态转移 — 根据当前状态和用户输入决定下一步动作
    """

    # 意图关键词映射
    INTENT_KEYWORDS = {
        INTENT_REPORT: [
            "报告", "使用报告", "生成报告", "月度报告", "总结", "usage report",
            "report", "generate report", "summary",
        ],
        INTENT_MEMORY_QUERY: [
            "记得", "之前", "上次", "之前说过", "你还记得", "历史",
            "remember", "before", "last time", "history",
        ],
        INTENT_KNOWLEDGE: [
            "知识库", "文档", "资料", "笔记", "knowledge", "document", "notes",
        ],
    }

    # 实体提取模式
    ENTITY_PATTERNS = {
        "city": r"([\u4e00-\u9fa5]{2,6}市?|[\u4e00-\u9fa5]{2,6}省?)",  # 城市名
        "month": r"(\d{1,2})月|(\d{4})年(\d{1,2})月",  # 月份
        "user_id": r"用户ID[是为:]\s*(\d+)",  # 用户ID
        "number": r"\b(\d+)\b",  # 数字
    }

    # 报告生成槽位
    REPORT_SLOTS = ["user_id", "month"]

    def __init__(self):
        logger.info("[DialogueTracker] 对话状态跟踪器初始化")

    def detect_intent(self, query: str, current_state: DialogueState) -> str:
        """
        识别用户意图

        :param query: 用户输入
        :param current_state: 当前对话状态
        :return: 意图类型
        """
        query_lower = query.lower()

        # 如果上一轮需要澄清，当前输入可能是澄清回复
        if current_state.pending_clarification:
            # 检查是否是有效回复（有内容且不是简单否定）
            if len(query) > 2 and not re.search(r"不知道|不清楚|没有|没有|no|not|don't", query_lower):
                return INTENT_GENERAL
            return INTENT_CLARIFICATION

        # 关键词匹配
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in query or kw in query_lower:
                    logger.debug(f"[DialogueTracker] 意图识别: {intent} (关键词: {kw})")
                    return intent

        return INTENT_GENERAL

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """
        从用户输入中提取实体

        :param query: 用户输入
        :return: 提取的实体字典
        """
        entities = {}

        # 城市提取
        city_match = re.search(self.ENTITY_PATTERNS["city"], query)
        if city_match:
            entities["city"] = city_match.group(1)

        # 月份提取
        month_match = re.search(self.ENTITY_PATTERNS["month"], query)
        if month_match:
            if month_match.group(1):
                entities["month"] = f"2026-{int(month_match.group(1)):02d}"
            elif month_match.group(2) and month_match.group(3):
                entities["month"] = f"{month_match.group(2)}-{int(month_match.group(3)):02d}"

        # 用户ID提取
        user_id_match = re.search(self.ENTITY_PATTERNS["user_id"], query)
        if user_id_match:
            entities["user_id"] = user_id_match.group(1)

        # 数字提取（作为通用数字）
        numbers = re.findall(r"\b(\d+)\b", query)
        if numbers:
            entities["numbers"] = numbers

        logger.debug(f"[DialogueTracker] 提取实体: {entities}")
        return entities

    def fill_slots(self, current_state: DialogueState, entities: Dict[str, Any]) -> DialogueState:
        """
        将提取的实体填充到槽位中

        :param current_state: 当前状态
        :param entities: 提取的实体
        :return: 更新后的状态
        """
        for key, value in entities.items():
            if key not in ("numbers",):
                current_state.slot_values[key] = value

        return current_state

    def check_missing_slots(self, state: DialogueState) -> List[str]:
        """
        检查缺失的槽位

        :param state: 当前状态
        :return: 缺失的槽位列表
        """
        if state.intent != INTENT_REPORT:
            return []

        missing = []
        for slot in self.REPORT_SLOTS:
            if slot not in state.slot_values or not state.slot_values[slot]:
                missing.append(slot)
        return missing

    def update_state(self, query: str, current_state: DialogueState) -> DialogueState:
        """
        根据用户输入更新对话状态

        :param query: 用户输入
        :param current_state: 当前状态
        :return: 更新后的状态
        """
        # 识别意图
        intent = self.detect_intent(query, current_state)
        current_state.intent = intent

        # 提取实体并填充槽位
        entities = self.extract_entities(query)
        current_state = self.fill_slots(current_state, entities)

        # 更新轮次
        current_state.turn_count += 1

        # 检查是否需要澄清
        if intent == INTENT_REPORT:
            missing = self.check_missing_slots(current_state)
            if missing:
                current_state.pending_clarification = f"请提供以下信息: {', '.join(missing)}"
                current_state.requires_follow_up = True
            else:
                current_state.pending_clarification = None
                current_state.requires_follow_up = False

        logger.info(f"[DialogueTracker] 状态更新: intent={intent}, slots={current_state.slot_values}")
        return current_state

    def get_next_action(self, state: DialogueState) -> Dict[str, Any]:
        """
        根据当前状态决定下一步动作

        :param state: 当前对话状态
        :return: 下一步动作建议
        """
        if state.pending_clarification:
            return {
                "action": "clarify",
                "message": state.pending_clarification,
                "requires_tool": False,
            }

        if state.intent == INTENT_REPORT:
            return {
                "action": "generate_report",
                "requires_tool": True,
                "tools": [TOOL_USER_ID, TOOL_MONTH, TOOL_FILL_REPORT, TOOL_FETCH_DATA],
            }

        if state.intent == INTENT_MEMORY_QUERY:
            return {
                "action": "search_memory",
                "requires_tool": True,
                "tools": [TOOL_MEMORY],
            }

        if state.intent == INTENT_KNOWLEDGE:
            return {
                "action": "search_knowledge",
                "requires_tool": True,
                "tools": [TOOL_RAG],
            }

        return {
            "action": "general_chat",
            "requires_tool": False,
            "tools": [TOOL_RAG],
        }


# 全局单例
dialogue_tracker = DialogueTracker()
