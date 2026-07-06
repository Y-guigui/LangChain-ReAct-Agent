"""
指令解析器 — 自然语言指令解析与格式化
将用户的复杂指令解析为结构化数据，供 Agent 和工具使用
"""
import re
from typing import Dict, Any, Optional, List
from utils.logger_handler import logger


class IntentParser:
    """
    指令解析器

    支持解析：
    1. 命令式指令 — "搜索..."、"生成..."、"查询..."
    2. 查询式指令 — "...是什么"、"...怎么样"
    3. 条件式指令 — "如果...就..."
    4. 复合指令 — 包含多个子任务的指令
    """

    # 命令前缀模式
    COMMAND_PREFIXES = {
        "search": ["搜索", "查找", "查询", "search", "find", "look up"],
        "generate": ["生成", "创建", "制作", "生成", "generate", "create", "make"],
        "summarize": ["总结", "概括", "summarize", "summary"],
        "compare": ["对比", "比较", "compare", "versus", "vs"],
        "explain": ["解释", "说明", "explain", "describe"],
    }

    # 条件模式
    CONDITIONAL_PATTERNS = [
        r"如果(.+?)就(.+)",  # 如果...就...
        r"假如(.+?)那么(.+)",  # 假如...那么...
        r"(.+?)的情况下(.+)",  # ...的情况下...
    ]

    def __init__(self):
        logger.info("[IntentParser] 指令解析器初始化")

    def parse(self, query: str) -> Dict[str, Any]:
        """
        解析用户输入为结构化指令

        :param query: 用户输入
        :return: 解析结果 {command, subject, conditions, modifiers}
        """
        result = {
            "raw": query,
            "command": None,
            "subject": None,
            "conditions": [],
            "modifiers": {},
            "is_conditional": False,
        }

        # 1. 识别命令类型
        result["command"] = self._detect_command(query)

        # 2. 提取主体内容
        result["subject"] = self._extract_subject(query)

        # 3. 检查条件模式
        conditions = self._extract_conditions(query)
        if conditions:
            result["conditions"] = conditions
            result["is_conditional"] = True

        # 4. 提取修饰词（如时间、范围、格式等）
        result["modifiers"] = self._extract_modifiers(query)

        logger.debug(f"[IntentParser] 解析结果: {result}")
        return result

    def _detect_command(self, query: str) -> Optional[str]:
        """检测命令类型"""
        for cmd_type, prefixes in self.COMMAND_PREFIXES.items():
            for prefix in prefixes:
                if prefix in query or prefix.lower() in query.lower():
                    return cmd_type
        return None

    def _extract_subject(self, query: str) -> Optional[str]:
        """提取指令主体"""
        # 去除命令前缀和标点
        cleaned = re.sub(r"^[\s\u3000]*[A-Za-z\u4e00-\u9fa5]+[，,、\s]", "", query)
        cleaned = re.sub(r"[?？!！.。]$", "", cleaned).strip()
        return cleaned if cleaned else None

    def _extract_conditions(self, query: str) -> List[str]:
        """提取条件子句"""
        conditions = []
        for pattern in self.CONDITIONAL_PATTERNS:
            match = re.search(pattern, query)
            if match:
                conditions.append(match.group(1).strip())
        return conditions

    def _extract_modifiers(self, query: str) -> Dict[str, Any]:
        """提取修饰词（时间、范围、格式等）"""
        modifiers = {}

        # 时间修饰
        time_patterns = [
            (r"(\d{4})年", "year"),
            (r"(\d{1,2})月", "month"),
            (r"(\d{1,2})日", "day"),
            (r"最近", "recent"),
            (r"上周", "last_week"),
            (r"上个月", "last_month"),
        ]
        for pattern, key in time_patterns:
            match = re.search(pattern, query)
            if match:
                modifiers["time"] = modifiers.get("time", {})
                modifiers["time"][key] = match.group(1) if match.groups() else True

        # 范围修饰
        if re.search(r"最.+的", query):
            modifiers["scope"] = "best"
        if re.search(r"所有|全部", query):
            modifiers["scope"] = "all"
        if re.search(r"前\d+个|前\d+名", query):
            match = re.search(r"前(\d+)", query)
            if match:
                modifiers["top_n"] = int(match.group(1))

        # 格式修饰
        if re.search(r"表格|列表|table|list", query):
            modifiers["format"] = "table"
        elif re.search(r"总结|概括|summary", query):
            modifiers["format"] = "summary"

        return modifiers


# 全局单例
intent_parser = IntentParser()
