"""
多轮对话上下文管理测试
覆盖: SessionManager, DialogueTracker, IntentParser, Session
"""
import time
from datetime import datetime, timedelta

import pytest

from context.session_manager import SessionManager
from context.tracker import DialogueTracker
from context.parser import IntentParser
from context.schemas import Session, Message, DialogueState
from core.exceptions import SessionNotFound, SessionExpired
from core.constants import INTENT_REPORT, INTENT_GENERAL, INTENT_MEMORY_QUERY


# ═══════════════════════════════════════════════════════════════
#  SessionManager 测试
# ═══════════════════════════════════════════════════════════════

class TestSessionManager:
    """会话管理器测试"""

    def test_create_session(self, fresh_session_manager):
        """测试创建会话"""
        sid = fresh_session_manager.create_session()
        assert sid is not None
        assert len(sid) == 8  # UUID 前8位

        session = fresh_session_manager.get_session(sid)
        assert session.session_id == sid
        assert session.status == "active"
        assert len(session.messages) == 0

    def test_get_session_not_found(self, fresh_session_manager):
        """测试获取不存在的会话"""
        with pytest.raises(SessionNotFound):
            fresh_session_manager.get_session("nonexistent")

    def test_add_message(self, fresh_session_manager):
        """测试添加消息"""
        sid = fresh_session_manager.create_session()
        session = fresh_session_manager.add_message(sid, "user", "你好")
        assert len(session.messages) == 1
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "你好"

        session = fresh_session_manager.add_message(sid, "assistant", "您好！")
        assert len(session.messages) == 2

    def test_get_recent_context(self, fresh_session_manager):
        """测试获取最近上下文"""
        sid = fresh_session_manager.create_session()
        for i in range(5):
            fresh_session_manager.add_message(sid, "user", f"消息{i}")
            fresh_session_manager.add_message(sid, "assistant", f"回复{i}")

        # 默认获取最近10条
        context = fresh_session_manager.get_recent_context(sid)
        assert len(context) == 10

        # 获取最近3条
        context = fresh_session_manager.get_recent_context(sid, n=3)
        assert len(context) == 3

    def test_session_expired(self, fresh_session_manager):
        """测试会话过期"""
        # 临时修改超时时间为1秒
        original_timeout = fresh_session_manager._session_timeout
        fresh_session_manager._session_timeout = 1

        try:
            sid = fresh_session_manager.create_session()
            session = fresh_session_manager.get_session(sid)
            assert not session.is_expired(1)

            # 等待过期
            time.sleep(2)
            assert session.is_expired(1)

            with pytest.raises(SessionExpired):
                fresh_session_manager.get_session(sid)
        finally:
            fresh_session_manager._session_timeout = original_timeout

    def test_close_and_delete_session(self, fresh_session_manager):
        """测试关闭和删除会话"""
        sid = fresh_session_manager.create_session()
        fresh_session_manager.close_session(sid)

        session = fresh_session_manager.get_session(sid)
        assert session.status == "closed"

        fresh_session_manager.delete_session(sid)
        with pytest.raises(SessionNotFound):
            fresh_session_manager.get_session(sid)

    def test_update_state(self, fresh_session_manager):
        """测试更新对话状态"""
        sid = fresh_session_manager.create_session()
        state = fresh_session_manager.update_state(sid, intent="report", turn_count=1)
        assert state.intent == "report"
        assert state.turn_count == 1

    def test_list_sessions(self, fresh_session_manager):
        """测试列出会话"""
        s1 = fresh_session_manager.create_session()
        s2 = fresh_session_manager.create_session()
        s3 = fresh_session_manager.create_session()

        active_sessions = fresh_session_manager.list_sessions("active")
        assert len(active_sessions) >= 3

        fresh_session_manager.close_session(s1)
        active_sessions = fresh_session_manager.list_sessions("active")
        assert len(active_sessions) >= 2


# ═══════════════════════════════════════════════════════════════
#  DialogueTracker 测试
# ═══════════════════════════════════════════════════════════════

class TestDialogueTracker:
    """对话状态跟踪器测试"""

    def test_detect_intent_report(self, tracker):
        """测试报告意图识别"""
        state = DialogueState()
        intent = tracker.detect_intent("帮我生成6月使用报告", state)
        assert intent == INTENT_REPORT

    def test_detect_intent_memory(self, tracker):
        """测试记忆查询意图识别"""
        state = DialogueState()
        intent = tracker.detect_intent("你还记得我之前说的吗", state)
        assert intent == INTENT_MEMORY_QUERY

    def test_detect_intent_general(self, tracker):
        """测试一般意图识别"""
        state = DialogueState()
        intent = tracker.detect_intent("扫地机器人怎么选购", state)
        assert intent == INTENT_GENERAL

    def test_extract_entities_city(self, tracker):
        """测试城市实体提取"""
        entities = tracker.extract_entities("我在深圳使用扫地机器人")
        assert "city" in entities
        assert "深圳" in entities["city"]

    def test_extract_entities_month(self, tracker):
        """测试月份实体提取"""
        entities = tracker.extract_entities("帮我生成6月使用报告")
        assert "month" in entities
        assert "2026-06" in entities["month"]

    def test_extract_entities_user_id(self, tracker):
        """测试用户ID实体提取"""
        entities = tracker.extract_entities("用户ID为1001")
        assert "user_id" in entities
        assert entities["user_id"] == "1001"

    def test_fill_slots(self, tracker):
        """测试槽位填充"""
        state = DialogueState()
        entities = {"city": "深圳", "month": "2026-06"}
        state = tracker.fill_slots(state, entities)
        assert state.slot_values["city"] == "深圳"
        assert state.slot_values["month"] == "2026-06"

    def test_check_missing_slots(self, tracker):
        """测试缺失槽位检查"""
        state = DialogueState()
        state.intent = INTENT_REPORT
        state.slot_values = {"city": "深圳"}
        missing = tracker.check_missing_slots(state)
        assert "user_id" in missing
        assert "month" in missing

    def test_update_state_full_flow(self, tracker):
        """测试完整状态更新流程"""
        state = DialogueState()
        state = tracker.update_state("帮我生成6月使用报告，用户ID是1001", state)
        assert state.intent == INTENT_REPORT
        assert state.slot_values.get("month") == "2026-06"
        assert state.slot_values.get("user_id") == "1001"
        assert state.pending_clarification is None  # 所有槽位已填充

    def test_update_state_needs_clarification(self, tracker):
        """测试需要澄清的状态更新"""
        state = DialogueState()
        state = tracker.update_state("帮我生成使用报告", state)
        assert state.intent == INTENT_REPORT
        assert state.pending_clarification is not None
        assert state.requires_follow_up is True

    def test_get_next_action_clarify(self, tracker):
        """测试下一步动作：澄清"""
        state = DialogueState()
        state.pending_clarification = "请提供用户ID"
        action = tracker.get_next_action(state)
        assert action["action"] == "clarify"
        assert action["requires_tool"] is False

    def test_get_next_action_report(self, tracker):
        """测试下一步动作：生成报告"""
        state = DialogueState()
        state.intent = INTENT_REPORT
        state.pending_clarification = None
        action = tracker.get_next_action(state)
        assert action["action"] == "generate_report"
        assert action["requires_tool"] is True


# ═══════════════════════════════════════════════════════════════
#  IntentParser 测试
# ═══════════════════════════════════════════════════════════════

class TestIntentParser:
    """指令解析器测试"""

    def test_parse_search_command(self, parser):
        """测试搜索命令解析"""
        result = parser.parse("搜索扫地机器人的选购指南")
        assert result["command"] == "search"
        assert "选购指南" in result["subject"]

    def test_parse_generate_command(self, parser):
        """测试生成命令解析"""
        result = parser.parse("生成6月使用报告")
        assert result["command"] == "generate"

    def test_parse_conditional(self, parser):
        """测试条件指令解析"""
        result = parser.parse("如果下雨就不要扫地")
        assert result["is_conditional"] is True
        assert len(result["conditions"]) > 0

    def test_extract_modifiers_time(self, parser):
        """测试时间修饰词提取"""
        result = parser.parse("2026年6月的使用报告")
        assert "time" in result["modifiers"]

    def test_extract_modifiers_scope(self, parser):
        """测试范围修饰词提取"""
        result = parser.parse("所有扫地机器人的故障")
        assert result["modifiers"].get("scope") == "all"


# ═══════════════════════════════════════════════════════════════
#  Session 模型测试
# ═══════════════════════════════════════════════════════════════

class TestSession:
    """Session 数据模型测试"""

    def test_add_message(self):
        """测试添加消息"""
        session = Session(session_id="test123")
        session.add_message("user", "你好")
        session.add_message("assistant", "您好！")
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.state.turn_count == 2

    def test_get_recent_messages(self):
        """测试获取最近消息"""
        session = Session(session_id="test123")
        for i in range(5):
            session.add_message("user", f"消息{i}")

        recent = session.get_recent_messages(3)
        assert len(recent) == 3
        assert recent[-1].content == "消息4"

    def test_get_message_history(self):
        """测试获取消息历史（LangChain 格式）"""
        session = Session(session_id="test123")
        session.add_message("user", "你好")
        session.add_message("assistant", "您好！")

        history = session.get_message_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_is_expired(self):
        """测试过期检测"""
        session = Session(session_id="test123")
        session.updated_at = datetime.now() - timedelta(hours=2)
        assert session.is_expired(3600) is True  # 1小时超时
        assert session.is_expired(7200) is False  # 2小时超时

    def test_to_langchain_messages(self):
        """测试转换为 LangChain 格式"""
        session = Session(session_id="test123")
        session.add_message("user", "问题1")
        session.add_message("assistant", "回答1")
        session.add_message("user", "问题2")

        messages = session.to_langchain_messages()
        assert len(messages) == 3
        assert all("role" in m and "content" in m for m in messages)
