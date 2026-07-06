"""
pytest 配置和共享 fixtures
"""
import pytest
from context.session_manager import SessionManager
from context.tracker import DialogueTracker
from context.parser import IntentParser


@pytest.fixture
def fresh_session_manager():
    """提供一个全新的 SessionManager 实例（重置单例）"""
    # 重置单例
    SessionManager._instance = None
    manager = SessionManager()
    yield manager
    # 清理
    SessionManager._instance = None


@pytest.fixture
def tracker():
    """提供 DialogueTracker 实例"""
    return DialogueTracker()


@pytest.fixture
def parser():
    """提供 IntentParser 实例"""
    return IntentParser()
