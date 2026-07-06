"""
会话管理器 — 多轮对话上下文管理
负责会话的创建、存储、检索和过期清理
"""
import uuid
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, List

from core.config import settings
from core.exceptions import SessionNotFound, SessionExpired
from context.schemas import Session, Message, DialogueState
from utils.logger_handler import logger


class SessionManager:
    """
    会话管理器（单例模式）

    内存中维护活跃会话，支持：
    - 创建新会话
    - 按 session_id 检索会话
    - 消息追加
    - 自动过期清理
    - 对话状态跟踪
    """

    _instance: Optional["SessionManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SessionManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # 会话存储: {session_id: Session}
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()
        self._session_timeout = settings.session_timeout

        # 启动后台清理线程
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self._cleanup_thread.start()
        logger.info("[SessionManager] 会话管理器初始化完成")

    # ── 会话生命周期 ────────────────────────────────────

    def create_session(self, metadata: Dict = None) -> str:
        """
        创建新会话

        :param metadata: 可选的会话元数据
        :return: 新会话 ID
        """
        session_id = str(uuid.uuid4())[:8]
        session = Session(
            session_id=session_id,
            metadata=metadata or {},
        )
        with self._lock:
            self._sessions[session_id] = session

        logger.info(f"[SessionManager] 创建新会话 [{session_id}]")
        return session_id

    def get_session(self, session_id: str) -> Session:
        """
        获取会话，检查是否过期

        :param session_id: 会话 ID
        :return: Session 对象
        :raises SessionNotFound: 会话不存在
        :raises SessionExpired: 会话已过期
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise SessionNotFound(f"会话 [{session_id}] 不存在")

            if session.is_expired(self._session_timeout):
                session.status = "expired"
                raise SessionExpired(f"会话 [{session_id}] 已过期")

            return session

    def add_message(self, session_id: str, role: str, content: str,
                    metadata: Dict = None) -> Session:
        """
        向会话添加消息

        :param session_id: 会话 ID
        :param role: 消息角色
        :param content: 消息内容
        :param metadata: 额外元数据
        :return: 更新后的 Session
        """
        session = self.get_session(session_id)
        session.add_message(role, content, metadata)
        logger.debug(f"[SessionManager] [{session_id}] 添加 {role} 消息: {content[:60]}...")
        return session

    def close_session(self, session_id: str) -> None:
        """关闭会话"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.status = "closed"
                logger.info(f"[SessionManager] 关闭会话 [{session_id}]")

    def delete_session(self, session_id: str) -> None:
        """删除会话"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"[SessionManager] 删除会话 [{session_id}]")

    # ── 状态管理 ────────────────────────────────────────

    def update_state(self, session_id: str, **kwargs) -> DialogueState:
        """
        更新对话状态

        :param session_id: 会话 ID
        :param kwargs: 要更新的状态字段
        :return: 更新后的 DialogueState
        """
        session = self.get_session(session_id)
        for key, value in kwargs.items():
            if hasattr(session.state, key):
                setattr(session.state, key, value)
        session.updated_at = datetime.now()
        return session.state

    def get_recent_context(self, session_id: str, n: int = None) -> List[Dict[str, str]]:
        """
        获取最近的消息上下文

        :param session_id: 会话 ID
        :param n: 最近 n 条消息（默认使用配置值）
        :return: LangChain 格式的消息列表
        """
        if n is None:
            n = settings.max_context_messages
        session = self.get_session(session_id)
        return session.get_message_history()[-n:] if len(session.messages) > n else session.get_message_history()

    # ── 查询与统计 ──────────────────────────────────────

    def list_sessions(self, status: str = "active") -> List[Session]:
        """列出指定状态的会话"""
        with self._lock:
            return [s for s in self._sessions.values() if s.status == status]

    def get_session_count(self) -> int:
        """获取活跃会话数"""
        with self._lock:
            return len(self._sessions)

    def _cleanup_expired(self):
        """后台线程：定期清理过期会话"""
        while True:
            time.sleep(300)  # 每 5 分钟检查一次
            with self._lock:
                expired = [
                    sid for sid, session in self._sessions.items()
                    if session.is_expired(self._session_timeout)
                ]
                for sid in expired:
                    self._sessions[sid].status = "expired"
                    del self._sessions[sid]
                    logger.info(f"[SessionManager] 自动清理过期会话 [{sid}]")


# 全局单例
session_manager = SessionManager()
