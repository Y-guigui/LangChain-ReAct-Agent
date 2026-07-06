"""
多轮对话上下文数据模型
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Message(BaseModel):
    """单条消息"""
    role: str = Field(..., description="消息角色: user/assistant/system/tool")
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class DialogueState(BaseModel):
    """对话状态"""
    intent: str = Field(default="general", description="当前意图")
    entities: Dict[str, Any] = Field(default_factory=dict, description="已提取的实体")
    slot_values: Dict[str, Any] = Field(default_factory=dict, description="已填充的槽位")
    turn_count: int = Field(default=0, description="当前轮次")
    last_tool: Optional[str] = Field(default=None, description="上一次调用的工具")
    pending_clarification: Optional[str] = Field(default=None, description="待澄清的问题")
    requires_follow_up: bool = Field(default=False, description="是否需要后续追问")


class Session(BaseModel):
    """会话模型"""
    session_id: str = Field(..., description="会话ID")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="最后更新时间")
    status: str = Field(default="active", description="状态: active/expired/closed")
    messages: List[Message] = Field(default_factory=list, description="消息列表")
    state: DialogueState = Field(default_factory=DialogueState, description="对话状态")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="会话元数据")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """添加消息"""
        self.messages.append(
            Message(role=role, content=content, metadata=metadata or {})
        )
        self.updated_at = datetime.now()
        self.state.turn_count += 1

    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """获取最近 n 条消息"""
        return self.messages[-n:] if len(self.messages) > n else self.messages

    def get_message_history(self) -> List[Dict[str, str]]:
        """获取 LangChain 兼容的消息历史格式"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]

    def is_expired(self, timeout_seconds: int = 3600) -> bool:
        """检查会话是否已过期"""
        from datetime import timedelta
        return datetime.now() - self.updated_at > timedelta(seconds=timeout_seconds)

    def to_langchain_messages(self) -> List[Dict[str, str]]:
        """转换为 LangChain 消息格式（用于 Agent 调用）"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]
