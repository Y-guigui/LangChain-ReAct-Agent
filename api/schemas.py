"""
API 请求/响应数据模型
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    """聊天请求"""
    query: str = Field(..., description="用户问题", min_length=1, max_length=5000)
    session_id: Optional[str] = Field(None, description="会话ID，用于多轮对话追踪")


class ChatResponse(BaseModel):
    """聊天响应"""
    role: str = Field(..., description="消息角色")
    content: str = Field(..., description="回复内容")
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    service: str = "智扫通 · 智能客服"
    version: str = "2.0.0"
    framework: str = "FastAPI"


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(None, description="详细错误描述")
