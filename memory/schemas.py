"""记忆系统数据模型"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Memory(BaseModel):
    """单条记忆"""
    memory_id: Optional[str] = Field(default=None, description="记忆唯一ID")
    content: str = Field(..., description="记忆内容（完整描述）")
    category: str = Field(default="fact", description="分类: preference/fact/goal/feedback/personality")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性评分")
    source: str = Field(default="", description="来源（对话ID或会话ID）")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MemoryQuery(BaseModel):
    """记忆查询请求"""
    query: str = Field(..., description="查询内容")
    top_k: int = Field(default=5, ge=1, le=20, description="返回数量")
    category: Optional[str] = Field(default=None, description="按分类过滤")
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0, description="最小重要性")
