"""自定义异常类"""


class AgentException(Exception):
    """Agent 基础异常"""
    pass


class SessionException(Exception):
    """会话相关异常"""
    pass


class SessionNotFound(SessionException):
    """会话不存在"""
    pass


class SessionExpired(SessionException):
    """会话已过期"""
    pass


class MemoryException(Exception):
    """记忆系统异常"""
    pass


class RAGException(Exception):
    """RAG 系统异常"""
    pass


class ConfigException(Exception):
    """配置异常"""
    pass


class ToolCallException(Exception):
    """工具调用异常"""
    pass
