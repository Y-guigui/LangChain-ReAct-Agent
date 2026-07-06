"""全局常量定义"""

# 会话状态
SESSION_STATUS_ACTIVE = "active"
SESSION_STATUS_EXPIRED = "expired"
SESSION_STATUS_CLOSED = "closed"

# 消息角色
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLE_TOOL = "tool"

# 对话意图类型
INTENT_GENERAL = "general"           # 一般问答
INTENT_REPORT = "report"             # 报告生成
INTENT_MEMORY_QUERY = "memory_query" # 记忆查询
INTENT_KNOWLEDGE = "knowledge"       # 知识库查询
INTENT_CLARIFICATION = "clarification" # 需要澄清

# 工具名称
TOOL_RAG = "rag_summarize"
TOOL_WEATHER = "get_weather"
TOOL_LOCATION = "get_user_location"
TOOL_USER_ID = "get_user_id"
TOOL_MONTH = "get_current_month"
TOOL_FETCH_DATA = "fetch_external_data"
TOOL_FILL_REPORT = "fill_context_for_report"
TOOL_MEMORY = "search_memory"

# RRF 混合搜索参数
RRF_K = 60

# 记忆分类标签
MEMORY_CATEGORIES = [
    "preference",   # 用户偏好
    "fact",         # 事实信息
    "goal",         # 目标/意图
    "feedback",     # 用户反馈
    "personality",  # 性格/习惯
]
