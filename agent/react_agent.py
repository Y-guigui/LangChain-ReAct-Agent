import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_current_time, search_memory, store_memory)
from agent.tools.middleware import log_before_model, report_prompt_switch
from utils.logger_handler import logger


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_current_time, search_memory, store_memory],
            middleware=[log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str, messages: list[dict] = None, session_id: str = None):
        """
        流式执行 Agent，支持多轮对话上下文注入

        :param query: 当前用户查询
        :param messages: 历史消息上下文（LangChain 格式: [{"role": ..., "content": ...}, ...]）
        :param session_id: 会话 ID（用于日志和后续扩展）
        :yield: 每个 chunk 的回复内容
        """
        # 构建消息列表
        if messages and len(messages) > 0:
            # 检查最后一条消息是否是当前用户的 query，避免重复
            last_msg = messages[-1]
            if last_msg.get("role") == "user" and last_msg.get("content") == query:
                # 已经包含当前 query，直接使用
                all_messages = messages
            else:
                # 追加当前 query
                all_messages = messages + [{"role": "user", "content": query}]
        else:
            all_messages = [{"role": "user", "content": query}]

        input_dict = {"messages": all_messages}

        log_prefix = f"[{session_id}] " if session_id else ""
        logger.info(f"[Agent] {log_prefix}执行流式推理，消息数: {len(all_messages)}")

        # 第三个参数 context 就是上下文 runtime 中的信息，用于提示词切换
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    
    pass
