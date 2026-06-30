"""
模型工厂
为整个工程提供统一的模型工厂
"""

from abc import ABC, abstractmethod
from typing import Optional
import os
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


class BaseModelFactory(ABC):
    """
    模型工厂基类
    """
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    """
    聊天模型工厂
    """
    def generator(self) -> Optional[BaseChatModel]:
        return ChatOpenAI(
            model=os.getenv("MODEL_ID"),
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )


class EmbeddingsFactory(BaseModelFactory):
    """
    嵌入模型工厂
    """
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_NAME"),
            dashscope_api_key=os.getenv("API_KEY"),
        )


chat_model = ChatModelFactory().generator()

embed_model = EmbeddingsFactory().generator()
