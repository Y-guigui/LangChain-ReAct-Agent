"""
RAG服务类
为整个工程提供统一的RAG服务
"""
import os
import sys

# 动态获取项目根目录并加入 sys.path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model



class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self.__init_chain()

    def __init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain
    
    def retriever_docs(self, query: str)->list[Document]:
        res = self.retriever.invoke(query)
        return res

    def rag_summarize(self, query: str)->str:
        context_docs = self.retriever_docs(query)

        if not context_docs:
            logger.info(f"[RAG] 知识库未检索到相关内容，直接调用 LLM 回答: {query[:50]}...")
            return self.model.invoke(f"请回答以下问题：{query}。如果知识库中未找到相关内容，请直接基于你的知识回答。").content

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
        
        return self.chain.invoke(
            {
                "input": query,
                "context": context
            }
        )

if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))