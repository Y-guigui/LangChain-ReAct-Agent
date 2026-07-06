"""
向量数据库服务类
为整个工程提供统一的向量数据库服务
"""

from langchain_chroma import Chroma
import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import txt_loader, pdf_loader, md_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"])
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )
    
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_documents(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False            # md5 没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True     # md5 处理过

                return False            # md5 没处理过

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            if read_path.endswith("md"):
                return md_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue

    def add_single_document(self, file_path: str) -> dict:
        """
        向量化单个文件（用于文件上传）

        :param file_path: 文件的绝对路径
        :return: {success, filename, chunk_count, error}
        """
        filename = os.path.basename(file_path)
        try:
            documents = self._get_file_documents(file_path)
            if not documents:
                return {"success": False, "filename": filename, "error": "文件内容为空"}

            split_documents = self.spliter.split_documents(documents)
            if not split_documents:
                return {"success": False, "filename": filename, "error": "分片后无有效内容"}

            self.vector_store.add_documents(split_documents)
            logger.info(f"[知识库] 已添加: {filename} ({len(split_documents)} 个分片)")
            return {"success": True, "filename": filename, "chunk_count": len(split_documents)}
        except Exception as e:
            logger.error(f"[知识库] 添加失败 {filename}: {str(e)}")
            return {"success": False, "filename": filename, "error": str(e)}

    def remove_document(self, filename: str) -> dict:
        """
        从向量库中删除指定文件的文档

        :param filename: 文件名
        :return: {success, deleted_count}
        """
        try:
            collection = self.vector_store._collection
            results = collection.get(include=["metadatas"])
            ids_to_delete = []

            if results and results.get("ids"):
                for i, meta in enumerate(results.get("metadatas", [])):
                    if meta and meta.get("source", "").endswith(filename):
                        ids_to_delete.append(results["ids"][i])

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"[知识库] 已删除: {filename} ({len(ids_to_delete)} 个分片)")
                return {"success": True, "deleted_count": len(ids_to_delete)}
            else:
                return {"success": True, "deleted_count": 0, "message": "未找到匹配的文档"}
        except Exception as e:
            logger.error(f"[知识库] 删除失败 {filename}: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_stats(self) -> dict:
        """获取向量库统计信息"""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            return {
                "total_chunks": count,
                "collection_name": chroma_conf["collection_name"],
                "persist_directory": chroma_conf["persist_directory"],
            }
        except Exception as e:
            return {"error": str(e)}

    def search(self, query: str, top_k: int = None) -> list[Document]:
        """语义搜索"""
        if top_k is None:
            top_k = chroma_conf["k"]
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        return retriever.invoke(query)

    def _get_file_documents(self, read_path: str):
        """获取文件文档（与 load_documents 内部函数一致）"""
        if read_path.endswith("txt"):
            return txt_loader(read_path)
        if read_path.endswith("pdf"):
            return pdf_loader(read_path)
        if read_path.endswith("md"):
            return md_loader(read_path)
        return []

    def list_documents(self) -> list[dict]:
        """列出向量库中的所有文档文件"""
        try:
            collection = self.vector_store._collection
            results = collection.get(include=["metadatas"])
            files = {}
            if results and results.get("metadatas"):
                for meta in results.get("metadatas", []):
                    if meta and "source" in meta:
                        source = meta["source"]
                        fname = os.path.basename(source)
                        if fname not in files:
                            files[fname] = {"filename": fname, "source": source, "chunk_count": 0}
                        files[fname]["chunk_count"] += 1
            return sorted(files.values(), key=lambda x: x["filename"])
        except Exception as e:
            logger.error(f"[知识库] 列举文档失败: {str(e)}")
            return []

if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_documents()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)