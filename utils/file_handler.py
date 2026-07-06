"""
文件处理工具
为整个工程提供统一的文件处理
"""

import os,hashlib
from utils.logger_handler import logger

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader

# 获取文件的md5的十六进制字符串
def get_file_md5_hex(filepath:str):         
    if not os.path.exists(filepath):
        logger.error(f"[MD5计算]文件{filepath}不存在")
        return
    if not os.path.isfile(filepath):
        logger.error(f"[MD5计算]路径{filepath}不是文件")
        return

    md5_obj = hashlib.md5()
    chunk_size=4096     # 4KB分片，避免文件过大爆内存
    try:
            with open(filepath, "rb")as f:  # 必须二进制读取
                while chunk:=f.read(chunk_size):
                    md5_obj.update(chunk)

            md5_hex=md5_obj.hexdigest()
            return md5_hex
    except Exception as  e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def get_content_md5_hex(content: bytes) -> str:
    """从文件内容计算 MD5（用于上传前查重）"""
    md5_obj = hashlib.md5()
    md5_obj.update(content)
    return md5_obj.hexdigest()


def check_md5_exists(md5_hex: str, md5_store_path: str) -> bool:
    """检查 MD5 是否已存在于记录中"""
    if not os.path.exists(md5_store_path):
        return False
    try:
        with open(md5_store_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == md5_hex:
                    return True
    except Exception as e:
        logger.error(f"[MD5查重]读取记录失败: {str(e)}")
    return False


def save_md5_hex(md5_hex: str, md5_store_path: str):
    """将 MD5 记录到文件中"""
    try:
        with open(md5_store_path, "a", encoding="utf-8") as f:
            f.write(md5_hex + "\n")
    except Exception as e:
        logger.error(f"[MD5记录]保存失败: {str(e)}")


def listdir_with_allowed_type(path:str,allowed_types:tuple[str]):   # 返回文件夹内的文件列表（允许的文件后缀，支持递归）
    files= []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return tuple(files)

    for root, dirs, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(allowed_types):
                files.append(os.path.join(root, f))

    return tuple(files)

def pdf_loader(filepath:str,passwd=None)->list[Document]:

    return PyPDFLoader(filepath,passwd).load()

def txt_loader(filepath:str)->list[Document]:

    return TextLoader(filepath,encoding="utf-8").load()

def md_loader(filepath:str)->list[Document]:
    """Markdown 文件加载器（使用 TextLoader 加载）"""
    return TextLoader(filepath, encoding="utf-8").load()
