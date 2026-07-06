"""
知识库管理 API — 文件上传、列表、删除、重新索引
"""
import os
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf
from utils.file_handler import get_content_md5_hex, check_md5_exists, save_md5_hex
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

router = APIRouter(tags=["知识库管理"])

# ── 全局服务单例 ────────────────────────────────────────────
_vs: VectorStoreService | None = None


def get_vs() -> VectorStoreService:
    global _vs
    if _vs is None:
        _vs = VectorStoreService()
    return _vs


# ── 数据模型 ────────────────────────────────────────────────

class FileInfo(BaseModel):
    filename: str
    size: int
    type: str
    chunk_count: int


class KnowledgeStats(BaseModel):
    file_count: int
    total_chunks: int
    collection_name: str
    files: List[FileInfo]


# ── 辅助函数 ────────────────────────────────────────────────

ALLOWED_EXTENSIONS = set(chroma_conf.get("allow_knowledge_file_type", ["txt", "pdf", "md"]))
KNOWLEDGE_DIR = get_abs_path(chroma_conf.get("data_path", "data/knowledge"))


def _get_file_size(filepath: str) -> int:
    try:
        return os.path.getsize(filepath)
    except OSError:
        return 0


# ── 辅助函数：去掉文件名中的 UUID 后缀，获取原始文件名 ─────────
def _clean_filename(filename: str) -> str:
    """去掉文件名中的 _xxxxxx UUID 后缀，返回原始文件名"""
    import re
    base, ext = os.path.splitext(filename)
    cleaned_base = re.sub(r'_[0-9a-f]{6}$', '', base)
    return cleaned_base + ext


# ── 辅助函数：对文件列表去重（基于原始文件名） ─────────────────
def _deduplicate_files(files: list) -> list:
    """按原始文件名去重，保留 chunk_count 最多的版本"""
    seen = {}
    for f in files:
        clean_name = _clean_filename(f["filename"])
        if clean_name not in seen:
            seen[clean_name] = f
        else:
            if f.get("chunk_count", 0) > seen[clean_name].get("chunk_count", 0):
                seen[clean_name] = f
    result = list(seen.values())
    result.sort(key=lambda x: x["filename"])
    return result


# ── 上传文件 ────────────────────────────────────────────────

@router.post("/knowledge/upload")
async def upload_knowledge_file(file: UploadFile = File(...)):
    """
    上传知识库文件（支持 .txt, .md, .pdf）
    文件将自动进行向量化并存入 Chroma 数据库
    相同 MD5 的文件会被拒绝上传，避免重复
    """
    # 验证文件扩展名
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: .{ext}，支持: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # 读取文件内容
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")

    # 计算 MD5 并查重
    md5_hex = get_content_md5_hex(content)
    md5_path = get_abs_path(chroma_conf.get("md5_hex_store", "md5.txt"))
    if check_md5_exists(md5_hex, md5_path):
        logger.info(f"[知识库] 文件 {file.filename} 已存在（MD5: {md5_hex}），跳过上传")
        return {
            "success": False,
            "filename": file.filename,
            "md5": md5_hex,
            "message": f"文件 {file.filename} 已存在于知识库中，无需重复上传",
        }

    # 确保上传目录存在
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

    # 保存文件（使用原始文件名，重名则覆盖）
    saved_path = os.path.join(KNOWLEDGE_DIR, file.filename)
    try:
        with open(saved_path, "wb") as f:
            f.write(content)
        logger.info(f"[知识库] 文件已保存: {file.filename} ({len(content)} bytes)")
    except Exception as e:
        logger.error(f"[知识库] 文件保存失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # 向量化
    vs = get_vs()
    result = vs.add_single_document(saved_path)

    if not result["success"]:
        # 向量化失败时删除已保存的文件
        try:
            os.remove(saved_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"向量化失败: {result.get('error', '未知错误')}")

    # 记录 MD5
    save_md5_hex(md5_hex, md5_path)

    return {
        "success": True,
        "filename": file.filename,
        "chunk_count": result["chunk_count"],
        "file_size": len(content),
        "md5": md5_hex,
        "message": f"文件 {file.filename} 上传成功，已生成 {result['chunk_count']} 个向量分片",
    }


# ── 获取知识库统计 ──────────────────────────────────────────

@router.get("/knowledge/stats")
async def get_knowledge_stats():
    """
    获取知识库统计信息：文件数、总向量分片数、文件列表
    """
    vs = get_vs()

    # 列出磁盘上的知识库文件
    disk_files = []
    if os.path.isdir(KNOWLEDGE_DIR):
        for root, dirs, filenames in os.walk(KNOWLEDGE_DIR):
            for fname in filenames:
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if ext in ALLOWED_EXTENSIONS:
                    full_path = os.path.join(root, fname)
                    disk_files.append({
                        "filename": fname,
                        "path": full_path,
                        "size": _get_file_size(full_path),
                    })

    # 列出向量库中的文档
    vector_docs = vs.list_documents()

    # 合并：将向量库中的chunk_count匹配到磁盘文件
    vector_map = {d["filename"]: d["chunk_count"] for d in vector_docs}

    files = []
    for f in disk_files:
        files.append({
            "filename": f["filename"],
            "size": f["size"],
            "type": f["filename"].rsplit(".", 1)[-1] if "." in f["filename"] else "unknown",
            "chunk_count": vector_map.get(f["filename"], 0),
        })

    # 去重：按原始文件名去重，保留 chunk_count 最多的版本
    files = _deduplicate_files(files)
    
    # 转换为 FileInfo 对象
    file_infos = [FileInfo(
        filename=f["filename"],
        size=f["size"],
        type=f["type"],
        chunk_count=f["chunk_count"],
    ) for f in files]

    vs_stats = vs.get_stats()

    return KnowledgeStats(
        file_count=len(file_infos),
        total_chunks=vs_stats.get("total_chunks", 0),
        collection_name=vs_stats.get("collection_name", ""),
        files=file_infos,
    )


# ── 删除文件 ────────────────────────────────────────────────

@router.delete("/knowledge/{filename}")
async def delete_knowledge_file(filename: str):
    """
    删除知识库文件（从磁盘和向量库中同时移除）
    """
    vs = get_vs()

    # 从向量库删除
    result = vs.remove_document(filename)

    # 从磁盘删除
    deleted_disk = False
    if os.path.isdir(KNOWLEDGE_DIR):
        for root, dirs, filenames in os.walk(KNOWLEDGE_DIR):
            if filename in filenames:
                try:
                    os.remove(os.path.join(root, filename))
                    deleted_disk = True
                    logger.info(f"[知识库] 磁盘文件已删除: {filename}")
                except OSError as e:
                    logger.error(f"[知识库] 磁盘文件删除失败: {str(e)}")

    return {
        "success": result["success"],
        "filename": filename,
        "deleted_chunks": result.get("deleted_count", 0),
        "deleted_disk": deleted_disk,
        "message": f"已删除 {filename}（向量: {result.get('deleted_count', 0)} 个分片, 磁盘: {'是' if deleted_disk else '否'}）",
    }


# ── 重新索引 ────────────────────────────────────────────────

@router.post("/knowledge/reindex")
async def reindex_knowledge():
    """
    清空向量库，重新从磁盘加载所有知识库文件并向量化
    """
    vs = VectorStoreService()  # 重新初始化

    # 清空向量库
    try:
        vs.vector_store._collection.delete(ids=vs.vector_store._collection.get()["ids"])
        logger.info("[知识库] 向量库已清空，开始重新索引...")
    except Exception as e:
        logger.warning(f"[知识库] 清空向量库警告: {str(e)}")

    # 清空 MD5 记录（否则 load_documents 会跳过已处理文件）
    md5_path = get_abs_path(chroma_conf.get("md5_hex_store", "md5.txt"))
    try:
        open(md5_path, "w", encoding="utf-8").close()
        logger.info("[知识库] MD5 记录已清空")
    except Exception as e:
        logger.warning(f"[知识库] MD5 清空警告: {str(e)}")

    # 重新加载
    vs.load_documents()
    stats = vs.get_stats()

    return {
        "success": True,
        "total_chunks": stats.get("total_chunks", 0),
        "message": f"重新索引完成，共 {stats.get('total_chunks', 0)} 个向量分片",
    }
