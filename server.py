"""
智扫通 · 智能客服 — FastAPI 服务入口
基于 LangChain ReAct Agent + RAG 检索增强的扫地机器人智能客服系统

启动方式:
    python server.py
    # 或
    uvicorn server:app --host 0.0.0.0 --port 8080 --reload
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.routes import router as api_router
from api.knowledge import router as knowledge_router
from api.evaluation import router as evaluation_router
from utils.logger_handler import logger


# ── 应用生命周期 ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭时的生命周期管理"""
    logger.info("=" * 60)
    logger.info("  智扫通 · 智能客服 v2.0 (FastAPI)")
    logger.info("  LangChain ReAct Agent + RAG + Chroma")
    logger.info("=" * 60)

    # 启动时预初始化 Agent
    try:
        from api.routes import get_agent
        get_agent()
        logger.info("  [OK] ReAct Agent 初始化完成")
    except Exception as e:
        logger.warning(f"  [WARN] Agent 延迟初始化: {e}")

    yield

    logger.info("  服务已关闭")


# ── FastAPI 应用 ─────────────────────────────────────────────

app = FastAPI(
    title="智扫通 · 智能客服",
    description="基于 LangChain ReAct Agent + RAG 检索增强生成的扫地机器人智能客服 API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — 允许前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由
app.include_router(api_router, prefix="/api")
app.include_router(knowledge_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")

# 静态前端页面
static_dir = ROOT / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── 前端页面路由 ─────────────────────────────────────────────

@app.get("/", response_class=FileResponse, tags=["前端"])
async def index():
    """返回前端聊天界面"""
    return static_dir / "index.html"


# ── 直接启动 ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    reload = os.getenv("RELOAD", "true").lower() == "true"

    logger.info(f"启动服务: http://{host}:{port}")
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
