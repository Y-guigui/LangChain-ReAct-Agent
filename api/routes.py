"""
API 路由层 — 聊天和健康检查端点（支持多轮对话上下文管理）
"""
import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from agent.react_agent import ReactAgent
from api.schemas import ChatRequest, ChatResponse, HealthResponse, ErrorResponse
from context.session_manager import session_manager
from context.tracker import dialogue_tracker
from context.parser import intent_parser
from core.config import settings
from core.exceptions import SessionNotFound, SessionExpired
from utils.logger_handler import logger

router = APIRouter()

# ── 全局 Agent 单例（懒加载） ──────────────────────────────────
_agent: ReactAgent | None = None


def get_agent() -> ReactAgent:
    """获取 Agent 单例，首次访问时初始化"""
    global _agent
    if _agent is None:
        logger.info("[FastAPI] 初始化 ReAct Agent 单例...")
        _agent = ReactAgent()
    return _agent


# ── 健康检查 ──────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """服务健康检查"""
    return HealthResponse(
        status="ok",
        service="智扫通 · 智能客服",
        version="2.0.0",
        framework="FastAPI",
    )


# ── 辅助函数：处理对话流程 ────────────────────────────────────

def _process_chat(query: str, session_id: str | None) -> tuple[str, list[dict]]:
    """
    处理对话流程：创建/获取会话，跟踪状态，构建上下文

    :param query: 用户输入
    :param session_id: 会话 ID（可选）
    :return: (session_id, messages_context)
    """
    # 创建或获取会话
    if not session_id:
        session_id = session_manager.create_session()
        logger.info(f"[API] 创建新会话 [{session_id}]")
    else:
        try:
            session = session_manager.get_session(session_id)
            logger.info(f"[API] 恢复会话 [{session_id}] (turn={session.state.turn_count})")
        except (SessionNotFound, SessionExpired):
            # 会话不存在或已过期，创建新会话
            session_id = session_manager.create_session()
            logger.info(f"[API] 会话失效，创建新会话 [{session_id}]")

    # 添加用户消息
    session_manager.add_message(session_id, "user", query)

    # 更新对话状态（意图识别、实体提取、槽位填充）
    session = session_manager.get_session(session_id)
    session.state = dialogue_tracker.update_state(query, session.state)

    # 构建消息上下文（供 Agent 使用）
    messages = session_manager.get_recent_context(
        session_id, n=settings.max_context_messages
    )

    return session_id, messages


def _save_response(session_id: str, content: str) -> None:
    """保存助手回复到会话"""
    session_manager.add_message(session_id, "assistant", content)


# ── 同步聊天（非流式）─────────────────────────────────────────

@router.post("/chat/sync", response_model=ChatResponse, tags=["聊天"])
async def chat_sync(request: ChatRequest):
    """
    同步聊天接口：等待 Agent 完整回复后一次性返回。
    支持多轮对话：通过 session_id 关联上下文。
    """
    logger.info(f"[API] 同步聊天请求: {request.query[:80]}...")
    try:
        session_id, messages = _process_chat(request.query, request.session_id)

        agent = get_agent()
        full_response: list[str] = []
        for chunk in agent.execute_stream(request.query, messages=messages, session_id=session_id):
            full_response.append(chunk)

        content = "".join(full_response)
        _save_response(session_id, content)

        return ChatResponse(
            role="assistant",
            content=content,
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"[API] 同步聊天出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ── SSE 流式聊天 ─────────────────────────────────────────────

async def _sse_event_generator(
    query: str, session_id: str, messages: list[dict]
) -> AsyncGenerator[str, None]:
    """
    SSE (Server-Sent Events) 事件生成器。
    将 Agent 流式输出的每个 chunk 封装为 SSE data 帧。
    """
    try:
        agent = get_agent()
        full_content = ""

        # 发送会话 ID（作为普通 data 帧，便于前端解析）
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"

        for chunk in agent.execute_stream(query, messages=messages, session_id=session_id):
            if chunk:
                full_content += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"

        # 保存完整回复到会话
        _save_response(session_id, full_content)

        # 发送结束信号（作为普通 data 帧，前端通过 reader done 感知结束）
        yield f"data: {json.dumps({'status': 'completed'})}\n\n"

    except Exception as e:
        logger.error(f"[SSE] 流式输出异常: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@router.post("/chat/stream", tags=["聊天"])
async def chat_stream(request: ChatRequest):
    """
    SSE 流式聊天接口：逐字实时推送 Agent 回复。
    支持多轮对话：通过 session_id 关联上下文。

    前端使用 EventSource 或 fetch + ReadableStream 消费。
    """
    try:
        session_id, messages = _process_chat(request.query, request.session_id)
    except Exception as e:
        logger.error(f"[API] 会话处理出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"[API] SSE 流式聊天请求 [{session_id}]: {request.query[:80]}...")

    return StreamingResponse(
        _sse_event_generator(request.query, session_id, messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
        },
    )


# ── WebSocket 聊天 ────────────────────────────────────────────

@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """
    WebSocket 聊天端点：支持双向通信和流式输出。
    支持多轮对话：首次连接自动创建会话，后续消息自动关联上下文。

    协议：
    - 客户端发送 JSON { "query": "...", "session_id": "..." }
    - 服务端逐 chunk 推送回复
    - 最后发送 { "type": "done" } 标记结束
    """
    await websocket.accept()
    session_id = None

    try:
        while True:
            # 接收客户端消息
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                query = data.get("query", "").strip()
                client_session_id = data.get("session_id")
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "error": "消息格式非法，需要 JSON"})
                continue

            if not query:
                await websocket.send_json({"type": "error", "error": "query 不能为空"})
                continue

            # 处理会话（首次连接创建新会话，后续复用）
            try:
                if client_session_id:
                    session_id = client_session_id
                    session_manager.get_session(session_id)
                else:
                    session_id = session_manager.create_session()
                    await websocket.send_json({"type": "session", "session_id": session_id})

                session_id, messages = _process_chat(query, session_id)
            except (SessionNotFound, SessionExpired):
                session_id = session_manager.create_session()
                await websocket.send_json({"type": "session", "session_id": session_id})
                session_id, messages = _process_chat(query, session_id)
            except Exception as e:
                await websocket.send_json({"type": "error", "error": f"会话处理失败: {str(e)}"})
                continue

            logger.info(f"[WebSocket] [{session_id}] 收到: {query[:80]}...")

            # 流式推送 Agent 回复
            try:
                agent = get_agent()
                full_content = ""
                for chunk in agent.execute_stream(query, messages=messages, session_id=session_id):
                    if chunk:
                        full_content += chunk
                        await websocket.send_json({"type": "chunk", "content": chunk})

                # 保存回复
                _save_response(session_id, full_content)
                await websocket.send_json({"type": "done"})

            except Exception as e:
                logger.error(f"[WebSocket] Agent 执行异常: {str(e)}")
                await websocket.send_json({"type": "error", "error": str(e)})

    except WebSocketDisconnect:
        logger.info(f"[WebSocket] 连接断开 [{session_id}]")
    except Exception as e:
        logger.error(f"[WebSocket] 未预期异常 [{session_id}]: {str(e)}")
        try:
            await websocket.close()
        except Exception:
            pass


# ── 会话管理接口 ─────────────────────────────────────────────

@router.post("/session", tags=["会话管理"])
async def create_session():
    """创建新会话"""
    session_id = session_manager.create_session()
    return {"session_id": session_id, "status": "created"}


@router.get("/session/{session_id}", tags=["会话管理"])
async def get_session_info(session_id: str):
    """获取会话信息"""
    try:
        session = session_manager.get_session(session_id)
        return {
            "session_id": session.session_id,
            "status": session.status,
            "turn_count": session.state.turn_count,
            "message_count": len(session.messages),
            "intent": session.state.intent,
            "slot_values": session.state.slot_values,
        }
    except (SessionNotFound, SessionExpired) as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/session/{session_id}", tags=["会话管理"])
async def delete_session(session_id: str):
    """删除会话"""
    session_manager.delete_session(session_id)
    return {"session_id": session_id, "status": "deleted"}
