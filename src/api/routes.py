"""FastAPI 路由定义。"""

import json
import logging
import traceback

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from configs.settings import settings
from src.agent.agent import CustomerServiceAgent
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse
from src.middleware.prompt_switch import detect_scenario
from src.vectordb.store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

_agent = CustomerServiceAgent()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查 — 返回服务和依赖状态。"""
    try:
        store = VectorStore()
        stats = store.get_collection_stats()
        vectordb_status = "ready" if stats["count"] > 0 else "empty"
    except Exception:
        vectordb_status = "error"

    return HealthResponse(
        status="running",
        version="1.0.0",
        model=settings.DEEPSEEK_MODEL,
        vectordb_status=vectordb_status,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """同步对话 — 返回完整 Agent 回答。"""
    try:
        scenario = detect_scenario(request.query)
        result = _agent.chat(
            query=request.query,
            session_id=request.session_id,
            scenario=scenario,
        )
        return ChatResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            scenario=scenario,
        )
    except Exception as e:
        logger.error(f"Chat error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE 流式对话 — 逐 token 推送 Agent 回答。"""
    scenario = detect_scenario(request.query)

    async def event_generator():
        try:
            async for event in _agent.astream_chat(
                query=request.query,
                session_id=request.session_id,
                scenario=scenario,
            ):
                event_type = event["type"]
                data = event["data"]

                if event_type == "token":
                    yield {"event": "token", "data": data}
                elif event_type == "tool_start":
                    yield {"event": "tool_start", "data": data}
                elif event_type == "tool_end":
                    yield {"event": "tool_end", "data": data}
                elif event_type == "done":
                    yield {"event": "done", "data": ""}

        except Exception as e:
            logger.error(f"Stream error: {traceback.format_exc()}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
