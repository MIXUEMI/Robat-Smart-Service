"""API 数据模型 (Pydantic)。"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求模型。"""
    session_id: str = Field(..., description="会话唯一标识", min_length=1)
    query: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    user_id: str | None = Field(None, description="用户 ID（可选）")
    device_id: str | None = Field(None, description="关联设备编号（可选）")


class ChatResponse(BaseModel):
    """对话响应模型（非流式）。"""
    answer: str = Field(..., description="Agent 的完整回答")
    session_id: str = Field(..., description="会话 ID")
    scenario: str = Field("default", description="检测到的业务场景")


class HealthResponse(BaseModel):
    """健康检查响应模型。"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    model: str = Field(..., description="当前使用的 LLM 模型")
    vectordb_status: str = Field(..., description="向量库状态：ready / empty / error")
