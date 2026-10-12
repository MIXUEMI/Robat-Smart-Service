"""ReAct Agent 引擎。

基于 LangGraph create_react_agent，封装 Agent 创建、同步/异步执行和流式输出。
"""

from langgraph.prebuilt import create_react_agent

from configs.settings import settings
from src.agent.memory import ConversationManager
from src.agent.prompt import get_system_prompt_for_scenario
from src.llm.chat_model import get_chat_model
from src.tools import ALL_TOOLS

# 全局实例（延迟初始化）
_agent_instance = None
_memory_manager = None


def get_memory_manager() -> ConversationManager:
    """获取全局 ConversationManager 实例。"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = ConversationManager()
    return _memory_manager


def get_agent():
    """获取全局 ReAct Agent 实例（单例模式）。"""
    global _agent_instance
    if _agent_instance is None:
        llm = get_chat_model()
        _agent_instance = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
        )
    return _agent_instance


class CustomerServiceAgent:
    """扫地机器人智能客服 Agent 封装。"""

    def __init__(self):
        self._agent = None
        self._memory = get_memory_manager()

    @property
    def agent(self):
        if self._agent is None:
            self._agent = get_agent()
        return self._agent

    def chat(self, query: str, session_id: str = "default", scenario: str = "default") -> dict:
        """同步对话 — 返回完整结果。"""
        system_prompt = get_system_prompt_for_scenario(scenario)
        self._memory.add_user_message(session_id, query)

        messages = self._memory.get_formatted_messages(session_id, system_prompt)

        config = {"recursion_limit": settings.AGENT_MAX_ITERATIONS * 2}
        result = self.agent.invoke({"messages": messages}, config=config)

        answer = result["messages"][-1].content
        self._memory.add_ai_message(session_id, answer)

        return {"answer": answer, "session_id": session_id}

    async def astream_chat(self, query: str, session_id: str = "default", scenario: str = "default"):
        """异步流式对话 — 逐事件返回。"""
        system_prompt = get_system_prompt_for_scenario(scenario)
        self._memory.add_user_message(session_id, query)

        messages = self._memory.get_formatted_messages(session_id, system_prompt)

        config = {"recursion_limit": settings.AGENT_MAX_ITERATIONS * 2}
        full_answer = ""

        async for event in self.agent.astream_events(
            {"messages": messages},
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    full_answer += chunk.content
                    yield {"type": "token", "data": chunk.content}

            elif kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                yield {"type": "tool_start", "data": tool_name}

            elif kind == "on_tool_end":
                tool_name = event.get("name", "unknown")
                yield {"type": "tool_end", "data": tool_name}

        if full_answer:
            self._memory.add_ai_message(session_id, full_answer)

        yield {"type": "done", "data": ""}

    def clear_session(self, session_id: str) -> None:
        """清除指定会话的历史。"""
        self._memory.clear(session_id)
