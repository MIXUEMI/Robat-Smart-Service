"""对话历史管理模块。

基于 LangChain InMemoryChatMessageHistory，
支持多会话隔离、最近 N 轮保留和手动清除。
"""

from langchain_community.chat_message_histories import ChatMessageHistory


class ConversationManager:
    """会话对话历史管理器。"""

    def __init__(self, max_history_per_session: int = 20):
        self._store: dict[str, ChatMessageHistory] = {}
        self._max_history = max_history_per_session

    def get_history(self, session_id: str) -> ChatMessageHistory:
        """获取或创建会话历史。"""
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def get_formatted_messages(self, session_id: str, system_prompt: str) -> list:
        """获取格式化的消息列表，用于传入 Agent。

        自动在开头插入 system message，然后拼接历史消息。
        """
        history = self.get_history(session_id)
        messages = [("system", system_prompt)]

        for msg in history.messages[-self._max_history:]:
            role = "human" if msg.type == "human" else "ai"
            messages.append((role, msg.content))

        return messages

    def add_user_message(self, session_id: str, content: str) -> None:
        """添加用户消息到历史。"""
        history = self.get_history(session_id)
        history.add_user_message(content)

    def add_ai_message(self, session_id: str, content: str) -> None:
        """添加 AI 回复到历史。"""
        history = self.get_history(session_id)
        history.add_ai_message(content)

    def clear(self, session_id: str) -> None:
        """清除指定会话的历史。"""
        if session_id in self._store:
            del self._store[session_id]

    def get_session_count(self) -> int:
        """获取当前活跃会话数。"""
        return len(self._store)

    def get_message_count(self, session_id: str) -> int:
        """获取指定会话的消息数。"""
        history = self.get_history(session_id)
        return len(history.messages)
