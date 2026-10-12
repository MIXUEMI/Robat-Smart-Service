from langchain_openai import ChatOpenAI

from configs.settings import settings


def get_chat_model():
    """
    初始化大模型。

    当前默认使用 DeepSeek 的 OpenAI 兼容接口。
    后续如果要换成通义千问、智谱、Kimi 或 Ollama，
    只需要改这里，不影响其他业务代码。
    """

    if settings.LLM_PROVIDER == "deepseek":
        if not settings.DEEPSEEK_API_KEY:
            raise ValueError("请先在 .env 文件中配置 DEEPSEEK_API_KEY")

        return ChatOpenAI(
            model=settings.DEEPSEEK_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            temperature=0.3,
        )

    raise ValueError(f"暂不支持的模型提供方: {settings.LLM_PROVIDER}")