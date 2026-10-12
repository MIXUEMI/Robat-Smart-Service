"""本地 Embedding 模型加载模块。

使用 BAAI/bge-small-zh-v1.5，512维中文向量模型。
模型首次加载后缓存在模块级变量中，避免重复初始化。
"""

from langchain_huggingface import HuggingFaceEmbeddings

from configs.settings import settings

_embedding_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """获取本地 Embedding 模型（单例模式）。

    Returns:
        HuggingFaceEmbeddings: 兼容 LangChain Embeddings 接口的实例
    """
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embedding_model
