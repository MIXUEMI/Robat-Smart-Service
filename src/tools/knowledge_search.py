"""知识库语义检索工具。

Agent 调用此工具搜索产品知识库，获取相关文档片段。
"""

from langchain_core.tools import tool

from src.vectordb.retriever import KnowledgeRetriever


@tool
def search_knowledge_base(query: str) -> str:
    """搜索扫地机器人产品知识库，获取使用手册、故障排查、维护指南等相关文档。

    适用场景：
    - 用户询问产品功能、使用方法、参数规格
    - 用户报告故障、错误代码
    - 用户询问保养和维护方法

    Args:
        query: 自然语言搜索查询，尽量具体描述用户问题

    Returns:
        str: 最相关的知识库文档片段（最多 Top-4），含来源标注
    """
    retriever = KnowledgeRetriever()
    docs = retriever.retrieve(query)

    if not docs:
        return "未在知识库中找到相关信息。建议引导用户联系人工客服或提交工单。"

    return retriever.format_results(docs)
