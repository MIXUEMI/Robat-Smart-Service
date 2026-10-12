"""知识库检索接口。

封装 VectorStore，提供高层检索 API：
- 批量文档入库
- 语义检索（带/不带分数）
- 检索结果格式化
"""

from src.vectordb.loader import DocumentLoader
from src.vectordb.store import VectorStore


class KnowledgeRetriever:
    """知识库高层检索接口。

    Usage:
        retriever = KnowledgeRetriever()
        retriever.ingest_directory("data/kb")
        results = retriever.retrieve("如何清理滤网")
    """

    def __init__(self):
        self.store = VectorStore()
        self.loader = DocumentLoader()

    def ingest_directory(self, directory_path: str = "data/kb", force: bool = False) -> int:
        """从目录加载文档并写入向量库。

        Args:
            directory_path: 文档目录路径
            force: 是否强制重新入库（即使 collection 已存在）

        Returns:
            int: 入库的文档数量
        """
        if self.store.collection_exists() and not force:
            stats = self.store.get_collection_stats()
            return stats["count"]

        chunks = self.loader.load_and_split(directory_path)
        count = self.store.ingest_documents(chunks)
        return count

    def retrieve(self, query: str, top_k: int | None = None) -> list:
        """语义检索相关文档。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            list[Document]: 相关文档列表
        """
        return self.store.similarity_search(query, k=top_k)

    def retrieve_with_scores(self, query: str, top_k: int | None = None) -> list:
        """语义检索，带相似度分数。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            list[tuple[Document, float]]: (文档, 分数) 列表
        """
        return self.store.similarity_search_with_score(query, k=top_k)

    def format_results(self, docs: list) -> str:
        """将检索结果格式化为可读文本。

        Args:
            docs: Document 对象列表

        Returns:
            str: 格式化的文本，每条含来源和内容
        """
        if not docs:
            return ""

        parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知来源")
            parts.append(f"[来源{i + 1}: {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)
