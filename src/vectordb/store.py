"""ChromaDB 向量存储模块。

封装 ChromaDB PersistentClient，提供 collection 管理、
文档入库和语义检索功能。
"""

import os

from langchain_chroma import Chroma

from configs.settings import settings
from src.embedding.embeddings import get_embedding_model


class VectorStore:
    """ChromaDB 向量存储封装。

    Usage:
        store = VectorStore()
        store.ingest_documents(chunks)
        results = store.similarity_search("如何清理滤网", k=4)
    """

    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embedding = get_embedding_model()

        os.makedirs(self.persist_dir, exist_ok=True)

    def _get_or_create_vectorstore(self) -> Chroma:
        """获取或创建 Chroma vectorstore 实例（懒加载）。"""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=self.persist_dir,
        )

    def collection_exists(self) -> bool:
        """检查 collection 是否已存在且非空。"""
        try:
            vs = self._get_or_create_vectorstore()
            coll = vs._collection
            return coll.count() > 0
        except Exception:
            return False

    def ingest_documents(self, documents: list) -> int:
        """将文档列表写入向量库。

        Args:
            documents: LangChain Document 对象列表

        Returns:
            int: 入库的文档数
        """
        if not documents:
            return 0

        # Delete existing collection first to avoid conflicts
        try:
            vs = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=self.persist_dir,
            )
            vs.delete_collection()
        except Exception:
            pass

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )

        return len(documents)

    def similarity_search(self, query: str, k: int | None = None) -> list:
        """语义相似度检索。

        Args:
            query: 搜索查询文本
            k: 返回的文档数量，默认使用 settings.RETRIEVAL_TOP_K

        Returns:
            list[Document]: 最相关的文档列表
        """
        if k is None:
            k = settings.RETRIEVAL_TOP_K

        vectorstore = self._get_or_create_vectorstore()
        return vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int | None = None) -> list:
        """语义检索，附带相似度分数。

        Args:
            query: 搜索查询文本
            k: 返回数量

        Returns:
            list[tuple[Document, float]]: (文档, 相似度分数) 列表
        """
        if k is None:
            k = settings.RETRIEVAL_TOP_K

        vectorstore = self._get_or_create_vectorstore()
        return vectorstore.similarity_search_with_relevance_scores(query, k=k)

    def delete_collection(self) -> None:
        """删除当前 collection。"""
        try:
            vs = self._get_or_create_vectorstore()
            vs.delete_collection()
        except Exception:
            pass

    def get_collection_stats(self) -> dict:
        """获取 collection 统计信息。"""
        try:
            vs = self._get_or_create_vectorstore()
            coll = vs._collection
            return {
                "name": self.collection_name,
                "count": coll.count(),
                "exists": True,
            }
        except Exception:
            return {"name": self.collection_name, "count": 0, "exists": False}
