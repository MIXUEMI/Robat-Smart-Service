"""文档加载和分块管道。

支持从目录加载 .md / .txt 文件，使用 RecursiveCharacterTextSplitter
进行递归分块，优先在段落和句子边界切分。
"""

import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from configs.settings import settings


class DocumentLoader:
    """从指定目录加载文档并执行分块。

    Usage:
        loader = DocumentLoader()
        docs = loader.load_and_split("data/kb")
        print(f"Loaded {len(docs)} chunks")
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def load_directory(self, directory_path: str) -> list:
        """从目录中加载所有支持的文档文件。

        Args:
            directory_path: 包含 .md / .txt 文件的目录路径

        Returns:
            list[Document]: LangChain Document 对象列表（未分块）
        """
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"目录不存在: {directory_path}")

        loader = DirectoryLoader(
            path=directory_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )

        md_docs = loader.load()

        txt_loader = DirectoryLoader(
            path=directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )

        txt_docs = txt_loader.load()

        all_docs = md_docs + txt_docs

        if not all_docs:
            raise ValueError(f"目录 {directory_path} 中未找到任何 .md 或 .txt 文件")

        return all_docs

    def split_documents(self, documents: list) -> list:
        """将文档列表分块。

        Args:
            documents: LangChain Document 对象列表

        Returns:
            list[Document]: 分块后的 Document 对象列表，每个包含 chunk_id 元数据
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            chunk.metadata["chunk_id"] = f"{Path(source).stem}_{i}"

        return chunks

    def load_and_split(self, directory_path: str) -> list:
        """加载目录中的文档并分块（一步完成）。

        Args:
            directory_path: 文档目录路径

        Returns:
            list[Document]: 分块并附加元数据后的 Document 列表
        """
        docs = self.load_directory(directory_path)
        chunks = self.split_documents(docs)
        return chunks
