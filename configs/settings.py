import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── App ──
    APP_NAME: str = os.getenv("APP_NAME", "RAG Agent Customer Service")
    APP_ENV: str = os.getenv("APP_ENV", "dev")

    # ── LLM ──
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek")

    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # ── Embedding ──
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # ── Retrieval ──
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # ── Agent ──
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))

    # ── Logging ──
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")

    # ── ChromaDB ──
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "robot_vacuum_kb")


settings = Settings()
