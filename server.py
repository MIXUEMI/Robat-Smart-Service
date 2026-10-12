"""FastAPI 服务入口。

启动命令：
    uvicorn server:app --reload --port 8000

访问文档：
    http://localhost:8000/docs
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from configs.settings import settings
from src.api.routes import router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="小扫 · 智能客服 API",
        description="基于 ReAct Agent 的扫地机器人智能客服系统",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.on_event("startup")
    async def startup():
        logger.info("=" * 50)
        logger.info(f"  App: {settings.APP_NAME}")
        logger.info(f"  Model: {settings.DEEPSEEK_MODEL}")
        logger.info(f"  Embedding: {settings.EMBEDDING_MODEL}")
        logger.info(f"  Docs: http://localhost:8000/docs")
        logger.info("=" * 50)

        try:
            from src.embedding.embeddings import get_embedding_model
            get_embedding_model()
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.warning(f"Embedding model load failed: {e}")

        try:
            from src.vectordb.store import VectorStore
            store = VectorStore()
            stats = store.get_collection_stats()
            logger.info(f"VectorDB: collection='{stats['name']}', count={stats['count']}")
        except Exception as e:
            logger.warning(f"VectorDB init failed: {e}")

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Service shutting down.")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
