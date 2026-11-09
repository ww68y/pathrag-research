import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load the project-local environment file before importing application routes.
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pathrag")

if not os.getenv("SECRET_KEY"):
    logger.warning("SECRET_KEY is not set. Configure it in .env before using authentication.")

from models.database import create_tables
from api.auth.routes import router as auth_router
from api.features.users.routes import router as users_router
from api.features.chats.routes import router as chats_router
from api.features.documents.routes import router as documents_router
from api.features.knowledge_graph.routes import router as knowledge_graph_router
from api.features.innovation.routes import router as innovation_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting PathRAG API...")
    create_tables()

    for variable, default in (
        ("UPLOAD_DIR", "./uploads"),
        ("WORKING_DIR", "./data"),
    ):
        directory = os.getenv(variable, default)
        os.makedirs(directory, exist_ok=True)
        logger.info("%s directory: %s", variable, directory)

    yield
    logger.info("PathRAG API stopped.")


app = FastAPI(
    title="PathRAG API",
    description="API for document management, chat, and knowledge-graph operations.",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
if "*" in cors_origins:
    logger.warning("CORS_ORIGINS includes '*'. Restrict origins before deployment.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(chats_router)
app.include_router(documents_router)
app.include_router(knowledge_graph_router)
app.include_router(innovation_router)


@app.get("/")
async def root():
    return {
        "message": "Welcome to PathRAG API",
        "docs": "/docs",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() in ("true", "1", "t"),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
