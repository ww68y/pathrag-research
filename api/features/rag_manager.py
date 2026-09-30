"""
PathRAG Manager - Centralized management of PathRAG instances
"""

import os
import logging
from functools import partial
from PathRAG import PathRAG
from PathRAG.llm import openai_complete, siliconcloud_embedding
from PathRAG.prompt import PROMPTS
from PathRAG.utils import EmbeddingFunc

logger = logging.getLogger("PathRAG")

# Setup a working directory for PathRAG.
WORKING_DIR = os.getenv("WORKING_DIR", os.path.join(os.getcwd(), "data"))
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Global PathRAG instance
_rag_instance = None


def _build_rag_instance():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY is not set. LLM calls will fail until it is configured.")

    PROMPTS["process_tickers"] = ["-", "\\", "|", "/"]

    return PathRAG(
        working_dir=WORKING_DIR,
        llm_model_func=openai_complete,
        llm_model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
        llm_model_kwargs={
            "api_key": api_key,
            "base_url": os.getenv("DEEPSEEK_API_BASE", "https://api.siliconflow.cn/v1"),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("EMBEDDING_MAX_TOKENS", "8192")),
            func=partial(
                siliconcloud_embedding,
                model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
                base_url=os.getenv(
                    "EMBEDDING_API_BASE",
                    "https://api.siliconflow.cn/v1/embeddings",
                ),
                api_key=api_key,
                max_token_size=int(os.getenv("EMBEDDING_MAX_TOKENS", "8192")),
            ),
        ),
        embedding_func_max_async=int(os.getenv("EMBEDDING_ASYNC", "1")),
        llm_model_max_async=int(os.getenv("LLM_ASYNC", "1")),
        chunk_token_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap_token_size=int(os.getenv("CHUNK_OVERLAP", "100")),
        entity_extract_max_gleaning=int(os.getenv("ENTITY_EXTRACT_MAX_GLEANING", "0")),
    )

def get_rag_instance():
    """
    Get the current PathRAG instance, initializing it if necessary.
    """
    global _rag_instance
    
    if _rag_instance is None:
        logger.info("Initializing PathRAG instance...")
        _rag_instance = _build_rag_instance()
        logger.info("PathRAG instance initialized successfully")
    
    return _rag_instance

def reload_rag_instance():
    """
    Reload the PathRAG instance to recognize new data files.
    """
    global _rag_instance
    
    logger.info("Reloading PathRAG instance...")
    
    # Create a new instance
    _rag_instance = _build_rag_instance()
    
    logger.info("PathRAG instance reloaded successfully")
    return _rag_instance

# Initialize the instance on module import
rag = get_rag_instance()
