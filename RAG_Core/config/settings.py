# RAG_Core/config/settings.py (ADD FAQ SECTION)

from typing import Optional

try:
    from pydantic_settings import BaseSettings

    V2 = True
except Exception:
    from pydantic import BaseSettings

    V2 = False


class Settings(BaseSettings):
    """App settings"""

    # ===== Milvus =====
    MILVUS_HOST: str = "milvus"
    MILVUS_PORT: str = "19530"
    DOCUMENT_COLLECTION: str = "document_embeddings"
    FAQ_COLLECTION: str = "faq_embeddings"

    # ===== Ollama / LLM =====
    OLLAMA_URL: str = "http://ollama:11434"
    LLM_MODEL: str = "gpt-oss:20b"
    OLLAMA_BASE_URL: Optional[str] = None

    # ===== Embedding =====
    EMBEDDING_MODEL: str = "keepitreal/vietnamese-sbert"
    EMBEDDING_DIM: int = 768

    # ===== Search / RAG =====
    SIMILARITY_THRESHOLD: float = 0.5
    TOP_K: int = 20
    MAX_ITERATIONS: int = 5

    # ===== FAQ OPTIMIZATION SETTINGS =====
    # Vector search thresholds
    FAQ_VECTOR_THRESHOLD: float = 0.5  # Ngưỡng cho vector search (thấp để cast wide net)
    FAQ_TOP_K: int = 10  # Số lượng FAQ candidates cho reranking

    # Reranking thresholds
    FAQ_RERANK_THRESHOLD: float = 0.6  # Ngưỡng cho reranked results (cao hơn)

    # Reranking weights (tổng = 1.0)
    FAQ_QUESTION_WEIGHT: float = 0.5  # Trọng số cho question-only matching
    FAQ_QA_WEIGHT: float = 0.3  # Trọng số cho question+answer matching
    FAQ_ANSWER_WEIGHT: float = 0.2  # Trọng số cho answer-only matching

    # Bonus settings
    FAQ_CONSISTENCY_BONUS: float = 1.1  # Bonus multiplier khi tất cả variants đều cao
    FAQ_CONSISTENCY_THRESHOLD: float = 0.75  # Ngưỡng để được bonus

    # ===== Document Grader Settings =====
    DOCUMENT_RERANK_THRESHOLD: float = 0.7  # Rerank threshold cho documents

    # ===== Contact =====
    SUPPORT_PHONE: str = "Phòng vận hành 0904540490 - Phòng kinh doanh:0914616081"

    # ===== Optional API ports =====
    DOC_API_PORT: Optional[int] = None
    RAG_API_PORT: Optional[int] = None

    # Config
    if V2:
        model_config = {"env_file": ".env", "extra": "ignore"}
    else:
        class Config:
            env_file = ".env"
            extra = "ignore"


# Singleton settings
settings = Settings()


# Helper function to get FAQ config
def get_faq_config() -> dict:
    """Get FAQ-specific configuration"""
    return {
        "vector_threshold": settings.FAQ_VECTOR_THRESHOLD,
        "rerank_threshold": settings.FAQ_RERANK_THRESHOLD,
        "top_k": settings.FAQ_TOP_K,
        "weights": {
            "question": settings.FAQ_QUESTION_WEIGHT,
            "question_answer": settings.FAQ_QA_WEIGHT,
            "answer": settings.FAQ_ANSWER_WEIGHT
        },
        "consistency_bonus": settings.FAQ_CONSISTENCY_BONUS,
        "consistency_threshold": settings.FAQ_CONSISTENCY_THRESHOLD
    }