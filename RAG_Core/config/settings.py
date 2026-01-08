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
    GRADER_STAGE1_THRESHOLD: float = 0.4

    # Stage 2: Semantic filter (medium threshold - refine results)
    GRADER_STAGE2_THRESHOLD: float = 0.55

    # Stage 3: LLM validation (strict threshold - final check)
    GRADER_LLM_THRESHOLD: float = 0.65

    # Control flags - Enable/disable each stage
    GRADER_USE_STAGE2: bool = True  # Bật/tắt Stage 2
    GRADER_USE_LLM: bool = True  # Bật/tắt Stage 3 (LLM)

    # LLM validation limits
    GRADER_MAX_LLM_DOCS: int = 10  # Số documents tối đa cho LLM validate

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
    SIMILARITY_THRESHOLD: float = 0.2
    TOP_K: int = 50
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
    DOCUMENT_RERANK_THRESHOLD: float = 0.6  # Rerank threshold cho documents

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


def get_grader_preset(mode: str = "balanced") -> dict:
    """
    Các preset cấu hình cho grader

    Modes:
    - "fast": Ưu tiên tốc độ, chỉ dùng Stage 1
    - "balanced": Cân bằng tốc độ & độ chính xác (Stage 1 + 2)
    - "accurate": Ưu tiên độ chính xác cao nhất (Full pipeline)
    """
    presets = {
        "fast": {
            "GRADER_STAGE1_THRESHOLD": 0.5,
            "GRADER_USE_STAGE2": False,
            "GRADER_USE_LLM": False
        },
        "balanced": {
            "GRADER_STAGE1_THRESHOLD": 0.45,
            "GRADER_STAGE2_THRESHOLD": 0.55,
            "GRADER_USE_STAGE2": True,
            "GRADER_USE_LLM": False
        },
        "accurate": {
            "GRADER_STAGE1_THRESHOLD": 0.4,
            "GRADER_STAGE2_THRESHOLD": 0.5,
            "GRADER_LLM_THRESHOLD": 0.65,
            "GRADER_USE_STAGE2": True,
            "GRADER_USE_LLM": True,
            "GRADER_MAX_LLM_DOCS": 15
        }
    }

    return presets.get(mode, presets["balanced"])