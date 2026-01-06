# RAG_Core/agents/grader_agent.py (NO FALLBACK VERSION)

from typing import Dict, Any, List
from tools.vector_search import rerank_documents
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class GraderAgent:
    def __init__(self):
        self.name = "GRADER"
        self.reranking_threshold = 0.6

    def process(self, question: str, documents: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Đánh giá chất lượng tài liệu bằng reranking model
        KHÔNG CÓ FALLBACK - nếu rerank fail → raise error
        """
        try:
            if not documents:
                logger.warning("No documents to grade")
                return {
                    "status": "INSUFFICIENT",
                    "qualified_documents": [],
                    "references": [],
                    "next_agent": "NOT_ENOUGH_INFO"
                }

            # Bước 1: Rerank documents (NO FALLBACK)
            logger.info(f"Reranking {len(documents)} documents for question: {question[:50]}...")

            reranked_docs = rerank_documents.invoke({
                "query": question,
                "documents": documents
            })

            if not reranked_docs:
                logger.error("❌ Reranking returned empty results - this should not happen")
                raise RuntimeError("Reranking failed: empty results")

            # Bước 2: Lọc documents theo 2 tiêu chí
            qualified_docs = []
            for doc in reranked_docs:
                rerank_score = doc.get("rerank_score", 0)
                similarity_score = doc.get("similarity_score", 0)

                # Kiểm tra CẢ HAI điểm số
                if (rerank_score >= self.reranking_threshold and
                        similarity_score >= settings.SIMILARITY_THRESHOLD):
                    qualified_docs.append(doc)
                    logger.debug(
                        f"Doc {doc.get('document_id')}: "
                        f"similarity={similarity_score:.3f}, rerank={rerank_score:.3f} ✓"
                    )
                else:
                    logger.debug(
                        f"Doc {doc.get('document_id')}: "
                        f"similarity={similarity_score:.3f}, rerank={rerank_score:.3f} ✗"
                    )

            # Bước 3: Quyết định
            if qualified_docs:
                logger.info(f"Found {len(qualified_docs)} qualified documents")
                return {
                    "status": "SUFFICIENT",
                    "qualified_documents": qualified_docs,
                    "references": [
                        {
                            "document_id": doc.get("document_id"),
                            "type": "DOCUMENT",
                            "description": doc.get("description", "")[:500],
                            "rerank_score": round(doc.get("rerank_score", 0), 4),
                            "similarity_score": round(doc.get("similarity_score", 0), 4)
                        }
                        for doc in qualified_docs
                    ],
                    "next_agent": "GENERATOR"
                }
            else:
                logger.warning("No documents passed grading thresholds")
                return {
                    "status": "INSUFFICIENT",
                    "qualified_documents": [],
                    "references": [],
                    "next_agent": "NOT_ENOUGH_INFO"
                }

        except RuntimeError as e:
            # Reranking errors - propagate up
            logger.error(f"❌ Critical error in grader agent: {e}")
            raise

        except Exception as e:
            # Other errors - also propagate
            logger.error(f"❌ Unexpected error in grader agent: {e}", exc_info=True)
            raise RuntimeError(f"Grader agent failed: {e}") from e