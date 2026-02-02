# RAG_Core/agents/personalization_retriever_agent.py

from typing import Dict, Any, List
from database.personalization_milvus_client import personalization_milvus_client
from models.embedding_model import embedding_model
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class PersonalizationRetrieverAgent:
    """
    Retriever Agent cho personalization_db
    Tìm kiếm documents trong personalization_default collection
    """

    def __init__(self):
        self.name = "PERSONALIZATION_RETRIEVER"

    def process(
            self,
            question: str,
            contextualized_question: str = "",
            is_followup: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Tìm kiếm tài liệu từ personalization_default collection

        Args:
            question: Câu hỏi gốc (for logging)
            contextualized_question: Câu hỏi đã được làm rõ (dùng để search)
            is_followup: Có phải follow-up question không
        """
        try:
            # Quyết định query cho vector search
            if is_followup or contextualized_question:
                search_query = contextualized_question
                logger.info(f"🔍 Using CONTEXTUALIZED QUESTION for search")
                logger.debug(f"Original: {question[:60]}")
                logger.debug(f"Contextualized: {contextualized_question[:100]}")
            else:
                search_query = question
                logger.info(f"🔍 Using ORIGINAL QUESTION for search")

            logger.info(f"📚 Searching PERSONALIZATION docs: {search_query[:100]}...")

            # Encode query
            query_vector = embedding_model.encode_single(search_query)

            # Search trong personalization_default collection
            search_results = personalization_milvus_client.search_documents(
                query_vector,
                top_k=settings.TOP_K
            )

            if not search_results or "error" in str(search_results):
                logger.warning("Personalization vector search failed or returned error")
                return {
                    "status": "ERROR",
                    "documents": [],
                    "next_agent": "NOT_ENOUGH_INFO"
                }

            # Lọc kết quả theo similarity threshold
            relevant_docs = [
                doc for doc in search_results
                if doc.get("similarity_score", 0) > settings.SIMILARITY_THRESHOLD
            ]

            if not relevant_docs:
                logger.info(
                    f"No personalization docs above threshold {settings.SIMILARITY_THRESHOLD}, "
                    f"returning all {len(search_results)} for grader"
                )
                return {
                    "status": "NOT_FOUND",
                    "documents": search_results,
                    "search_query_used": "contextualized" if (is_followup and contextualized_question) else "original",
                    "next_agent": "GRADER"
                }

            logger.info(
                f"✅ Found {len(relevant_docs)} relevant PERSONALIZATION documents "
                f"(searched with {'contextualized question' if is_followup and contextualized_question else 'original question'})"
            )

            return {
                "status": "SUCCESS",
                "documents": relevant_docs,
                "search_query_used": "contextualized" if (is_followup and contextualized_question) else "original",
                "next_agent": "GRADER"
            }

        except Exception as e:
            logger.error(f"❌ Personalization Retriever error: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "documents": [],
                "next_agent": "REPORTER"
            }