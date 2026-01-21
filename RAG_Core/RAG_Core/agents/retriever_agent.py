# RAG_Core/agents/retriever_agent.py - SEARCH WITH CONTEXT SUMMARY

from typing import Dict, Any, List
from models.llm_model import llm_model
from tools.vector_search import search_documents
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RetrieverAgent:
    def __init__(self):
        self.name = "RETRIEVER"
        self.tools = [search_documents]

    def process(
            self,
            question: str,
            context_summary: str = "",  # NEW: Accept context summary
            is_followup: bool = False,  # NEW: Know if it's a follow-up
            **kwargs
    ) -> Dict[str, Any]:
        """
        Tìm kiếm tài liệu liên quan đến câu hỏi

        Args:
            question: Câu hỏi gốc
            context_summary: Ngữ cảnh đã được làm rõ (dùng cho vector search)
            is_followup: Có phải follow-up question không
        """
        try:
            # ================================================================
            # QUYẾT ĐỊNH QUERY CHO VECTOR SEARCH
            # ================================================================

            # Nếu là follow-up và có context summary → dùng context summary
            if is_followup and context_summary:
                search_query = context_summary
                logger.info(f"🔍 Using CONTEXT SUMMARY for vector search (follow-up)")
                logger.debug(f"Context Summary: {context_summary[:200]}...")
            else:
                # Không phải follow-up hoặc không có context → dùng câu hỏi gốc
                search_query = question
                logger.info(f"🔍 Using ORIGINAL QUESTION for vector search")

            # ================================================================
            # VECTOR SEARCH
            # ================================================================

            logger.info(f"📚 Searching documents with query: {search_query[:100]}...")

            # Tìm kiếm tài liệu với query đã quyết định
            search_results = search_documents.invoke({"query": search_query})

            if not search_results or "error" in str(search_results):
                logger.warning("Vector search failed or returned error")
                return {
                    "status": "ERROR",
                    "documents": [],
                    "next_agent": "NOT_ENOUGH_INFO"
                }

            # ================================================================
            # LỌC KẾT QUẢ THEO SIMILARITY THRESHOLD
            # ================================================================

            relevant_docs = [
                doc for doc in search_results
                if doc.get("similarity_score", 0) > settings.SIMILARITY_THRESHOLD
            ]

            if not relevant_docs:
                logger.info(
                    f"No documents above threshold {settings.SIMILARITY_THRESHOLD}, "
                    f"returning all {len(search_results)} for grader"
                )
                return {
                    "status": "NOT_FOUND",
                    "documents": search_results,  # Pass all to GRADER for reranking
                    "search_query_used": "context_summary" if (is_followup and context_summary) else "question",
                    "next_agent": "GRADER"
                }

            logger.info(
                f"✅ Found {len(relevant_docs)} relevant documents "
                f"(searched with {'context' if is_followup and context_summary else 'question'})"
            )

            return {
                "status": "SUCCESS",
                "documents": relevant_docs,
                "search_query_used": "context_summary" if (is_followup and context_summary) else "question",
                "next_agent": "GRADER"
            }

        except Exception as e:
            logger.error(f"❌ Retriever error: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "documents": [],
                "next_agent": "REPORTER"
            }