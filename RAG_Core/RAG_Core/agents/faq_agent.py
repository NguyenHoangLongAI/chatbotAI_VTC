# RAG_Core/agents/faq_agent.py (NO FALLBACK VERSION)

from typing import Dict, Any, List
from models.llm_model import llm_model
from tools.vector_search import search_faq, rerank_faq
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class FAQAgent:
    def __init__(self):
        self.name = "FAQ"

        # Ngưỡng cho các giai đoạn khác nhau
        self.vector_threshold = 0.5
        self.rerank_threshold = 0.6
        self.direct_answer_threshold = 0.75
        self.force_similarity_threshold = 0.85
        self.use_llm = True

        self.standard_prompt = """Bạn là một chuyên viên tư vấn khách hàng người Việt Nam thân thiện và chuyên nghiệp.

Câu hỏi người dùng: "{question}"

Kết quả tìm kiếm FAQ (đã được rerank):
{faq_results}

Hướng dẫn:
1. Kết quả đã được sắp xếp theo độ phù hợp (rerank_score)
2. Nếu FAQ đầu tiên có rerank_score > {rerank_threshold}, hãy trả lời dựa trên đó
3. Nếu không có FAQ phù hợp, trả về "NOT_FOUND"
4. Trả lời bằng tiếng Việt, thân thiện và chính xác
5. Có thể kết hợp thông tin từ nhiều FAQ nếu cần

Trả lời:"""

    def process(
            self,
            question: str,  # ✅ This MUST be contextualized question (if follow-up)
            is_followup: bool = False,
            context: str = "",
            **kwargs
    ) -> Dict[str, Any]:
        """
        Xử lý câu hỏi FAQ - UPDATED: Receives contextualized question

        Args:
            question: CONTEXTUALIZED question (đã có context nếu là follow-up)
            is_followup: Flag để biết có phải follow-up không
            context: Additional context (optional, from supervisor)

        Returns:
            Dict with status, answer, references, next_agent
        """
        try:
            logger.info("=" * 50)
            logger.info("🤖 FAQ AGENT PROCESSING")
            logger.info("=" * 50)
            logger.info(f"📝 Question: '{question[:100]}'")
            logger.info(f"🔄 Is Follow-up: {is_followup}")
            logger.info(f"📚 Context: {context[:100] if context else 'None'}")

            # ===============================================
            # BƯỚC 1: VECTOR SEARCH
            # ===============================================
            logger.info(f"\n🔍 STEP 1: Vector Search (threshold={self.vector_threshold})")

            faq_results = search_faq.invoke({"query": question})
            # ↑ ✅ QUESTION is CONTEXTUALIZED (if needed)

            if not faq_results or "error" in str(faq_results):
                logger.warning("❌ Vector search failed or returned error")
                return self._route_to_retriever("Vector search failed")

            # Lọc theo vector threshold
            filtered_faqs = [
                faq for faq in faq_results
                if faq.get("similarity_score", 0) >= self.vector_threshold
            ]

            if not filtered_faqs:
                logger.info(f"⚠️  No FAQ passed vector threshold {self.vector_threshold}")
                return self._route_to_retriever("No FAQ above vector threshold")

            logger.info(f"✅ Found {len(filtered_faqs)} FAQs above threshold")

            # Log top 3 candidates
            for i, faq in enumerate(filtered_faqs[:3], 1):
                logger.info(
                    f"   {i}. Score: {faq.get('similarity_score', 0):.3f} - "
                    f"Q: '{faq.get('question', '')[:60]}...'"
                )

            # ===============================================
            # BƯỚC 2: RERANK (receives contextualized question)
            # ===============================================
            logger.info(f"\n🎯 STEP 2: Reranking with Cohere")
            logger.info(f"   Reranking query: '{question[:100]}'")

            reranked_faqs = rerank_faq.invoke({
                "query": question,  # ← ✅ CONTEXTUALIZED QUESTION
                "faq_results": filtered_faqs
            })

            if not reranked_faqs:
                logger.error("❌ Reranking returned empty results")
                raise RuntimeError("FAQ reranking failed: empty results")

            best_faq = reranked_faqs[0]
            rerank_score = best_faq.get("rerank_score", 0)
            similarity_score = best_faq.get("similarity_score", 0)

            logger.info(f"📊 Best FAQ Scores:")
            logger.info(f"   Rerank:     {rerank_score:.3f}")
            logger.info(f"   Similarity: {similarity_score:.3f}")
            logger.info(f"   Question:   '{best_faq.get('question', '')[:100]}'")

            # ===============================================
            # BƯỚC 3: CHECK THRESHOLD
            # ===============================================
            is_confident = (
                    similarity_score >= self.force_similarity_threshold
                    or rerank_score >= self.direct_answer_threshold
            )

            if not is_confident:
                logger.info(
                    f"⚠️  Not confident enough:\n"
                    f"   Rerank {rerank_score:.3f} < {self.rerank_threshold}\n"
                    f"   Similarity {similarity_score:.3f} < {self.force_similarity_threshold}\n"
                    f"   → Routing to RETRIEVER"
                )
                return self._route_to_retriever(
                    f"Not confident: rerank={rerank_score:.3f}, sim={similarity_score:.3f}"
                )

            # ===============================================
            # BƯỚC 4: TRẢ LỜI TRỰC TIẾP HAY QUA LLM
            # ===============================================
            if (rerank_score >= self.direct_answer_threshold or
                    similarity_score >= self.force_similarity_threshold):
                logger.info(
                    f"✅ HIGH CONFIDENCE - Direct Answer\n"
                    f"   Rerank: {rerank_score:.3f} (threshold: {self.direct_answer_threshold})\n"
                    f"   Similarity: {similarity_score:.3f} (threshold: {self.force_similarity_threshold})"
                )

                answer = self._format_direct_answer(best_faq, question)

                logger.info(f"📤 Answer: {answer[:100]}...")
                logger.info("=" * 50 + "\n")

                return {
                    "status": "SUCCESS",
                    "answer": answer,
                    "mode": "direct",
                    "references": [
                        {
                            "document_id": best_faq.get("faq_id"),
                            "type": "FAQ",
                            "description": best_faq.get("question", ""),
                            "rerank_score": round(rerank_score, 4),
                            "similarity_score": round(similarity_score, 4)
                        }
                    ],
                    "next_agent": "end"
                }

            # ===============================================
            # BƯỚC 5: DÙNG LLM
            # ===============================================
            logger.info(
                f"🤖 MEDIUM CONFIDENCE - Using LLM\n"
                f"   Rerank: {rerank_score:.3f}\n"
                f"   Similarity: {similarity_score:.3f}"
            )

            faq_text = self._format_reranked_faq(reranked_faqs[:3])

            prompt = self.standard_prompt.format(
                question=question,
                faq_results=faq_text,
                rerank_threshold=self.rerank_threshold
            )

            response = llm_model.invoke(prompt)

            if "NOT_FOUND" in response.upper():
                logger.info("🔄 LLM determined FAQ not sufficient → RETRIEVER")
                return self._route_to_retriever("LLM rejected FAQ")

            if not response or len(response.strip()) < 10:
                logger.warning("⚠️  Generated answer too short → RETRIEVER")
                return self._route_to_retriever("Answer too short")

            logger.info(f"✅ FAQ answer generated via LLM")
            logger.info(f"📤 Answer: {response[:100]}...")
            logger.info("=" * 50 + "\n")

            return {
                "status": "SUCCESS",
                "answer": response,
                "mode": "llm",
                "references": [
                    {
                        "document_id": best_faq.get("faq_id"),
                        "type": "FAQ",
                        "description": best_faq.get("question", ""),
                        "rerank_score": round(rerank_score, 4),
                        "similarity_score": round(similarity_score, 4)
                    }
                ],
                "next_agent": "end"
            }

        except RuntimeError as e:
            logger.error(f"❌ Critical FAQ error: {e}")
            raise

        except Exception as e:
            logger.error(f"❌ Unexpected error in FAQ agent: {e}", exc_info=True)
            raise RuntimeError(f"FAQ agent failed: {e}") from e

    # ===============================================================
    # Helper Functions
    # ===============================================================

    def _format_direct_answer(self, faq: Dict[str, Any], question: str) -> str:
        """Format câu trả lời trực tiếp"""
        return f"{faq.get('answer', '')}"

    def _format_reranked_faq(self, faq_results: List[Dict[str, Any]]) -> str:
        """Format FAQ đã được rerank"""
        if not faq_results:
            return "Không tìm thấy FAQ phù hợp"

        formatted_lines = []
        for i, faq in enumerate(faq_results, 1):
            question = faq.get('question', '')
            answer = faq.get('answer', '')
            rerank_score = faq.get('rerank_score', 0)
            similarity_score = faq.get('similarity_score', 0)

            formatted_lines.append(
                f"FAQ {i} (Rerank: {rerank_score:.3f}, Similarity: {similarity_score:.3f}):\n"
                f"Q: {question}\n"
                f"A: {answer}\n"
            )

        return "\n".join(formatted_lines)

    def _route_to_retriever(self, reason: str) -> Dict[str, Any]:
        logger.info(f"Routing to RETRIEVER: {reason}")
        return {
            "status": "NOT_FOUND",
            "answer": "",
            "references": [],
            "next_agent": "RETRIEVER"
        }

    def set_thresholds(
            self,
            vector_threshold: float = None,
            rerank_threshold: float = None,
            direct_answer_threshold: float = None,
            use_llm: bool = None
    ):
        if vector_threshold is not None:
            self.vector_threshold = vector_threshold
            logger.info(f"Vector threshold updated to {vector_threshold}")

        if rerank_threshold is not None:
            self.rerank_threshold = rerank_threshold
            logger.info(f"Rerank threshold updated to {rerank_threshold}")

        if direct_answer_threshold is not None:
            self.direct_answer_threshold = direct_answer_threshold
            logger.info(f"Direct answer threshold updated to {direct_answer_threshold}")

        if use_llm is not None:
            self.use_llm = use_llm
            logger.info(f"Use LLM mode: {use_llm}")