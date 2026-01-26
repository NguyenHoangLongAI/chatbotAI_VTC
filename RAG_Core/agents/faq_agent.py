# RAG_Core/agents/faq_agent.py - ALWAYS USE LLM WITH STREAMING

from typing import Dict, Any, List, AsyncIterator
from models.llm_model import llm_model
from tools.vector_search import search_faq, rerank_faq
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class FAQAgent:
    def __init__(self):
        self.name = "FAQ"

        # Ngưỡng cho các giai đoạn
        self.vector_threshold = 0.5
        self.rerank_threshold = 0.6

        # REMOVED: direct_answer_threshold, force_similarity_threshold
        # → Luôn dùng LLM

        self.llm_prompt = """Bạn là một chuyên viên tư vấn khách hàng người Việt Nam thân thiện và chuyên nghiệp.

Câu hỏi người dùng: "{question}"

Kết quả tìm kiếm FAQ (đã được rerank):
{faq_results}

Hướng dẫn:
1. Kết quả đã được sắp xếp theo độ phù hợp (rerank_score)
2. Hãy dựa vào FAQ có rerank_score cao nhất để trả lời
3. Nếu không có FAQ nào phù hợp (tất cả score quá thấp), trả về "NOT_FOUND"
4. Trả lời bằng tiếng Việt, thân thiện và chính xác
5. Có thể kết hợp thông tin từ nhiều FAQ nếu cần
6. Đừng nói "Dựa vào FAQ..." hay "Theo tài liệu..." - trả lời trực tiếp như bạn biết

Trả lời:"""

    def process(
            self,
            question: str,
            is_followup: bool = False,
            context: str = "",
            **kwargs
    ) -> Dict[str, Any]:
        """
        Non-streaming process (for backward compatibility)
        """
        try:
            logger.info("=" * 50)
            logger.info("🤖 FAQ AGENT PROCESSING (NON-STREAMING)")
            logger.info("=" * 50)
            logger.info(f"📝 Question: '{question[:100]}'")

            # Vector search
            faq_results = search_faq.invoke({"query": question})

            if not faq_results or "error" in str(faq_results):
                logger.warning("❌ Vector search failed")
                return self._route_to_retriever("Vector search failed")

            # Filter by threshold
            filtered_faqs = [
                faq for faq in faq_results
                if faq.get("similarity_score", 0) >= self.vector_threshold
            ]

            if not filtered_faqs:
                logger.info(f"⚠️  No FAQ passed vector threshold {self.vector_threshold}")
                return self._route_to_retriever("No FAQ above vector threshold")

            logger.info(f"✅ Found {len(filtered_faqs)} FAQs above threshold")

            # Rerank
            logger.info(f"🎯 Reranking with Cohere")
            reranked_faqs = rerank_faq.invoke({
                "query": question,
                "faq_results": filtered_faqs
            })

            if not reranked_faqs:
                logger.error("❌ Reranking returned empty results")
                raise RuntimeError("FAQ reranking failed")

            best_faq = reranked_faqs[0]
            rerank_score = best_faq.get("rerank_score", 0)
            similarity_score = best_faq.get("similarity_score", 0)

            logger.info(f"📊 Best FAQ Scores:")
            logger.info(f"   Rerank:     {rerank_score:.3f}")
            logger.info(f"   Similarity: {similarity_score:.3f}")

            # Check if confident enough
            if rerank_score < self.rerank_threshold:
                logger.info(f"⚠️  Not confident: rerank={rerank_score:.3f} < {self.rerank_threshold}")
                return self._route_to_retriever(f"Rerank score too low: {rerank_score:.3f}")

            # ALWAYS USE LLM (no direct answer)
            logger.info(f"🤖 Using LLM to generate answer")

            faq_text = self._format_reranked_faq(reranked_faqs[:3])

            prompt = self.llm_prompt.format(
                question=question,
                faq_results=faq_text
            )

            response = llm_model.invoke(prompt)

            if "NOT_FOUND" in response.upper():
                logger.info("🔄 LLM determined FAQ not sufficient → RETRIEVER")
                return self._route_to_retriever("LLM rejected FAQ")

            if not response or len(response.strip()) < 10:
                logger.warning("⚠️  Generated answer too short → RETRIEVER")
                return self._route_to_retriever("Answer too short")

            logger.info(f"✅ FAQ answer generated via LLM")
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

    async def process_streaming(
            self,
            question: str,
            is_followup: bool = False,
            context: str = "",
            **kwargs
    ) -> AsyncIterator[str]:
        """
        REAL STREAMING: Always use LLM streaming
        """
        try:
            logger.info("=" * 50)
            logger.info("🤖 FAQ AGENT PROCESSING (STREAMING)")
            logger.info("=" * 50)
            logger.info(f"📝 Question: '{question[:100]}'")

            # Vector search
            logger.info(f"🔍 STEP 1: Vector Search")
            faq_results = search_faq.invoke({"query": question})

            if not faq_results or "error" in str(faq_results):
                logger.warning("❌ Vector search failed")
                yield "Không tìm thấy câu trả lời phù hợp. Hệ thống sẽ tìm kiếm trong tài liệu."
                return

            # Filter by threshold
            filtered_faqs = [
                faq for faq in faq_results
                if faq.get("similarity_score", 0) >= self.vector_threshold
            ]

            if not filtered_faqs:
                logger.info(f"⚠️  No FAQ passed vector threshold")
                yield "Không tìm thấy câu trả lời phù hợp. Hệ thống sẽ tìm kiếm trong tài liệu."
                return

            logger.info(f"✅ Found {len(filtered_faqs)} FAQs")

            # Rerank
            logger.info(f"🎯 STEP 2: Reranking")
            reranked_faqs = rerank_faq.invoke({
                "query": question,
                "faq_results": filtered_faqs
            })

            if not reranked_faqs:
                logger.error("❌ Reranking failed")
                yield "Không tìm thấy câu trả lời phù hợp. Hệ thống sẽ tìm kiếm trong tài liệu."
                return

            best_faq = reranked_faqs[0]
            rerank_score = best_faq.get("rerank_score", 0)

            logger.info(f"📊 Best rerank score: {rerank_score:.3f}")

            # Check confidence
            if rerank_score < self.rerank_threshold:
                logger.info(f"⚠️  Not confident enough → route to retriever")
                yield "Không tìm thấy câu trả lời phù hợp. Hệ thống sẽ tìm kiếm trong tài liệu."
                return

            # STREAM FROM LLM
            logger.info(f"🚀 STEP 3: Streaming from LLM")

            faq_text = self._format_reranked_faq(reranked_faqs[:3])

            prompt = self.llm_prompt.format(
                question=question,
                faq_results=faq_text
            )

            chunk_count = 0
            full_response = ""

            async for chunk in llm_model.astream(prompt):
                if chunk:
                    chunk_count += 1
                    full_response += chunk
                    logger.debug(f"Chunk #{chunk_count}: {chunk[:30]}...")
                    yield chunk

            logger.info(f"✅ Streamed {chunk_count} chunks")
            logger.info(f"📝 Full response length: {len(full_response)}")

            # Check if LLM said "NOT_FOUND"
            if "NOT_FOUND" in full_response.upper():
                logger.info("⚠️  LLM determined FAQ not sufficient")
                # Already yielded the response, just log
                return

            logger.info("=" * 50 + "\n")

        except Exception as e:
            logger.error(f"❌ FAQ streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi FAQ: {str(e)}]"

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
        """Route to retriever"""
        logger.info(f"→ Routing to RETRIEVER: {reason}")
        return {
            "status": "NOT_FOUND",
            "answer": "",
            "references": [],
            "next_agent": "RETRIEVER"
        }

    def set_thresholds(
            self,
            vector_threshold: float = None,
            rerank_threshold: float = None
    ):
        """Update thresholds"""
        if vector_threshold is not None:
            self.vector_threshold = vector_threshold
            logger.info(f"Vector threshold updated to {vector_threshold}")

        if rerank_threshold is not None:
            self.rerank_threshold = rerank_threshold
            logger.info(f"Rerank threshold updated to {rerank_threshold}")