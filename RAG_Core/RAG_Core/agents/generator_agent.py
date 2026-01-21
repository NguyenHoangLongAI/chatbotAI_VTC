# RAG_Core/agents/generator_agent.py - WITH URL FORMATTING

from typing import Dict, Any, List, AsyncIterator
from models.llm_model import llm_model
import logging

logger = logging.getLogger(__name__)


class GeneratorAgent:
    def __init__(self):
        self.name = "GENERATOR"

        # Standard prompt (unchanged)
        self.standard_prompt = """Bạn là một chuyên viên tư vấn khách hàng người Việt Nam thân thiện và chuyên nghiệp.

Câu hỏi của khách hàng: "{question}"

Thông tin tham khảo từ tài liệu:
{documents}

Lịch sử trò chuyện gần đây:
{history}

Yêu cầu trả lời:
- Trả lời bằng giọng văn tự nhiên như người Việt Nam nói chuyện
- Trả lời thẳng vào vấn đề, ngắn gọn súc tích
- Dựa vào thông tin tài liệu nhưng diễn đạt theo cách hiểu của bạn
- Kết thúc bằng câu hỏi ngắn để tiếp tục hỗ trợ nếu cần

Hãy trả lời như đang nói chuyện trực tiếp với khách hàng:"""

        # Follow-up prompt (unchanged)
        self.followup_prompt = """Bạn là một chuyên viên tư vấn khách hàng người Việt Nam thân thiện và chuyên nghiệp.

🔍 NGỮ CẢNH CUỘC TRÒ CHUYỆN:
{context_summary}

📝 LỊCH SỬ GẦN NHẤT:
{recent_history}

❓ CÂU HỎI FOLLOW-UP CỦA KHÁCH HÀNG: "{question}"

📚 THÔNG TIN TÀI LIỆU LIÊN QUAN:
{documents}

⚠️ YÊU CẦU ĐẶC BIỆT cho follow-up question:
1. Nhận biết rằng khách hàng đang hỏi tiếp về chủ đề đã thảo luận
2. Tham chiếu đến thông tin đã cung cấp trước đó một cách tự nhiên
3. Trả lời cụ thể vào phần mà khách hàng muốn biết thêm
4. KHÔNG lặp lại toàn bộ thông tin đã nói, chỉ tập trung vào phần được hỏi

📋 YÊU CẦU CHUNG:
- Trả lời bằng giọng văn tự nhiên như người Việt Nam nói chuyện
- Ngắn gọn, súc tích, đúng trọng tâm
- Kết thúc bằng câu hỏi để tiếp tục hỗ trợ nếu cần

Hãy trả lời:"""

    def _deduplicate_references(self, references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Loại bỏ references trùng lặp"""
        if not references:
            return []

        seen_doc_ids = set()
        unique_references = []

        for ref in references:
            doc_id = ref.get('document_id')
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_references.append(ref)

        return unique_references

    def _format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents thành text"""
        if not documents:
            return "Không có tài liệu tham khảo"

        doc_lines = []
        for i, doc in enumerate(documents[:5], 1):
            description = doc.get('description', '')
            score = doc.get('similarity_score', 0)
            doc_lines.append(f"[Tài liệu {i}] (Độ liên quan: {score:.2f})\n{description}")

        return "\n\n".join(doc_lines)

    def _format_history(self, history: List, max_turns: int = 2) -> str:
        """Format lịch sử hội thoại"""
        if not history:
            return "Không có lịch sử"

        # Normalize history
        normalized_history = []
        for msg in history:
            if isinstance(msg, dict):
                normalized_history.append({
                    "role": msg.get("role", ""),
                    "content": msg.get("content", "")
                })
            else:
                normalized_history.append({
                    "role": getattr(msg, "role", ""),
                    "content": getattr(msg, "content", "")
                })

        # Lấy N turn gần nhất
        recent_history = normalized_history[-(max_turns * 2):] if len(
            normalized_history) > max_turns * 2 else normalized_history

        history_lines = []
        for msg in recent_history:
            role = "👤 Khách hàng" if msg.get("role") == "user" else "🤖 Trợ lý"
            content = msg.get("content", "")
            if content:
                history_lines.append(f"{role}: {content}")

        return "\n".join(history_lines) if history_lines else "Không có lịch sử"

    def _extract_context_summary(self, history: List) -> str:
        """Trích xuất context summary"""
        if not history or len(history) < 2:
            return "Đây là câu hỏi đầu tiên"

        # Normalize history
        normalized_history = []
        for msg in history:
            if isinstance(msg, dict):
                normalized_history.append(msg)
            else:
                normalized_history.append({
                    "role": getattr(msg, "role", ""),
                    "content": getattr(msg, "content", "")
                })

        # Lấy câu hỏi trước
        for i in range(len(normalized_history) - 1, -1, -1):
            if normalized_history[i].get("role") == "user":
                prev_question = normalized_history[i].get("content", "")

                for j in range(i + 1, len(normalized_history)):
                    if normalized_history[j].get("role") == "assistant":
                        prev_answer = normalized_history[j].get("content", "")
                        return f"Chủ đề đang thảo luận: {prev_question}\nĐã trả lời: {prev_answer[:200]}..."

                return f"Chủ đề đang thảo luận: {prev_question}"

        return "Đang trong cuộc trò chuyện"

    def _format_references_footer(self, references: List[Dict[str, Any]]) -> str:
        """
        NEW: Format references với URLs thành footer

        Returns formatted text như:

        📚 Tài liệu tham khảo:
        1. Thông tư 01/2022/TT-BTTTT
           📎 https://ngrok.../01_2022_TT-BTTTT.pdf
        2. Quyết định 02/2023/QĐ-TTg
           📎 https://ngrok.../02_2023_QD-TTg.pdf
        """
        if not references:
            return ""

        # Chỉ lấy references có URL
        refs_with_urls = [ref for ref in references if ref.get('url')]

        if not refs_with_urls:
            return ""

        footer_lines = ["\n\n📚 Tài liệu tham khảo:"]

        for i, ref in enumerate(refs_with_urls[:3], 1):  # Limit 5 URLs
            filename = ref.get('filename', ref.get('document_id', 'Unknown'))
            url = ref.get('url')

            # Remove extension for display
            display_name = filename.rsplit('.', 1)[0] if '.' in filename else filename

            footer_lines.append(f"{i}. {display_name}")
            footer_lines.append(f"   📎 {url}")

        return "\n".join(footer_lines)

    def process(
            self,
            question: str,
            documents: List[Dict[str, Any]],
            references: List[Dict[str, Any]] = None,
            history: List[Dict[str, str]] = None,
            is_followup: bool = False,
            context_summary: str = "",
            **kwargs
    ) -> Dict[str, Any]:
        """Non-streaming generation with URL footer"""
        try:
            if not documents:
                return {
                    "status": "ERROR",
                    "answer": "Không có tài liệu để tạo câu trả lời",
                    "references": [],
                    "next_agent": "end"
                }

            doc_text = self._format_documents(documents)
            history_text = self._format_history(history or [], max_turns=2)

            # Chọn prompt
            if is_followup:
                if not context_summary:
                    context_summary = self._extract_context_summary(history or [])

                prompt = self.followup_prompt.format(
                    question=question,
                    context_summary=context_summary,
                    recent_history=history_text,
                    documents=doc_text
                )
            else:
                prompt = self.standard_prompt.format(
                    question=question,
                    history=history_text,
                    documents=doc_text
                )

            # Generate answer
            answer = llm_model.invoke(prompt)

            if not answer or len(answer.strip()) < 10:
                answer = "Tôi đã tìm thấy thông tin liên quan nhưng gặp khó khăn trong việc tạo câu trả lời."

            # ===== NEW: Append references footer với URLs =====
            unique_references = self._deduplicate_references(references or [])
            references_footer = self._format_references_footer(unique_references)

            if references_footer:
                answer = answer + references_footer
                logger.info(f"✅ Appended {len([r for r in unique_references if r.get('url')])} URLs to answer")

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": unique_references,
                "next_agent": "end"
            }

        except Exception as e:
            logger.error(f"Error in generator agent: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "answer": f"Lỗi tạo câu trả lời: {str(e)}",
                "references": [],
                "next_agent": "end"
            }

    async def process_streaming(
            self,
            question: str,
            documents: List[Dict[str, Any]],
            references: List[Dict[str, Any]] = None,
            history: List[Dict[str, str]] = None,
            is_followup: bool = False,
            context_summary: str = "",
            **kwargs
    ) -> AsyncIterator[str]:
        """
        Streaming generation with URL footer appended at the end
        """
        try:
            logger.info(f"🚀 Generator: Starting streaming for: {question[:50]}...")

            if not documents:
                yield "Không có tài liệu để tạo câu trả lời."
                return

            # Format inputs
            doc_text = self._format_documents(documents)
            history_text = self._format_history(history or [], max_turns=2)

            # Choose prompt
            if is_followup:
                if not context_summary:
                    context_summary = self._extract_context_summary(history or [])

                prompt = self.followup_prompt.format(
                    question=question,
                    context_summary=context_summary,
                    recent_history=history_text,
                    documents=doc_text
                )
            else:
                prompt = self.standard_prompt.format(
                    question=question,
                    history=history_text,
                    documents=doc_text
                )

            logger.info(f"📝 Generator: Prompt prepared, length={len(prompt)}")

            # Stream from LLM
            chunk_count = 0
            async for chunk in llm_model.astream(prompt):
                if chunk:
                    chunk_count += 1
                    logger.debug(f"Generator yielding chunk #{chunk_count}: {chunk[:30]}...")
                    yield chunk

            logger.info(f"✅ Generator: Completed streaming {chunk_count} chunks")

            # ===== NEW: Stream references footer with URLs =====
            unique_references = self._deduplicate_references(references or [])
            references_footer = self._format_references_footer(unique_references)

            if references_footer:
                logger.info("📎 Streaming references footer with URLs...")
                # Stream footer slowly for better UX
                for line in references_footer.split('\n'):
                    yield line + '\n'
                    await asyncio.sleep(0.05)  # Small delay between lines

                logger.info(f"✅ Streamed {len([r for r in unique_references if r.get('url')])} URLs")

        except Exception as e:
            logger.error(f"❌ Generator streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi: {str(e)}]"