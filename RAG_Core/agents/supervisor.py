# RAG_Core/agents/supervisor.py

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from models.llm_model import llm_model
from tools.vector_search import check_database_connection
from utils.context_processor import context_processor
import logging
import json
import re

logger = logging.getLogger(__name__)


class SupervisorAgent:
    def __init__(self):
        self.name = "SUPERVISOR"
        self.classification_prompt = """Bạn là chuyên viên đào tạo kỹ năng chuyển đổi số, kiến thức sử dụng công nghệ thông tin cơ bản cho người dân - người điều phối chính của hệ thống chatbot.

        Nhiệm vụ:
        1. Dựa vào lịch sử hội thoại và câu hỏi hiện tại, hãy xác định ngữ cảnh (context) mà người dùng đang đề cập đến.
        2. Phân loại câu hỏi và chọn agent phù hợp để xử lý.

        Các agent có thể chọn:
        - FAQ: Dùng cho chào hỏi thân thiện, câu hỏi thường gặp, hoặc các yêu cầu liên quan đến đào tạo kỹ năng chuyển đổi số cho người dân và doanh nghiệp.
        - OTHER: Câu hỏi hoặc yêu cầu nằm ngoài phạm vi chuyển đổi số.
        - CHATTER: Người dùng có dấu hiệu không hài lòng, giận dữ, hoặc cần được an ủi, làm dịu.
        - REPORTER: Khi người dùng phản ánh lỗi, mất kết nối, hoặc vấn đề kỹ thuật của hệ thống.

        Đầu vào:
        Câu hỏi gốc: "{original_question}"
        Câu hỏi đã được làm rõ ngữ cảnh: "{contextualized_question}"
        Lịch sử hội thoại: {history}
        Trạng thái hệ thống: {system_status}
        Có phải follow-up question: {is_followup}
        Context liên quan: {relevant_context}

        YÊU CẦU QUAN TRỌNG:
        - BẮT BUỘC trả lời context_summary BẰNG TIẾNG VIỆT
        - KHÔNG được dùng tiếng Anh trong context_summary
        - Nếu không có context, ghi "Câu hỏi độc lập" hoặc "Không có ngữ cảnh"

        Hãy trả lời đúng định dạng JSON:
        {{
          "context_summary": "Tóm tắt ngắn gọn ngữ cảnh BẰNG TIẾNG VIỆT (nếu có)",
          "agent": "FAQ" hoặc "CHATTER" hoặc "REPORTER" hoặc "OTHER"
        }}

        Ví dụ context_summary hợp lệ:
        - "Hỏi về cách sử dụng AI"
        - "Tiếp tục về chủ đề an toàn thông tin"
        - "Câu hỏi độc lập"
        - "Không có ngữ cảnh"

        Ví dụ KHÔNG hợp lệ (đừng làm thế này):
        - "Asking about AI usage"
        - "Follow-up question about security"

        Chỉ trả về JSON, không thêm text nào khác."""

    def classify_request(
            self,
            question: str,
            history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Phân loại yêu cầu và chọn agent phù hợp

        UPDATED: Better logging for context extraction
        """
        try:
            logger.info("-" * 50)
            logger.info("👨‍💼 SUPERVISOR CLASSIFICATION")
            logger.info("-" * 50)
            logger.info(f"📝 Original Question: '{question}'")
            logger.info(f"📚 History Length: {len(history) if history else 0} messages")

            # Kiểm tra trạng thái hệ thống
            db_status = check_database_connection.invoke({})

            if not db_status.get("connected", False):
                logger.warning("⚠️  Database not connected → REPORTER")
                return {
                    "agent": "REPORTER",
                    "contextualized_question": question,
                    "context_summary": "Hệ thống mất kết nối",
                    "is_followup": False
                }

            # Xử lý context từ history
            logger.info("🔍 Extracting context from history...")

            context_info = context_processor.extract_context_from_history(
                history or [],
                question
            )

            contextualized_question = context_info["contextualized_question"]
            is_followup = context_info["is_followup"]
            relevant_context = context_info["relevant_context"]

            # Log contextualization result
            if is_followup:
                logger.info("✅ FOLLOW-UP QUESTION DETECTED")
                logger.info(f"   Original:      '{question}'")
                logger.info(f"   Contextualized: '{contextualized_question}'")
                logger.info(f"   Context:       {relevant_context[:100]}...")
            else:
                logger.info("📌 STANDALONE QUESTION")
                logger.info(f"   Question: '{contextualized_question}'")

            # Nếu LLM không xác định được ngữ cảnh hợp lệ
            if "[cần làm rõ]" in contextualized_question:
                logger.info("⚠️  Context unclear → treating as standalone")
                contextualized_question = question
                is_followup = False
                relevant_context = ""

            # Format lịch sử
            history_text = self._format_history(history or [])

            # Tạo prompt
            prompt = self.classification_prompt.format(
                original_question=question,
                contextualized_question=contextualized_question,
                history=history_text,
                system_status="Bình thường" if db_status.get("connected") else "Lỗi kết nối",
                is_followup="Có" if is_followup else "Không",
                relevant_context=relevant_context[:300] if relevant_context else "Không có"
            )

            # Gọi LLM để phân loại
            logger.info("🤖 Calling LLM for agent classification...")
            response = llm_model.invoke(prompt)

            # Parse JSON response
            classification = self._parse_classification_response(response)

            # Validate agent choice
            valid_agents = ["FAQ", "CHATTER", "REPORTER", "OTHER"]
            agent_choice = classification.get("agent", "").upper()

            if agent_choice not in valid_agents:
                logger.warning(f"⚠️  Invalid agent '{agent_choice}' → fallback classification")
                agent_choice = self._fallback_classify(contextualized_question)

            logger.info(f"\n🎯 CLASSIFICATION RESULT:")
            logger.info(f"   Agent: {agent_choice}")
            logger.info(f"   Is Follow-up: {is_followup}")
            logger.info(f"   Context Summary: {classification.get('context_summary', '')[:100]}")
            logger.info("-" * 50 + "\n")

            return {
                "agent": agent_choice,
                "contextualized_question": contextualized_question,
                "context_summary": classification.get("context_summary", ""),
                "is_followup": is_followup,
                "reasoning": classification.get("reasoning", "")
            }

        except Exception as e:
            logger.error(f"❌ Error in supervisor classification: {e}", exc_info=True)
            logger.info("↩️  Using fallback classification")
            return {
                "agent": self._fallback_classify(question),
                "contextualized_question": question,
                "context_summary": "",
                "is_followup": False,
                "reasoning": "Fallback due to error"
            }

    def _parse_classification_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response từ LLM"""
        try:
            # Tìm JSON block trong response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            # Fallback parsing
            return {
                "agent": "FAQ",
                "context_summary": "",
                "reasoning": "Parse failed"
            }

        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            return {
                "agent": "FAQ",
                "context_summary": "",
                "reasoning": "Parse error"
            }

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format history thành text, xử lý cả dict và ChatMessage objects"""
        if not history:
            return "Không có lịch sử"

        # Chỉ lấy 3 turn gần nhất (6 messages)
        recent_history = history[-6:] if len(history) > 6 else history

        history_lines = []
        for msg in recent_history:
            # Xử lý cả dict và ChatMessage object
            if isinstance(msg, dict):
                role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
                content = msg.get("content", "")[:200]
            else:
                # ChatMessage object
                role = "Người dùng" if getattr(msg, "role", "") == "user" else "Trợ lý"
                content = getattr(msg, "content", "")[:200]

            if content:
                history_lines.append(f"{role}: {content}")

        return "\n".join(history_lines) if history_lines else "Không có lịch sử"

    def _fallback_classify(self, question: str) -> str:
        """Phân loại dự phòng dựa trên từ khóa"""
        question_lower = question.lower()

        # Chatter keywords - cảm xúc tiêu cực
        chatter_keywords = [
            "tệ", "kém", "tồi", "không hài lòng", "giận",
            "phản đối", "khiếu nại", "thất vọng", "tức giận"
        ]
        if any(keyword in question_lower for keyword in chatter_keywords):
            return "CHATTER"

        # Reporter keywords - lỗi hệ thống
        reporter_keywords = [
            "lỗi", "không hoạt động", "bị lỗi", "không kết nối",
            "không truy cập được", "hỏng", "không phản hồi"
        ]
        if any(keyword in question_lower for keyword in reporter_keywords):
            return "REPORTER"

        # FAQ keywords - câu hỏi thông thường
        faq_keywords = [
            "là gì", "như thế nào", "sao", "tại sao", "có phải",
            "giờ làm việc", "liên hệ", "hướng dẫn", "cách", "thế nào"
        ]
        if any(keyword in question_lower for keyword in faq_keywords):
            return "FAQ"

        # Default to FAQ for document search
        return "FAQ"