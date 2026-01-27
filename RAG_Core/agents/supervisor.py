from typing import Dict, Any, List
import logging
import json
import re

from models.llm_model import llm_model
from tools.vector_search import check_database_connection

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    SUPERVISOR AGENT
    - Resolve follow-up question
    - Replace ambiguous references (thành phần thứ X, nó, phần này...)
    - Classify agent
    """

    def __init__(self):
        self.name = "SUPERVISOR"

        self.prompt_template = """
Bạn là CHUYÊN GIA phân tích hội thoại và điều phối chatbot đào tạo chuyển đổi số.

=====================
NHIỆM VỤ BẮT BUỘC
=====================
1. Phân tích câu hỏi hiện tại dựa trên lịch sử hội thoại.
2. Xác định câu hỏi có phải FOLLOW-UP hay không.
3. Nếu là follow-up:
   - Truy vết lịch sử để xác định chính xác đối tượng được nhắc tới.
   - Đặc biệt chú ý các cụm:
     "thành phần thứ X", "phần này", "nó", "ý trên", "cái đó"
   - Nếu lịch sử có DANH SÁCH ĐÁNH SỐ → ánh xạ theo ĐÚNG THỨ TỰ.
   - Viết lại câu hỏi RÕ NGHĨA, KHÔNG ĐẠI TỪ.
4. Nếu không đủ dữ liệu để làm rõ → giữ nguyên câu hỏi gốc.

=====================
PHÂN LOẠI AGENT
=====================
- FAQ: đào tạo kỹ năng số, chuyển đổi số cho người dân / doanh nghiệp, kiến thức công nghệ thông tin
- CHATTER: cảm xúc tiêu cực, cần làm dịu
- REPORTER: lỗi hệ thống, sự cố kỹ thuật
- OTHER: ngoài phạm vi

=====================
INPUT
=====================
Câu hỏi hiện tại:
"{question}"

Lịch sử hội thoại (mới → cũ):
{history}

Trạng thái hệ thống:
{system_status}

=====================
OUTPUT (JSON ONLY)
=====================
{{
  "is_followup": true | false,
  "contextualized_question": "...",
  "context_summary": "...",
  "agent": "FAQ | CHATTER | REPORTER | OTHER"
}}

=====================
VÍ DỤ BẮT BUỘC
=====================
Lịch sử:
1. Cơ sở hạ tầng
2. Hệ thống quản lý
3. Công cụ khai thác dữ liệu

Câu hỏi:
"Chi tiết về thành phần thứ 2"

➡ contextualized_question:
"Chi tiết về hệ thống quản lý trong nền tảng chuyển đổi số cho doanh nghiệp"

CHỈ TRẢ VỀ JSON. KHÔNG GIẢI THÍCH.
"""

    # =========================
    # PUBLIC API
    # =========================
    def classify_request(
        self,
        question: str,
        history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:

        try:
            logger.info("👨‍💼 SUPERVISOR START")
            logger.info(f"Question: {question}")

            # 1. Check system status
            db_status = check_database_connection.invoke({})
            if not db_status.get("connected", False):
                return self._reporter_response(question)

            # 2. Format history
            history_text = self._format_history(history or [])

            # 3. Build prompt
            prompt = self.prompt_template.format(
                question=question,
                history=history_text,
                system_status="Bình thường"
            )

            # 4. Call LLM
            logger.info("🤖 Calling LLM (context resolution + classification)")
            raw_response = llm_model.invoke(prompt)

            # 5. Parse JSON
            parsed = self._parse_json(raw_response)

            # 6. Validate output
            return self._normalize_output(parsed, question)

        except Exception as e:
            logger.error("❌ Supervisor error", exc_info=True)
            return self._fallback_response(question)

    # =========================
    # INTERNAL METHODS
    # =========================
    def _format_history(self, history: List[Any]) -> str:
        """
        Format history an toàn cho cả:
        - dict: {"role": "...", "content": "..."}
        - ChatMessage / HumanMessage / AIMessage (LangChain, Pydantic)
        """

        if not history:
            return "Không có lịch sử"

        recent = history[-6:]
        lines = []

        for msg in recent:
            # CASE 1: dict
            if isinstance(msg, dict):
                role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
                content = msg.get("content", "")

            # CASE 2: ChatMessage / LangChain message
            else:
                role_attr = getattr(msg, "role", None)
                content = getattr(msg, "content", "")

                role = "Người dùng" if role_attr == "user" else "Trợ lý"

            if content:
                lines.append(f"{role}: {content[:300]}")

        return "\n".join(lines) if lines else "Không có lịch sử"

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON object from LLM output
        """
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError("No JSON found")

            return json.loads(match.group(0))

        except Exception as e:
            logger.warning(f"⚠️ JSON parse failed: {e}")
            return {}

    def _normalize_output(
        self,
        parsed: Dict[str, Any],
        original_question: str
    ) -> Dict[str, Any]:

        agent = parsed.get("agent", "FAQ").upper()
        if agent not in {"FAQ", "CHATTER", "REPORTER", "OTHER"}:
            agent = "FAQ"

        contextualized_question = parsed.get(
            "contextualized_question",
            original_question
        ).strip() or original_question

        return {
            "agent": agent,
            "is_followup": bool(parsed.get("is_followup", False)),
            "contextualized_question": contextualized_question,
            "context_summary": parsed.get("context_summary", "")
        }

    def _reporter_response(self, question: str) -> Dict[str, Any]:
        return {
            "agent": "REPORTER",
            "is_followup": False,
            "contextualized_question": question,
            "context_summary": "Hệ thống mất kết nối"
        }

    def _fallback_response(self, question: str) -> Dict[str, Any]:
        return {
            "agent": "FAQ",
            "is_followup": False,
            "contextualized_question": question,
            "context_summary": "Fallback do lỗi xử lý supervisor"
        }