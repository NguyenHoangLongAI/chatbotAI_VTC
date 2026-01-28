# RAG_Core/agents/personalization_agent.py

from typing import Dict, Any, AsyncIterator
from models.llm_model import llm_model
import logging

logger = logging.getLogger(__name__)


class PersonalizationAgent:
    """
    Agent cá nhân hóa câu trả lời dựa trên thông tin khách hàng

    Nhiệm vụ:
    - Phân tích name và introduction để xác định context khách hàng
    - Điều chỉnh tone, từ xưng hô, và nội dung phù hợp
    - Tích hợp thông tin cá nhân vào câu trả lời một cách tự nhiên
    """

    def __init__(self):
        self.name = "PERSONALIZATION"

        # Prompt template cho personalization
        self.personalization_prompt = """Bạn là trợ lý ảo Onetouch - chuyên gia đào tạo kỹ năng số cho người dân và doanh nghiệp.

THÔNG TIN KHÁCH HÀNG:
- Tên: {customer_name}
- Giới thiệu: {customer_introduction}

PHÂN TÍCH KHÁCH HÀNG:
{customer_analysis}

CÂU HỎI CỦA KHÁCH HÀNG:
"{question}"

LỊCH SỬ HỘI THOẠI:
{history}

CÂU TRẢ LỜI GỐC (từ RAG system):
{original_answer}

NHIỆM VỤ CỦA BẠN:
1. **Xưng hô phù hợp**:
   - Nếu là Giám đốc/Tổng giám đốc/CEO: "Thưa Anh/Chị [Tên]"
   - Nếu là Manager/Trưởng phòng: "Anh/Chị [Tên]"
   - Nếu là nhân viên/cá nhân: "Bạn [Tên]"
   - Nếu không rõ: "Anh/Chị [Tên]"

2. **Cá nhân hóa nội dung**:
   - Liên kết câu trả lời với lĩnh vực/ngành nghề của khách hàng
   - Đưa ra ví dụ phù hợp với context công ty/vai trò
   - Điều chỉnh độ chuyên sâu dựa trên vị trí (lãnh đạo → tổng quan chiến lược, nhân viên → chi tiết thực hành)

3. **Tone phù hợp**:
   - Lãnh đạo cấp cao: Tôn trọng, tư vấn chiến lược, tầm nhìn
   - Quản lý: Chuyên nghiệp, thực tiễn, giải pháp cụ thể
   - Nhân viên: Thân thiện, hướng dẫn chi tiết, dễ hiểu

4. **Giữ nguyên thông tin chính xác** từ câu trả lời gốc - CHỈ thêm phần cá nhân hóa

YÊU CẦU ĐỊNH DẠNG:
- BẮT ĐẦU bằng lời xưng hô phù hợp
- Nội dung chính: Tích hợp câu trả lời gốc với context cá nhân
- KẾT THÚC: Câu hỏi mở để tiếp tục hỗ trợ

HÃY TRẢ LỜI:"""

        # Prompt phân tích khách hàng
        self.analysis_prompt = """Phân tích thông tin khách hàng sau:

TÊN: {customer_name}
GIỚI THIỆU: {customer_introduction}

Hãy trả về JSON format:
{{
    "title": "Tổng giám đốc/Giám đốc/Trưởng phòng/Nhân viên/Cá nhân",
    "company_type": "Công nghệ/Truyền thông/Sản xuất/Dịch vụ/...",
    "seniority_level": "C-level/Manager/Staff/Individual",
    "industry_focus": "Mô tả ngắn gọn ngành/lĩnh vực",
    "addressing": "Anh/Chị",
    "tone_recommendation": "formal/professional/friendly"
}}

CHỈ TRẢ VỀ JSON, KHÔNG GIẢI THÍCH THÊM."""

    def analyze_customer_profile(
            self,
            customer_name: str,
            customer_introduction: str
    ) -> Dict[str, str]:
        """
        Phân tích profile khách hàng để xác định cách xưng hô và tone

        Args:
            customer_name: Tên khách hàng
            customer_introduction: Giới thiệu về khách hàng

        Returns:
            Dict chứa thông tin phân tích
        """
        try:
            # Nếu không có thông tin → default
            if not customer_name and not customer_introduction:
                return {
                    "title": "Quý khách",
                    "company_type": "Không xác định",
                    "seniority_level": "Individual",
                    "industry_focus": "Chung",
                    "addressing": "Anh/Chị",
                    "tone_recommendation": "professional"
                }

            # Gọi LLM để phân tích
            prompt = self.analysis_prompt.format(
                customer_name=customer_name or "Không cung cấp",
                customer_introduction=customer_introduction or "Không cung cấp"
            )

            analysis_result = llm_model.invoke(
                prompt,
                temperature=0.1,
                max_tokens=200
            )

            # Parse JSON
            import json
            import re

            # Tìm JSON block
            json_match = re.search(r'\{[^}]+\}', analysis_result, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group(0))
                logger.info(f"✅ Customer analysis: {analysis_data}")
                return analysis_data

            # Fallback nếu parse fail
            logger.warning("Failed to parse customer analysis, using defaults")
            return self._extract_basic_info(customer_name, customer_introduction)

        except Exception as e:
            logger.error(f"Error analyzing customer profile: {e}")
            return self._extract_basic_info(customer_name, customer_introduction)

    def _extract_basic_info(
            self,
            customer_name: str,
            customer_introduction: str
    ) -> Dict[str, str]:
        """Fallback: phân tích đơn giản bằng pattern matching"""
        intro_lower = (customer_introduction or "").lower()

        # Detect title
        title = "Quý khách"
        addressing = "Anh/Chị"
        seniority = "Individual"
        tone = "professional"

        if any(x in intro_lower for x in ["tổng giám đốc", "tổng gd", "ceo"]):
            title = "Tổng giám đốc"
            seniority = "C-level"
            tone = "formal"
        elif any(x in intro_lower for x in ["giám đốc", "gd", "director"]):
            title = "Giám đốc"
            seniority = "C-level"
            tone = "formal"
        elif any(x in intro_lower for x in ["trưởng phòng", "tp", "manager"]):
            title = "Trưởng phòng"
            seniority = "Manager"
            tone = "professional"
        elif any(x in intro_lower for x in ["nhân viên", "nv", "staff"]):
            title = "Nhân viên"
            seniority = "Staff"
            tone = "friendly"

        # Detect industry
        industry = "Không xác định"
        if any(x in intro_lower for x in ["công nghệ", "technology", "tech", "cntt"]):
            industry = "Công nghệ thông tin"
        elif any(x in intro_lower for x in ["truyền thông", "media", "marketing"]):
            industry = "Truyền thông & Marketing"
        elif any(x in intro_lower for x in ["sản xuất", "manufacturing"]):
            industry = "Sản xuất"

        return {
            "title": title,
            "company_type": industry,
            "seniority_level": seniority,
            "industry_focus": industry,
            "addressing": addressing,
            "tone_recommendation": tone
        }

    def _format_history(self, history: list, max_turns: int = 2) -> str:
        """Format lịch sử hội thoại"""
        if not history:
            return "Không có lịch sử"

        recent = history[-(max_turns * 2):] if len(history) > max_turns * 2 else history

        lines = []
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                lines.append(f"Khách hàng: {content}")
            elif role == "assistant":
                lines.append(f"Trợ lý: {content}")

        return "\n".join(lines) if lines else "Không có lịch sử"

    async def personalize_streaming(
            self,
            original_answer: str,
            question: str,
            customer_name: str = "",
            customer_introduction: str = "",
            history: list = None
    ) -> AsyncIterator[str]:
        """
        Streaming personalization - cá nhân hóa câu trả lời

        Args:
            original_answer: Câu trả lời gốc từ RAG system
            question: Câu hỏi của khách hàng
            customer_name: Tên khách hàng
            customer_introduction: Giới thiệu khách hàng
            history: Lịch sử hội thoại

        Yields:
            Chunks của câu trả lời đã cá nhân hóa
        """
        try:
            logger.info("🎭 Starting personalization streaming...")
            logger.info(f"   Customer: {customer_name}")
            logger.info(f"   Introduction: {customer_introduction[:50]}...")

            # Phân tích profile khách hàng
            customer_analysis = self.analyze_customer_profile(
                customer_name,
                customer_introduction
            )

            # Format analysis thành text
            analysis_text = f"""
- Chức danh: {customer_analysis.get('title')}
- Cấp độ: {customer_analysis.get('seniority_level')}
- Lĩnh vực: {customer_analysis.get('industry_focus')}
- Xưng hô: {customer_analysis.get('addressing')}
- Tone: {customer_analysis.get('tone_recommendation')}
"""

            # Format history
            history_text = self._format_history(history or [])

            # Tạo prompt
            prompt = self.personalization_prompt.format(
                customer_name=customer_name or "Quý khách",
                customer_introduction=customer_introduction or "Không có thông tin",
                customer_analysis=analysis_text,
                question=question,
                history=history_text,
                original_answer=original_answer
            )

            logger.info("🚀 Streaming personalized answer...")

            # Stream từ LLM
            chunk_count = 0
            async for chunk in llm_model.astream(prompt):
                if chunk:
                    chunk_count += 1
                    logger.debug(f"Personalization chunk #{chunk_count}: {chunk[:30]}...")
                    yield chunk

            logger.info(f"✅ Personalization completed: {chunk_count} chunks")

        except Exception as e:
            logger.error(f"❌ Personalization streaming error: {e}", exc_info=True)
            # Fallback: trả về câu trả lời gốc
            yield f"\n\nThưa {customer_name or 'Anh/Chị'},\n\n"
            yield original_answer

    def personalize(
            self,
            original_answer: str,
            question: str,
            customer_name: str = "",
            customer_introduction: str = "",
            history: list = None
    ) -> str:
        """
        Non-streaming personalization

        Args:
            original_answer: Câu trả lời gốc
            question: Câu hỏi
            customer_name: Tên khách hàng
            customer_introduction: Giới thiệu
            history: Lịch sử

        Returns:
            Câu trả lời đã cá nhân hóa
        """
        try:
            # Phân tích profile
            customer_analysis = self.analyze_customer_profile(
                customer_name,
                customer_introduction
            )

            analysis_text = f"""
- Chức danh: {customer_analysis.get('title')}
- Cấp độ: {customer_analysis.get('seniority_level')}
- Lĩnh vực: {customer_analysis.get('industry_focus')}
- Xưng hô: {customer_analysis.get('addressing')}
- Tone: {customer_analysis.get('tone_recommendation')}
"""

            history_text = self._format_history(history or [])

            prompt = self.personalization_prompt.format(
                customer_name=customer_name or "Quý khách",
                customer_introduction=customer_introduction or "Không có thông tin",
                customer_analysis=analysis_text,
                question=question,
                history=history_text,
                original_answer=original_answer
            )

            # Gọi LLM
            personalized_answer = llm_model.invoke(
                prompt,
                temperature=0.3,
                max_tokens=1500
            )

            logger.info("✅ Personalization completed (non-streaming)")
            return personalized_answer

        except Exception as e:
            logger.error(f"❌ Personalization error: {e}")
            # Fallback
            return f"Thưa {customer_name or 'Anh/Chị'},\n\n{original_answer}"