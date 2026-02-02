# RAG_Core/agents/personalization_not_enough_info_agent.py

from typing import Dict, Any, List, AsyncIterator
from models.llm_model import llm_model
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class PersonalizationNotEnoughInfoAgent:
    """
    NotEnoughInfoAgent với personalization - Cá nhân hóa câu trả lời khi không đủ thông tin

    Chức năng:
    - Trả lời dựa trên kiến thức chung khi không có đủ dữ liệu
    - Cá nhân hóa theo thông tin khách hàng (tên, chức danh, lĩnh vực)
    - Điều chỉnh tone phù hợp với vị trí khách hàng
    - Hỗ trợ streaming
    """

    def __init__(self):
        self.name = "PERSONALIZATION_NOT_ENOUGH_INFO"

        # Personalized prompt
        self.personalized_prompt = """Bạn là trợ lý ảo Onetouch - chuyên gia đào tạo kỹ năng số cho người dân và doanh nghiệp.

THÔNG TIN KHÁCH HÀNG:
- Tên: {customer_name}
- Giới thiệu: {customer_introduction}
- Phân tích: {customer_analysis}

TÌNH HUỐNG: Không có đủ dữ liệu trong hệ thống để trả lời chính xác câu hỏi này.

CÂU HỎI CỦA KHÁCH HÀNG: "{question}"

YÊU CẦU TRẢ LỜI:

1. **Xưng hô phù hợp**:
   - Nếu là Giám đốc/Tổng giám đốc/CEO: "Thưa Anh/Chị {customer_name}"
   - Nếu là Manager/Trưởng phòng: "Thưa Anh/Chị {customer_name}"
   - Nếu là nhân viên/cá nhân: "Bạn {customer_name}"
   - Nếu không rõ: "Thưa Anh/Chị {customer_name}"

2. **Cấu trúc câu trả lời** (BẮT BUỘC NGẮN GỌN - tối đa 3-4 câu):
   a) MỞ ĐẦU (1 câu):
      "Thưa Anh/Chị {customer_name}, dựa trên tổng hợp từ các nguồn thông tin, bạn có thể tham khảo như sau:"

   b) NỘI DUNG CHÍNH (1-2 câu):
      - Cung cấp thông tin mang tính tham khảo chung
      - KHÔNG suy đoán chi tiết kỹ thuật
      - KHÔNG phân tích dài dòng
      - Liên kết với lĩnh vực/ngành nghề của khách hàng (nếu có)

   c) KẾT THÚC (1 câu):
      "Để được tư vấn chính xác hơn, Anh/Chị vui lòng liên hệ hotline {support_phone}."

3. **Cá nhân hóa nội dung**:
   - Nếu biết lĩnh vực: Đưa ví dụ phù hợp (công nghệ, truyền thông, sản xuất...)
   - Điều chỉnh độ kỹ thuật theo vị trí:
     * Lãnh đạo → Tổng quan, chiến lược
     * Quản lý → Giải pháp thực tế
     * Nhân viên → Dễ hiểu, ứng dụng

4. **Tone phù hợp**:
   - Lãnh đạo cấp cao: Tôn trọng, chuyên nghiệp
   - Quản lý: Thân thiện, hỗ trợ
   - Nhân viên: Gần gũi, dễ tiếp cận

5. **YÊU CẦU ĐẶC BIỆT**:
   - NGẮN GỌN (tối đa 3-4 câu)
   - KHÔNG kể ví dụ dài
   - KHÔNG giải thích chi tiết
   - BẮT ĐẦU bằng lời xưng hô phù hợp
   - KẾT THÚC bằng đề nghị liên hệ hotline

Hãy trả lời:"""

    def _analyze_customer_profile(
            self,
            customer_name: str,
            customer_introduction: str
    ) -> str:
        """
        Phân tích nhanh profile khách hàng
        """
        try:
            intro_lower = (customer_introduction or "").lower()

            # Detect title
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
            else:
                title = "Quý khách"
                seniority = "Individual"
                tone = "professional"

            # Detect industry
            if any(x in intro_lower for x in ["công nghệ", "technology", "tech", "cntt"]):
                industry = "Công nghệ thông tin"
            elif any(x in intro_lower for x in ["truyền thông", "media", "marketing"]):
                industry = "Truyền thông & Marketing"
            elif any(x in intro_lower for x in ["sản xuất", "manufacturing"]):
                industry = "Sản xuất"
            else:
                industry = "Không xác định"

            return f"""
- Chức danh: {title}
- Cấp độ: {seniority}
- Lĩnh vực: {industry}
- Tone khuyến nghị: {tone}
"""
        except Exception as e:
            logger.error(f"Error analyzing profile: {e}")
            return "- Chức danh: Quý khách\n- Cấp độ: Individual\n- Tone: professional"

    def process(
            self,
            question: str,
            customer_name: str = "",
            customer_introduction: str = "",
            **kwargs
    ) -> Dict[str, Any]:
        """
        Non-streaming process với personalization

        Args:
            question: Câu hỏi
            customer_name: Tên khách hàng
            customer_introduction: Giới thiệu về khách hàng
        """
        try:
            logger.info("🎭 Personalized Not Enough Info (non-streaming)")
            logger.info(f"   Customer: {customer_name}")

            # Analyze customer profile
            customer_analysis = self._analyze_customer_profile(
                customer_name,
                customer_introduction
            )

            # Create personalized prompt
            prompt = self.personalized_prompt.format(
                customer_name=customer_name or "Quý khách",
                customer_introduction=customer_introduction or "Không có thông tin",
                customer_analysis=customer_analysis,
                question=question,
                support_phone=settings.SUPPORT_PHONE
            )

            # Generate answer
            logger.info("🤖 Generating personalized answer (not enough info)...")

            answer = llm_model.invoke(
                prompt,
                temperature=0.2,  # Thấp để tuân thủ format
                top_p=0.7,
                max_tokens=150,  # Giới hạn độ dài
                frequency_penalty=0.5,
                presence_penalty=0.0
            )

            logger.info("✅ Personalized answer generated")

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": [
                    {
                        "document_id": "llm_knowledge",
                        "type": "GENERAL_KNOWLEDGE"
                    }
                ],
                "personalized": True,
                "customer_name": customer_name,
                "next_agent": "end"
            }

        except Exception as e:
            logger.error(f"❌ Personalized Not Enough Info error: {e}")

            # Fallback answer
            fallback_greeting = f"Thưa Anh/Chị {customer_name}" if customer_name else "Xin chào"

            return {
                "status": "ERROR",
                "answer": f"""{fallback_greeting},

Xin lỗi, hệ thống gặp lỗi khi xử lý câu hỏi của bạn.

Để được hỗ trợ tốt nhất, vui lòng liên hệ hotline: {settings.SUPPORT_PHONE}

Cảm ơn bạn!""",
                "references": [],
                "personalized": bool(customer_name),
                "next_agent": "end"
            }

    async def process_streaming(
            self,
            question: str,
            customer_name: str = "",
            customer_introduction: str = "",
            **kwargs
    ) -> AsyncIterator[str]:
        """
        Streaming process với personalization

        Args:
            question: Câu hỏi
            customer_name: Tên khách hàng
            customer_introduction: Giới thiệu về khách hàng
        """
        try:
            logger.info("🎭 Personalized Not Enough Info streaming")
            logger.info(f"   Customer: {customer_name}")

            # Analyze customer profile
            customer_analysis = self._analyze_customer_profile(
                customer_name,
                customer_introduction
            )

            # Create personalized prompt
            prompt = self.personalized_prompt.format(
                customer_name=customer_name or "Quý khách",
                customer_introduction=customer_introduction or "Không có thông tin",
                customer_analysis=customer_analysis,
                question=question,
                support_phone=settings.SUPPORT_PHONE
            )

            logger.info("🚀 Streaming personalized answer...")

            # Stream from LLM
            chunk_count = 0
            async for chunk in llm_model.astream(prompt):
                if chunk:
                    chunk_count += 1
                    logger.debug(f"Not Enough Info chunk #{chunk_count}: {chunk[:30]}...")
                    yield chunk

            logger.info(f"✅ Not Enough Info streaming completed: {chunk_count} chunks")

        except Exception as e:
            logger.error(f"❌ Streaming error: {e}", exc_info=True)

            # Fallback streaming
            fallback_greeting = f"Thưa Anh/Chị {customer_name}" if customer_name else "Xin chào"
            error_message = f"""{fallback_greeting},

Xin lỗi, hệ thống gặp lỗi khi xử lý câu hỏi của bạn.

Để được hỗ trợ tốt nhất, vui lòng liên hệ hotline: {settings.SUPPORT_PHONE}

Cảm ơn bạn!"""

            yield error_message