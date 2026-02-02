# RAG_Core/workflow/personalized_rag_workflow.py

from typing import Dict, Any, List, AsyncIterator
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from agents.supervisor import SupervisorAgent
from agents.retriever_agent import RetrieverAgent
from agents.grader_agent import GraderAgent
from agents.reporter_agent import ReporterAgent

# Import 2 agent mới
from agents.personalization_faq_agent import PersonalizationFAQAgent
from agents.personalization_generator_agent import PersonalizationGeneratorAgent

from agents.base_agent import (
    StreamingChatterAgent,
    StreamingOtherAgent
)

# NEW: Import personalized not enough info agent
from agents.personalization_not_enough_info_agent import PersonalizationNotEnoughInfoAgent

from services.document_url_service import document_url_service

logger = logging.getLogger(__name__)


class PersonalizedChatbotState(TypedDict):
    question: str
    original_question: str
    history: List[Dict[str, str]]
    is_followup: bool
    context_summary: str
    relevant_context: str
    current_agent: str
    documents: List[Dict[str, Any]]
    qualified_documents: List[Dict[str, Any]]
    references: List[Dict[str, Any]]
    answer: str
    status: str
    iteration_count: int
    supervisor_classification: Dict[str, Any]
    faq_result: Dict[str, Any]
    retriever_result: Dict[str, Any]
    parallel_mode: bool
    # Personalization fields
    customer_name: str
    customer_introduction: str


class PersonalizedRAGWorkflow:
    """
    RAG Workflow với personalization - Sử dụng 2 agent mới
    """

    def __init__(self):
        # Core agents
        self.supervisor = SupervisorAgent()
        self.retriever_agent = RetrieverAgent()
        self.grader_agent = GraderAgent()
        self.reporter_agent = ReporterAgent()

        # NEW: Personalization agents
        self.personalization_faq_agent = PersonalizationFAQAgent()
        self.personalization_generator_agent = PersonalizationGeneratorAgent()

        # Streaming-enabled agents
        self.chatter_agent = StreamingChatterAgent()
        self.other_agent = StreamingOtherAgent()

        # NEW: Personalized agent
        self.not_enough_info_agent = PersonalizationNotEnoughInfoAgent()

        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="PersonalizedRAG")
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Tạo workflow graph"""
        workflow = StateGraph(PersonalizedChatbotState)

        # Add nodes
        workflow.add_node("parallel_execution", self._parallel_execution_node)
        workflow.add_node("decision_router", self._decision_router_node)
        workflow.add_node("grader", self._grader_node)
        workflow.add_node("personalized_generator", self._personalized_generator_node)
        workflow.add_node("not_enough_info", self._not_enough_info_node)
        workflow.add_node("chatter", self._chatter_node)
        workflow.add_node("reporter", self._reporter_node)
        workflow.add_node("other", self._other_node)

        workflow.set_entry_point("parallel_execution")
        workflow.add_edge("parallel_execution", "decision_router")

        workflow.add_conditional_edges(
            "decision_router",
            self._route_after_decision,
            {
                "GRADER": "grader",
                "CHATTER": "chatter",
                "REPORTER": "reporter",
                "OTHER": "other",
                "end": "__end__"
            }
        )

        workflow.add_conditional_edges(
            "grader",
            self._route_next_agent,
            {
                "PERSONALIZED_GENERATOR": "personalized_generator",
                "NOT_ENOUGH_INFO": "not_enough_info"
            }
        )

        workflow.add_edge("personalized_generator", "__end__")
        workflow.add_edge("not_enough_info", "__end__")
        workflow.add_edge("chatter", "__end__")
        workflow.add_edge("reporter", "__end__")
        workflow.add_edge("other", "__end__")

        return workflow.compile()

    def _enrich_references_with_urls(self, references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich references with URLs"""
        try:
            if not references:
                return []

            enriched = document_url_service.enrich_references_with_urls(references)
            urls_added = sum(1 for ref in enriched if ref.get('url'))
            if urls_added > 0:
                logger.info(f"🔗 Enriched {urls_added}/{len(references)} references with URLs")
            return enriched

        except Exception as e:
            logger.error(f"Error enriching references: {e}")
            return references

    # ================================================================
    # PARALLEL EXECUTION
    # ================================================================

    def _parallel_execution_node(self, state: PersonalizedChatbotState) -> PersonalizedChatbotState:
        """
        Parallel execution với personalized FAQ agent
        """
        question = state["question"]
        history = state.get("history", [])
        customer_name = state.get("customer_name", "")
        customer_introduction = state.get("customer_introduction", "")

        skip_faq = state.get("streaming_mode", False)

        logger.info("🚀 Starting personalized parallel execution")
        logger.info(f"   Customer: {customer_name}")
        if skip_faq:
            logger.info("⏭️  Skipping FAQ (streaming mode)")

        # Step 1: Supervisor
        supervisor_result = self._get_result_with_timeout(
            self.executor.submit(self._safe_execute_supervisor, question, history),
            timeout=20,
            default={
                "agent": "FAQ",
                "contextualized_question": question,
                "is_followup": False,
                "context_summary": ""
            },
            name="Supervisor"
        )

        context_summary = supervisor_result.get("context_summary", "")
        is_followup = supervisor_result.get("is_followup", False)
        contextualized_question = supervisor_result.get("contextualized_question", question)

        logger.info(
            f"📋 Supervisor: agent={supervisor_result.get('agent')}, "
            f"follow-up={is_followup}\n"
            f"   Original: {question[:60]}\n"
            f"   Contextualized: {contextualized_question[:60]}"
        )

        # Step 2: FAQ + RETRIEVER in parallel
        if skip_faq:
            faq_result = {
                "status": "SKIPPED",
                "answer": "",
                "references": [],
                "message": "FAQ skipped for streaming mode"
            }
        else:
            # Use PERSONALIZATION FAQ AGENT
            future_faq = self.executor.submit(
                self._safe_execute_personalized_faq,
                contextualized_question,
                customer_name,
                customer_introduction,
                is_followup,
                context_summary
            )

            faq_result = self._get_result_with_timeout(
                future_faq,
                timeout=10,
                default={"status": "ERROR", "answer": "", "references": []},
                name="Personalized FAQ"
            )

            if faq_result.get("references"):
                faq_result["references"] = self._enrich_references_with_urls(
                    faq_result["references"]
                )

        # RETRIEVER
        future_retriever = self.executor.submit(
            self._safe_execute_retriever,
            question,
            contextualized_question,
            is_followup
        )

        retriever_result = self._get_result_with_timeout(
            future_retriever,
            timeout=10,
            default={"status": "ERROR", "documents": []},
            name="RETRIEVER"
        )

        state["supervisor_classification"] = supervisor_result
        state["question"] = contextualized_question
        state["original_question"] = question
        state["is_followup"] = is_followup
        state["context_summary"] = context_summary
        state["faq_result"] = faq_result
        state["retriever_result"] = retriever_result
        state["parallel_mode"] = True

        logger.info(
            f"✅ Personalized parallel execution completed:\n"
            f"  - FAQ: {faq_result.get('status')}\n"
            f"  - RETRIEVER: {retriever_result.get('status')}"
        )

        return state

    def _get_result_with_timeout(self, future, timeout: float, default: Dict, name: str) -> Dict:
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            logger.warning(f"⏱️ {name} timeout, using fallback")
            return default
        except Exception as e:
            logger.error(f"❌ {name} error: {e}")
            return default

    def _safe_execute_supervisor(self, question: str, history: List) -> Dict[str, Any]:
        try:
            return self.supervisor.classify_request(question, history)
        except Exception as e:
            logger.error(f"❌ Supervisor error: {e}")
            return {
                "agent": "FAQ",
                "contextualized_question": question,
                "context_summary": "",
                "is_followup": False
            }

    def _safe_execute_personalized_faq(
            self,
            question: str,
            customer_name: str,
            customer_introduction: str,
            is_followup: bool = False,
            context_summary: str = ""
    ) -> Dict[str, Any]:
        """Execute PERSONALIZED FAQ agent"""
        try:
            return self.personalization_faq_agent.process(
                question=question,
                customer_name=customer_name,
                customer_introduction=customer_introduction,
                is_followup=is_followup,
                context=context_summary
            )
        except Exception as e:
            logger.error(f"Personalized FAQ error: {e}")
            return {
                "status": "ERROR",
                "answer": "",
                "references": [],
                "next_agent": "RETRIEVER"
            }

    def _safe_execute_retriever(
            self,
            original_question: str,
            contextualized_question: str,
            is_followup: bool = False
    ) -> Dict[str, Any]:
        """Retriever execution"""
        try:
            return self.retriever_agent.process(
                question=original_question,
                contextualized_question=contextualized_question,
                is_followup=is_followup
            )
        except Exception as e:
            logger.error(f"RETRIEVER error: {e}")
            return {
                "status": "ERROR",
                "documents": [],
                "next_agent": "NOT_ENOUGH_INFO"
            }

    def _decision_router_node(self, state: PersonalizedChatbotState) -> PersonalizedChatbotState:
        """Router sau parallel execution"""
        supervisor_agent = state.get("supervisor_classification", {}).get("agent", "FAQ")
        faq_result = state.get("faq_result", {})
        retriever_result = state.get("retriever_result", {})

        logger.info(f"🤔 Decision Router: Supervisor={supervisor_agent}")

        if supervisor_agent in ["CHATTER", "REPORTER", "OTHER"]:
            state["current_agent"] = supervisor_agent
            return state

        # Accept FAQ if SUCCESS
        if faq_result.get("status") == "SUCCESS":
            logger.info("→ Personalized FAQ has answer")
            state["status"] = faq_result["status"]
            state["answer"] = faq_result.get("answer", "")
            state["references"] = faq_result.get("references", [])
            state["current_agent"] = "end"
            return state

        if retriever_result.get("documents"):
            logger.info("→ RETRIEVER → GRADER")
            state["documents"] = retriever_result.get("documents", [])
            state["status"] = retriever_result.get("status", "SUCCESS")
            state["current_agent"] = "GRADER"
            return state

        state["current_agent"] = "NOT_ENOUGH_INFO"
        return state

    def _grader_node(self, state: PersonalizedChatbotState) -> PersonalizedChatbotState:
        """Grader node"""
        try:
            question = state["question"]
            original_question = state.get("original_question", question)
            documents = state.get("documents", [])
            is_followup = state.get("is_followup", False)

            logger.info(f"📊 Grader: Processing {len(documents)} documents")

            result = self.grader_agent.process(
                question=original_question,
                documents=documents,
                contextualized_question=question,
                is_followup=is_followup
            )

            if result.get("references"):
                result["references"] = self._enrich_references_with_urls(
                    result["references"]
                )

            state["status"] = result["status"]
            state["qualified_documents"] = result.get("qualified_documents", [])
            state["references"] = result.get("references", [])

            # Route to PERSONALIZED_GENERATOR instead of GENERATOR
            if result.get("next_agent") == "GENERATOR":
                state["current_agent"] = "PERSONALIZED_GENERATOR"
            else:
                state["current_agent"] = result.get("next_agent", "NOT_ENOUGH_INFO")

            logger.info(f"📊 Grader result: {result['status']} → {state['current_agent']}")
            return state

        except Exception as e:
            logger.error(f"❌ Grader error: {e}", exc_info=True)
            state["current_agent"] = "NOT_ENOUGH_INFO"
            return state

    def _personalized_generator_node(self, state: PersonalizedChatbotState) -> PersonalizedChatbotState:
        """PERSONALIZED GENERATOR node"""
        try:
            logger.info("🎭 Using Personalized Generator")

            result = self.personalization_generator_agent.process(
                question=state["question"],
                documents=state.get("qualified_documents", []),
                customer_name=state.get("customer_name", ""),
                customer_introduction=state.get("customer_introduction", ""),
                references=state.get("references", []),
                history=state.get("history", []),
                is_followup=state.get("is_followup", False),
                context_summary=state.get("context_summary", "")
            )

            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state

        except Exception as e:
            logger.error(f"Personalized Generator error: {e}")
            state["answer"] = "Lỗi tạo câu trả lời"
            state["current_agent"] = "end"
            return state

    def _not_enough_info_node(self, state: PersonalizedChatbotState) -> PersonalizedChatbotState:
        """NOT ENOUGH INFO node với personalization"""
        try:
            logger.info("🎭 Using Personalized Not Enough Info")

            result = self.not_enough_info_agent.process(
                question=state["question"],
                customer_name=state.get("customer_name", ""),
                customer_introduction=state.get("customer_introduction", "")
            )

            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state

        except Exception as e:
            logger.error(f"Personalized Not Enough Info error: {e}")

            # Fallback with customer name if available
            customer_name = state.get("customer_name", "")
            greeting = f"Thưa Anh/Chị {customer_name}" if customer_name else "Xin chào"

            state["answer"] = f"""{greeting},

    Không tìm thấy thông tin phù hợp trong hệ thống.

    Để được hỗ trợ tốt nhất, vui lòng liên hệ hotline: {settings.SUPPORT_PHONE}"""
            state["current_agent"] = "end"
            return state

    def _chatter_node(self, state: PersonalizedChatbotState) -> PersonalizedChatbotState:
        try:
            result = self.chatter_agent.process(state["question"], state.get("history", []))
            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state
        except Exception as e:
            logger.error(f"Chatter error: {e}")
            state["answer"] = "Tôi hiểu cảm xúc của bạn"
            return state

    def _reporter_node(self, state: PersonalizedChatbotState) -> PersonalizedChatbotState:
        try:
            result = self.reporter_agent.process(state["question"])
            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state
        except Exception as e:
            logger.error(f"Reporter error: {e}")
            state["answer"] = "Hệ thống đang bảo trì"
            return state

    def _other_node(self, state: PersonalizedChatbotState) -> PersonalizedChatbotState:
        try:
            result = self.other_agent.process(state["question"])
            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state
        except Exception as e:
            logger.error(f"Other error: {e}")
            state["answer"] = "Đây không phải tác vụ của tôi"
            return state

    def _route_after_decision(self, state: PersonalizedChatbotState) -> str:
        return state.get("current_agent", "end")

    def _route_next_agent(self, state: PersonalizedChatbotState) -> str:
        return state.get("current_agent", "end")

    # ================================================================
    # RUN METHODS
    # ================================================================

    def run_with_personalization(
            self,
            question: str,
            history: List[Dict[str, str]] = None,
            customer_name: str = "",
            customer_introduction: str = ""
    ) -> Dict[str, Any]:
        """
        Non-streaming run với personalization
        """
        try:
            initial_state = self._create_initial_state(
                question,
                history,
                customer_name,
                customer_introduction,
                streaming_mode=False
            )

            logger.info(f"🚀 Personalized workflow start: {question[:100]}")
            logger.info(f"   Customer: {customer_name}")

            final_state = self.workflow.invoke(initial_state)

            return {
                "answer": final_state.get("answer", "Lỗi xử lý"),
                "references": final_state.get("references", []),
                "status": final_state.get("status", "ERROR"),
                "personalized": bool(customer_name or customer_introduction),
                "customer_name": customer_name
            }

        except Exception as e:
            logger.error(f"❌ Workflow error: {e}", exc_info=True)
            return {
                "answer": "Xin lỗi, hệ thống gặp sự cố.",
                "references": [],
                "status": "ERROR",
                "personalized": False
            }

    async def run_with_personalization_streaming(
            self,
            question: str,
            history: List[Dict[str, str]] = None,
            customer_name: str = "",
            customer_introduction: str = ""
    ) -> Dict[str, Any]:
        """
        Streaming run với personalization
        """
        try:
            logger.info(f"🚀 Personalized streaming workflow")
            logger.info(f"   Question: {question[:100]}")
            logger.info(f"   Customer: {customer_name}")

            initial_state = self._create_initial_state(
                question,
                history,
                customer_name,
                customer_introduction,
                streaming_mode=True
            )

            state = self._parallel_execution_node(initial_state)
            state = self._decision_router_node(state)

            current_agent = state.get("current_agent")
            supervisor_agent = state.get("supervisor_classification", {}).get("agent")

            logger.info(f"📍 Routed to: {current_agent}")

            # FAQ - Check confidence first
            if supervisor_agent == "FAQ":
                logger.info("🔍 FAQ classified - checking confidence...")

                from tools.vector_search import search_faq, rerank_faq
                from config.settings import settings

                faq_results = search_faq.invoke({"query": state["question"]})

                if not faq_results or "error" in str(faq_results):
                    logger.warning("❌ FAQ search failed → GRADER")
                    current_agent = "GRADER"
                else:
                    filtered_faqs = [
                        faq for faq in faq_results
                        if faq.get("similarity_score", 0) >= settings.FAQ_VECTOR_THRESHOLD
                    ]

                    if not filtered_faqs:
                        logger.info("⚠️  No FAQ above threshold → GRADER")
                        current_agent = "GRADER"
                    else:
                        reranked_faqs = rerank_faq.invoke({
                            "query": state["question"],
                            "faq_results": filtered_faqs
                        })

                        if not reranked_faqs:
                            logger.warning("❌ FAQ rerank failed → GRADER")
                            current_agent = "GRADER"
                        else:
                            best_score = reranked_faqs[0].get("rerank_score", 0)
                            logger.info(f"📊 FAQ best score: {best_score:.3f}")

                            if best_score >= settings.FAQ_RERANK_THRESHOLD:
                                logger.info("✅ FAQ CONFIDENT - Streaming personalized answer")

                                return {
                                    "answer_stream": self.personalization_faq_agent.process_streaming(
                                        question=state["question"],
                                        reranked_faqs=reranked_faqs,
                                        customer_name=customer_name,
                                        customer_introduction=customer_introduction,
                                        is_followup=state.get("is_followup", False),
                                        context=state.get("context_summary", "")
                                    ),
                                    "references": [
                                        {
                                            "document_id": reranked_faqs[0].get("faq_id"),
                                            "type": "FAQ",
                                            "description": reranked_faqs[0].get("question", ""),
                                            "rerank_score": round(best_score, 4)
                                        }
                                    ],
                                    "status": "STREAMING",
                                    "personalized": True
                                }
                            else:
                                logger.info(f"⚠️  FAQ not confident → GRADER")
                                current_agent = "GRADER"

            # GRADER → PERSONALIZED GENERATOR
            if current_agent == "GRADER":
                state = self._grader_node(state)

                if state.get("current_agent") == "PERSONALIZED_GENERATOR":
                    logger.info("✅ STREAMING: PERSONALIZED GENERATOR")
                    return {
                        "answer_stream": self.personalization_generator_agent.process_streaming(
                            question=state["question"],
                            documents=state.get("qualified_documents", []),
                            customer_name=customer_name,
                            customer_introduction=customer_introduction,
                            references=state.get("references", []),
                            history=history or [],
                            is_followup=state.get("is_followup", False),
                            context_summary=state.get("context_summary", "")
                        ),
                        "references": state.get("references", []),
                        "status": "STREAMING",
                        "personalized": True
                    }
                else:
                    logger.info("✅ STREAMING: PERSONALIZED NOT_ENOUGH_INFO")

                    return {
                        "answer_stream": self.not_enough_info_agent.process_streaming(
                            question=state["question"],
                            customer_name=state.get("customer_name", ""),
                            customer_introduction=state.get("customer_introduction", "")
                        ),
                        "references": [{"document_id": "llm_knowledge", "type": "GENERAL_KNOWLEDGE"}],
                        "status": "STREAMING",
                        "personalized": True  # Changed to True
                    }

            # Other agents (CHATTER, OTHER, REPORTER)
            elif current_agent == "CHATTER":
                logger.info("✅ STREAMING: CHATTER")
                from config.settings import settings

                return {
                    "answer_stream": self.chatter_agent.process_streaming(
                        question=state["question"],
                        history=state.get("history", []),
                        support_phone=settings.SUPPORT_PHONE
                    ),
                    "references": [{"document_id": "support_contact", "type": "SUPPORT"}],
                    "status": "STREAMING",
                    "personalized": False
                }

            elif current_agent == "OTHER":
                logger.info("✅ STREAMING: OTHER")
                from config.settings import settings

                return {
                    "answer_stream": self.other_agent.process_streaming(
                        question=state["question"],
                        support_phone=settings.SUPPORT_PHONE
                    ),
                    "references": [],
                    "status": "STREAMING",
                    "personalized": False
                }

            elif current_agent == "REPORTER":
                logger.info("📋 REPORTER")
                state = self._reporter_node(state)
                answer_text = state.get("answer", "")

                async def reporter_generator():
                    words = answer_text.split()
                    for word in words:
                        yield word + " "
                        await asyncio.sleep(0.01)

                return {
                    "answer_stream": reporter_generator(),
                    "references": state.get("references", []),
                    "status": state.get("status", "SUCCESS"),
                    "personalized": False
                }

            else:
                logger.warning(f"Unknown agent: {current_agent}")

                async def error_generator():
                    yield "Xin lỗi, không thể xử lý yêu cầu này."

                return {
                    "answer_stream": error_generator(),
                    "references": [],
                    "status": "ERROR",
                    "personalized": False
                }

        except Exception as e:
            logger.error(f"❌ Streaming workflow error: {e}", exc_info=True)

            async def error_generator():
                yield "Xin lỗi, hệ thống gặp sự cố."

            return {
                "answer_stream": error_generator(),
                "references": [],
                "status": "ERROR",
                "personalized": False
            }

    def _create_initial_state(
            self,
            question: str,
            history: List = None,
            customer_name: str = "",
            customer_introduction: str = "",
            streaming_mode: bool = False
    ) -> PersonalizedChatbotState:
        """Create initial state"""
        return PersonalizedChatbotState(
            question=question,
            original_question=question,
            history=history or [],
            is_followup=False,
            context_summary="",
            relevant_context="",
            current_agent="parallel_execution",
            documents=[],
            qualified_documents=[],
            references=[],
            answer="",
            status="",
            iteration_count=0,
            supervisor_classification={},
            faq_result={},
            retriever_result={},
            parallel_mode=False,
            streaming_mode=streaming_mode,
            customer_name=customer_name,
            customer_introduction=customer_introduction
        )

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True, timeout=5)
