# RAG_Core/workflow/rag_workflow.py - FIXED CONSISTENT RESULTS

from typing import Dict, Any, List, AsyncIterator
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from agents.supervisor import SupervisorAgent
from agents.faq_agent import FAQAgent
from agents.retriever_agent import RetrieverAgent
from agents.grader_agent import GraderAgent
from agents.generator_agent import GeneratorAgent
from agents.reporter_agent import ReporterAgent

# Import streaming agents
from agents.base_agent import (
    StreamingChatterAgent,
    StreamingOtherAgent,
    StreamingNotEnoughInfoAgent
)

logger = logging.getLogger(__name__)


class ChatbotState(TypedDict):
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


class RAGWorkflow:
    def __init__(self):
        # Initialize all agents
        self.supervisor = SupervisorAgent()
        self.faq_agent = FAQAgent()
        self.retriever_agent = RetrieverAgent()
        self.grader_agent = GraderAgent()
        self.generator_agent = GeneratorAgent()
        self.reporter_agent = ReporterAgent()

        # Streaming-enabled agents
        self.chatter_agent = StreamingChatterAgent()
        self.other_agent = StreamingOtherAgent()
        self.not_enough_info_agent = StreamingNotEnoughInfoAgent()

        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="RAG-Worker")
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Tạo workflow graph"""
        workflow = StateGraph(ChatbotState)

        # Add nodes
        workflow.add_node("parallel_execution", self._parallel_execution_node)
        workflow.add_node("decision_router", self._decision_router_node)
        workflow.add_node("grader", self._grader_node)
        workflow.add_node("generator", self._generator_node)
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
                "GENERATOR": "generator",
                "NOT_ENOUGH_INFO": "not_enough_info"
            }
        )

        workflow.add_edge("generator", "__end__")
        workflow.add_edge("not_enough_info", "__end__")
        workflow.add_edge("chatter", "__end__")
        workflow.add_edge("reporter", "__end__")
        workflow.add_edge("other", "__end__")

        return workflow.compile()

    def _parallel_execution_node(self, state: ChatbotState) -> ChatbotState:
        """
        ✅ FIXED: Chạy song song với ĐÚNG parameters

        Supervisor → Context → FAQ + RETRIEVER (với context)
        """
        question = state["question"]
        history = state.get("history", [])

        logger.info("🚀 Starting parallel execution")

        # ========================================
        # STEP 1: Run Supervisor FIRST (sequential)
        # ========================================
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

        # Extract context từ supervisor
        context_summary = supervisor_result.get("context_summary", "")
        is_followup = supervisor_result.get("is_followup", False)
        contextualized_question = supervisor_result.get("contextualized_question", question)

        logger.info(
            f"📋 Supervisor: agent={supervisor_result.get('agent')}, "
            f"follow-up={is_followup}, has_context={bool(context_summary)}"
        )

        # ========================================
        # STEP 2: Run FAQ + RETRIEVER in parallel (WITH context)
        # ========================================
        future_faq = self.executor.submit(
            self._safe_execute_faq,
            contextualized_question,
            history
        )

        # ✅ FIX: Pass context_summary và is_followup
        future_retriever = self.executor.submit(
            self._safe_execute_retriever,
            question,  # Original question
            context_summary,  # ✅ Context summary
            is_followup  # ✅ Follow-up flag
        )

        faq_result = self._get_result_with_timeout(
            future_faq,
            timeout=10,
            default={"status": "ERROR", "answer": "", "references": []},
            name="FAQ"
        )

        retriever_result = self._get_result_with_timeout(
            future_retriever,
            timeout=10,
            default={"status": "ERROR", "documents": []},
            name="RETRIEVER"
        )

        # ========================================
        # STEP 3: Update state
        # ========================================
        state["supervisor_classification"] = supervisor_result
        state["question"] = contextualized_question
        state["is_followup"] = is_followup
        state["context_summary"] = context_summary
        state["faq_result"] = faq_result
        state["retriever_result"] = retriever_result
        state["parallel_mode"] = True

        logger.info(
            f"✅ Parallel execution completed:\n"
            f"  - FAQ: {faq_result.get('status')}\n"
            f"  - RETRIEVER: {retriever_result.get('status')}, "
            f"docs={len(retriever_result.get('documents', []))}"
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
            return self._fallback_supervisor_classification(question)

    def _fallback_supervisor_classification(self, question: str) -> Dict[str, Any]:
        try:
            agent = self.supervisor._fallback_classify(question)
            return {
                "agent": agent,
                "contextualized_question": question,
                "context_summary": "",
                "is_followup": False
            }
        except Exception:
            return {
                "agent": "FAQ",
                "contextualized_question": question,
                "context_summary": "",
                "is_followup": False
            }

    def _safe_execute_faq(self, question: str, history: List) -> Dict[str, Any]:
        try:
            return self.faq_agent.process(question, is_followup=False, context="")
        except Exception as e:
            logger.error(f"FAQ error: {e}")
            return {"status": "ERROR", "answer": "", "references": [], "next_agent": "RETRIEVER"}

    def _safe_execute_retriever(
            self,
            question: str,
            context_summary: str = "",  # ✅ FIXED: Accept context
            is_followup: bool = False  # ✅ FIXED: Accept follow-up flag
    ) -> Dict[str, Any]:
        """
        ✅ FIXED: Execute retriever với context summary support
        """
        try:
            # Kiểm tra xem retriever_agent.process có nhận context không
            # Nếu không hỗ trợ, chỉ truyền question
            import inspect
            sig = inspect.signature(self.retriever_agent.process)

            if 'context_summary' in sig.parameters:
                # Retriever hỗ trợ context
                return self.retriever_agent.process(
                    question=question,
                    context_summary=context_summary,
                    is_followup=is_followup
                )
            else:
                # Retriever chưa hỗ trợ context - chỉ truyền question
                logger.debug("Retriever doesn't support context, using question only")
                return self.retriever_agent.process(question=question)

        except Exception as e:
            logger.error(f"RETRIEVER error: {e}")
            return {
                "status": "ERROR",
                "documents": [],
                "next_agent": "NOT_ENOUGH_INFO"
            }

    def _decision_router_node(self, state: ChatbotState) -> ChatbotState:
        """Router dựa trên kết quả parallel"""
        supervisor_agent = state.get("supervisor_classification", {}).get("agent", "FAQ")
        faq_result = state.get("faq_result", {})
        retriever_result = state.get("retriever_result", {})

        logger.info(f"🤔 Decision Router: Supervisor={supervisor_agent}")

        if supervisor_agent in ["CHATTER", "REPORTER", "OTHER"]:
            state["current_agent"] = supervisor_agent
            return state

        if faq_result.get("status") == "SUCCESS":
            logger.info("→ FAQ has answer")
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

    def _grader_node(self, state: ChatbotState) -> ChatbotState:
        """
        ✅ FIXED: Grader với context summary
        """
        try:
            question = state["question"]
            documents = state.get("documents", [])
            context_summary = state.get("context_summary", "")
            is_followup = state.get("is_followup", False)

            logger.info(
                f"📊 Grader: Processing {len(documents)} documents "
                f"(follow-up={is_followup})"
            )

            # Kiểm tra xem grader có hỗ trợ context không
            import inspect
            sig = inspect.signature(self.grader_agent.process)

            if 'context_summary' in sig.parameters:
                result = self.grader_agent.process(
                    question=question,
                    documents=documents,
                    context_summary=context_summary,
                    is_followup=is_followup
                )
            else:
                # Grader chưa hỗ trợ context
                result = self.grader_agent.process(
                    question=question,
                    documents=documents
                )

            state["status"] = result["status"]
            state["qualified_documents"] = result.get("qualified_documents", [])
            state["references"] = result.get("references", [])
            state["current_agent"] = result.get("next_agent", "GENERATOR")

            logger.info(
                f"📊 Grader result: {result['status']}, "
                f"qualified={len(result.get('qualified_documents', []))}"
            )

            return state

        except Exception as e:
            logger.error(f"❌ Grader node error: {e}", exc_info=True)
            state["current_agent"] = "NOT_ENOUGH_INFO"
            return state

    def _generator_node(self, state: ChatbotState) -> ChatbotState:
        try:
            result = self.generator_agent.process(
                question=state["question"],
                documents=state.get("qualified_documents", []),
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
            logger.error(f"Generator error: {e}")
            state["answer"] = "Lỗi tạo câu trả lời"
            state["current_agent"] = "end"
            return state

    def _not_enough_info_node(self, state: ChatbotState) -> ChatbotState:
        try:
            result = self.not_enough_info_agent.process(
                state["question"],
                is_followup=state.get("is_followup", False)
            )
            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state
        except Exception as e:
            logger.error(f"Not enough info error: {e}")
            state["answer"] = "Không tìm thấy thông tin"
            return state

    def _chatter_node(self, state: ChatbotState) -> ChatbotState:
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

    def _reporter_node(self, state: ChatbotState) -> ChatbotState:
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

    def _other_node(self, state: ChatbotState) -> ChatbotState:
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

    def _route_after_decision(self, state: ChatbotState) -> str:
        return state.get("current_agent", "end")

    def _route_next_agent(self, state: ChatbotState) -> str:
        return state.get("current_agent", "end")

    def run(self, question: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Non-streaming run"""
        try:
            initial_state = self._create_initial_state(question, history)
            logger.info(f"🚀 Workflow start: {question[:100]}")
            final_state = self.workflow.invoke(initial_state)

            return {
                "answer": final_state.get("answer", "Lỗi xử lý"),
                "references": final_state.get("references", []),
                "status": final_state.get("status", "ERROR")
            }
        except Exception as e:
            logger.error(f"❌ Workflow error: {e}", exc_info=True)
            return {
                "answer": "Xin lỗi, hệ thống gặp sự cố.",
                "references": [],
                "status": "ERROR"
            }

    async def run_with_streaming(
            self,
            question: str,
            history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        ✅ FIXED: TRUE STREAMING với CONSISTENT LOGIC
        """
        try:
            logger.info(f"🚀 Streaming workflow start: {question[:100]}")

            # ✅ FIX: Sử dụng CÙNG logic với run() - qua workflow graph
            initial_state = self._create_initial_state(question, history)
            state = self._parallel_execution_node(initial_state)
            state = self._decision_router_node(state)

            current_agent = state.get("current_agent")
            logger.info(f"📍 Routed to: {current_agent}")

            # FAQ - Direct answer
            if current_agent == "end":
                answer_text = state.get("answer", "")

                async def faq_generator():
                    words = answer_text.split()
                    for word in words:
                        yield word + " "
                        await asyncio.sleep(0.01)

                return {
                    "answer_stream": faq_generator(),
                    "references": state.get("references", []),
                    "status": state.get("status", "SUCCESS")
                }

            # GRADER → GENERATOR or NOT_ENOUGH_INFO
            elif current_agent == "GRADER":
                state = self._grader_node(state)

                if state.get("current_agent") == "GENERATOR":
                    logger.info("✅ TRUE STREAMING: GENERATOR")
                    return {
                        "answer_stream": self.generator_agent.process_streaming(
                            question=state["question"],
                            documents=state.get("qualified_documents", []),
                            references=state.get("references", []),
                            history=history or [],
                            is_followup=state.get("is_followup", False),
                            context_summary=state.get("context_summary", "")
                        ),
                        "references": state.get("references", []),
                        "status": "STREAMING"
                    }
                else:
                    logger.info("✅ TRUE STREAMING: NOT_ENOUGH_INFO")
                    from config.settings import settings

                    return {
                        "answer_stream": self.not_enough_info_agent.process_streaming(
                            question=state["question"],
                            support_phone=settings.SUPPORT_PHONE
                        ),
                        "references": [{"document_id": "llm_knowledge", "type": "GENERAL_KNOWLEDGE"}],
                        "status": "STREAMING"
                    }

            # CHATTER
            elif current_agent == "CHATTER":
                logger.info("✅ TRUE STREAMING: CHATTER")
                from config.settings import settings

                return {
                    "answer_stream": self.chatter_agent.process_streaming(
                        question=state["question"],
                        history=state.get("history", []),
                        support_phone=settings.SUPPORT_PHONE
                    ),
                    "references": [{"document_id": "support_contact", "type": "SUPPORT"}],
                    "status": "STREAMING"
                }

            # OTHER
            elif current_agent == "OTHER":
                logger.info("✅ TRUE STREAMING: OTHER")
                from config.settings import settings

                return {
                    "answer_stream": self.other_agent.process_streaming(
                        question=state["question"],
                        support_phone=settings.SUPPORT_PHONE
                    ),
                    "references": [],
                    "status": "STREAMING"
                }

            # REPORTER
            elif current_agent == "REPORTER":
                logger.info("📋 REPORTER: Static message")
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
                    "status": state.get("status", "SUCCESS")
                }

            # Fallback
            else:
                logger.warning(f"Unknown agent: {current_agent}")

                async def error_generator():
                    yield "Xin lỗi, không thể xử lý yêu cầu này."

                return {
                    "answer_stream": error_generator(),
                    "references": [],
                    "status": "ERROR"
                }

        except Exception as e:
            logger.error(f"❌ Streaming workflow error: {e}", exc_info=True)

            async def error_generator():
                yield "Xin lỗi, hệ thống gặp sự cố."

            return {
                "answer_stream": error_generator(),
                "references": [],
                "status": "ERROR"
            }

    def _create_initial_state(self, question: str, history: List = None) -> ChatbotState:
        """Create initial state"""
        return ChatbotState(
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
            parallel_mode=False
        )

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True, timeout=5)