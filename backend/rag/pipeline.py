"""
RAG Pipeline with structured orchestration harness.
Uses NVIDIA NIM (free) for LLM answer generation.
"""
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from ..vectorstore.faiss_store import VectorStore, SearchResult
from ..rag.guardrails import Guardrails, GuardrailResult
from ..analytics.latency import LatencyTracker

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    SUCCESS = "success"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    NO_RETRIEVAL = "no_retrieval"
    LLM_ERROR = "llm_error"
    STT_ERROR = "stt_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class PipelineResponse:
    answer: str
    status: PipelineStatus
    query: str
    transcribed_text: Optional[str] = None
    contexts: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    guardrail: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    error: Optional[str] = None


class RAGPipeline:
    """
    Production RAG pipeline with:
    - NVIDIA NIM free LLM for answer generation
    - Structured orchestration (retries, error recovery)
    - Guardrails integration
    - Latency analytics
    """

    def __init__(
        self,
        vector_store: VectorStore,
        guardrails: Optional[Guardrails] = None,
        latency_tracker: Optional[LatencyTracker] = None,
        config=None,
    ):
        from ..config import RAGConfig
        self.vector_store = vector_store
        self.guardrails = guardrails or Guardrails()
        self.latency = latency_tracker or LatencyTracker()
        self.config = config or RAGConfig()
        self._llm_client = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI(
                    api_key=self.config.nvidia_api_key,
                    base_url=self.config.nvidia_base_url,
                )
            except ImportError:
                raise RuntimeError("openai package not installed (needed for NVIDIA NIM)")
        return self._llm_client

    async def process(
        self,
        query: str = None,
        audio_bytes: bytes = None,
        stt_module=None,
        language_code: str = "en",
    ) -> PipelineResponse:
        """
        Full pipeline: input -> guard -> retrieve -> generate -> guard -> respond.
        """
        timer = self.latency.start()
        status = PipelineStatus.SUCCESS

        try:
            # Step 1: Handle speech-to-text if audio provided
            transcribed = query
            if audio_bytes and stt_module:
                stt_result = await stt_module.transcribe(audio_bytes, language_code)
                if not stt_result["success"]:
                    timer.stop()
                    return PipelineResponse(
                        answer="",
                        status=PipelineStatus.STT_ERROR,
                        query=query or "",
                        error=stt_result["error"],
                        latency_ms=timer.elapsed_ms,
                    )
                transcribed = stt_result["text"]

            if not transcribed or not transcribed.strip():
                timer.stop()
                return PipelineResponse(
                    answer="",
                    status=PipelineStatus.GUARDRAIL_BLOCKED,
                    query=query or "",
                    error="No query provided",
                    latency_ms=timer.elapsed_ms,
                )

            # Step 2: Input guardrails
            input_check = self.guardrails.check_input(transcribed)
            if not input_check.passed:
                timer.stop()
                return PipelineResponse(
                    answer="",
                    status=PipelineStatus.GUARDRAIL_BLOCKED,
                    query=transcribed,
                    guardrail=input_check.__dict__,
                    latency_ms=timer.elapsed_ms,
                )

            # Step 3: Retrieve context
            search_results = self.vector_store.search(transcribed, top_k=8)
            if not search_results:
                timer.stop()
                return PipelineResponse(
                    answer="I couldn't find relevant information for your query.",
                    status=PipelineStatus.NO_RETRIEVAL,
                    query=transcribed,
                    latency_ms=timer.elapsed_ms,
                )

            contexts = [r.text for r in search_results]
            scores = [r.score for r in search_results]

            # Step 4: Relevance guardrail
            relevance_check = self.guardrails.check_relevance(
                transcribed, contexts, scores
            )
            if not relevance_check.passed and relevance_check.severity == "block":
                timer.stop()
                return PipelineResponse(
                    answer="I couldn't find sufficiently relevant information to answer your question accurately.",
                    status=PipelineStatus.NO_RETRIEVAL,
                    query=transcribed,
                    contexts=contexts[:3],
                    scores=scores[:3],
                    guardrail=relevance_check.__dict__,
                    latency_ms=timer.elapsed_ms,
                )

            # Step 5: Generate answer with retries
            answer = await self._generate_with_retries(transcribed, contexts)

            # Step 6: Grounding guardrail
            grounding_check = self.guardrails.check_grounding(
                transcribed, answer, contexts
            )
            if not grounding_check.passed and grounding_check.severity == "block":
                answer = (
                    "Based on the available information, I cannot provide a fully "
                    "grounded answer to this question. Here is what I found:\n\n"
                    f"{answer}"
                )

            timer.stop()
            return PipelineResponse(
                answer=answer,
                status=PipelineStatus.SUCCESS,
                query=transcribed,
                transcribed_text=transcribed if audio_bytes else None,
                contexts=contexts[:3],
                scores=scores[:3],
                guardrail={
                    "input": input_check.__dict__,
                    "relevance": relevance_check.__dict__,
                    "grounding": grounding_check.__dict__,
                },
                latency_ms=timer.elapsed_ms,
            )

        except Exception as e:
            logger.exception("Pipeline error")
            timer.stop()
            return PipelineResponse(
                answer="An unexpected error occurred while processing your query.",
                status=PipelineStatus.UNKNOWN_ERROR,
                query=query or "",
                error=str(e),
                latency_ms=timer.elapsed_ms,
            )

    async def _generate_with_retries(self, query: str, contexts: List[str]) -> str:
        """Generate answer with NVIDIA NIM LLM and structured retries."""
        context_block = "\n\n---\n\n".join(contexts)
        system_prompt = self._build_system_prompt()
        user_prompt = f"""Based ONLY on the following context, answer the question.
If the context does not contain enough information, say so clearly.
Do not make up information that is not in the context.

CONTEXT:
{context_block}

QUESTION: {query}

ANSWER:"""

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                last_error = e
                logger.warning(f"NVIDIA NIM attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        return f"I apologize, but I encountered an error generating the answer: {last_error}"

    def _build_system_prompt(self) -> str:
        return """You are a helpful AI assistant that answers questions based on provided context.
Rules:
1. ONLY use information from the provided context
2. If the context doesn't contain enough information, say "I don't have enough information to fully answer this question"
3. Be concise and accurate
4. Cite which part of the context supports your answer
5. Do not hallucinate or make up facts"""

    def get_latency_stats(self) -> Dict[str, Any]:
        return self.latency.get_stats()
