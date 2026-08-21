"""
RAG Pipeline with web search fallback and structured orchestration harness.
Uses LLM for answer generation, DuckDuckGo for web search when corpus is insufficient.
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

try:
    from ddgs import DDGS
    DDG_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDG_AVAILABLE = True
    except ImportError:
        DDG_AVAILABLE = False
        logger.warning("duckduckgo-search not installed, web fallback disabled")


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
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    web_search_ms: float = 0.0
    source: str = "corpus"
    error: Optional[str] = None


class RAGPipeline:

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
                raise RuntimeError("openai package not installed (needed for LLM)")
        return self._llm_client

    def _web_search(self, query: str, max_results: int = 5) -> List[str]:
        if not DDG_AVAILABLE:
            return []
        try:
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=max_results))
            bodies = [r.get("body", "") for r in results if r.get("body")]
            logger.info(f"Web search for '{query[:50]}': got {len(bodies)} results")
            return bodies
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []

    async def process(
        self,
        query: str = None,
        audio_bytes: bytes = None,
        stt_module=None,
        language_code: str = "en",
        filename: str = "audio.wav",
    ) -> PipelineResponse:
        timer = self.latency.start()
        retrieval_ms = 0.0
        llm_ms = 0.0
        web_search_ms = 0.0
        source = "corpus"

        try:
            # Step 1: STT
            transcribed = query
            if audio_bytes and stt_module:
                stt_result = await stt_module.transcribe(audio_bytes, language_code, filename=filename)
                if not stt_result["success"]:
                    timer.stop()
                    return PipelineResponse(
                        answer="", status=PipelineStatus.STT_ERROR,
                        query=query or "", error=stt_result["error"],
                        latency_ms=timer.elapsed_ms,
                    )
                transcribed = stt_result["text"]

            if not transcribed or not transcribed.strip():
                timer.stop()
                return PipelineResponse(
                    answer="", status=PipelineStatus.GUARDRAIL_BLOCKED,
                    query=query or "", error="No query provided",
                    latency_ms=timer.elapsed_ms,
                )

            # Step 2: Input guardrails
            input_check = self.guardrails.check_input(transcribed)
            if not input_check.passed:
                timer.stop()
                return PipelineResponse(
                    answer="", status=PipelineStatus.GUARDRAIL_BLOCKED,
                    query=transcribed, guardrail=input_check.__dict__,
                    latency_ms=timer.elapsed_ms,
                )

            # Step 3: Retrieve from vector store
            t0 = time.perf_counter()
            search_results = self.vector_store.search(transcribed, top_k=8)
            retrieval_ms = (time.perf_counter() - t0) * 1000

            contexts = [r.text for r in search_results] if search_results else []
            scores = [r.score for r in search_results] if search_results else []

            # Step 4: Check relevance — if low, fall back to web search
            use_web = False
            if search_results:
                relevance_check = self.guardrails.check_relevance(transcribed, contexts, scores)
                if not relevance_check.passed and relevance_check.severity == "block":
                    use_web = True
            else:
                use_web = True

            if use_web and DDG_AVAILABLE:
                t0 = time.perf_counter()
                web_results = self._web_search(transcribed, max_results=5)
                web_search_ms = (time.perf_counter() - t0) * 1000
                if web_results:
                    contexts = web_results
                    scores = [0.5] * len(web_results)
                    source = "web"
                relevance_check = GuardrailResult(
                    passed=True, reason="Web search results used", severity="pass"
                )
            elif not search_results:
                timer.stop()
                return PipelineResponse(
                    answer="I couldn't find relevant information for your query.",
                    status=PipelineStatus.NO_RETRIEVAL,
                    query=transcribed, latency_ms=timer.elapsed_ms,
                )
            else:
                relevance_check = self.guardrails.check_relevance(transcribed, contexts, scores)

            # Step 5: Generate answer
            if self.config.has_llm:
                t0 = time.perf_counter()
                answer = await self._generate_with_retries(transcribed, contexts, source)
                llm_ms = (time.perf_counter() - t0) * 1000

                # If LLM says it can't answer from corpus, try web search
                insufficient = any(phrase in answer.lower() for phrase in [
                    "don't have enough", "not contain enough", "cannot provide",
                    "not enough information", "insufficient information"
                ])
                if insufficient and DDG_AVAILABLE and source == "corpus":
                    t0 = time.perf_counter()
                    web_results = self._web_search(transcribed, max_results=5)
                    web_search_ms = (time.perf_counter() - t0) * 1000
                    if web_results:
                        t0 = time.perf_counter()
                        answer = await self._generate_with_retries(transcribed, web_results, "web")
                        llm_ms += (time.perf_counter() - t0) * 1000
                        source = "web"
                        contexts = web_results
                        scores = [0.5] * len(web_results)
            else:
                answer = "Based on the retrieved information:\n\n" + "\n\n---\n\n".join(contexts[:3])

            # Step 6: Grounding check
            grounding_check = self.guardrails.check_grounding(transcribed, answer, contexts)
            if not grounding_check.passed and grounding_check.severity == "block":
                answer = (
                    "Based on the available information, I cannot provide a fully "
                    "grounded answer to this question. Here is what I found:\n\n"
                    f"{answer}"
                )

            timer.stop()
            return PipelineResponse(
                answer=answer, status=PipelineStatus.SUCCESS,
                query=transcribed, transcribed_text=transcribed if audio_bytes else None,
                contexts=contexts[:3], scores=scores[:3],
                guardrail={
                    "input": input_check.__dict__,
                    "relevance": relevance_check.__dict__,
                    "grounding": grounding_check.__dict__,
                },
                latency_ms=timer.elapsed_ms,
                retrieval_ms=retrieval_ms,
                llm_ms=llm_ms,
                web_search_ms=web_search_ms,
                source=source,
            )

        except Exception as e:
            logger.exception("Pipeline error")
            timer.stop()
            return PipelineResponse(
                answer="An unexpected error occurred while processing your query.",
                status=PipelineStatus.UNKNOWN_ERROR,
                query=query or "", error=str(e),
                latency_ms=timer.elapsed_ms,
            )

    async def _generate_with_retries(self, query: str, contexts: List[str], source: str = "corpus") -> str:
        context_block = "\n\n---\n\n".join(contexts)

        if source == "web":
            system_prompt = (
                "You are a helpful AI assistant. Answer the user's question directly using the "
                "web search results below. These are real-time search results from the internet. "
                "Provide a clear, direct, factual answer. Be concise."
            )
            user_prompt = f"""Answer this question using the web search results below.

WEB SEARCH RESULTS:
{context_block}

QUESTION: {query}

DIRECT ANSWER:"""
        else:
            system_prompt = self._build_system_prompt()
            user_prompt = f"""Answer the following question based on the provided context.

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
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
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
