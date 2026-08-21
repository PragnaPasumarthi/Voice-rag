"""
FastAPI Server for Voice-Enabled RAG Pipeline.
Provides REST API endpoints for text and voice queries.
"""
import os
import sys
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import config, INDEX_DIR
from backend.chunking.engine import ChunkingEngine
from backend.vectorstore.faiss_store import VectorStore
from backend.rag.pipeline import RAGPipeline, PipelineResponse
from backend.rag.guardrails import Guardrails
from backend.analytics.latency import LatencyTracker
from backend.stt.sarvam_stt import ElevenLabsSTT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice-Enabled RAG",
    description="HH Goa 2026 - Task 2: Voice-Enabled RAG Pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global state ---
vector_store: Optional[VectorStore] = None
rag_pipeline: Optional[RAGPipeline] = None
stt_module: Optional[ElevenLabsSTT] = None
latency_tracker: Optional[LatencyTracker] = None


class TextQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    language: str = Field(default="en")
    top_k: int = Field(default=8, ge=1, le=20)


class TextQueryResponse(BaseModel):
    answer: str
    status: str
    query: str
    transcribed_text: Optional[str] = None
    contexts: list = []
    scores: list = []
    guardrail: Optional[dict] = None
    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    web_search_ms: float = 0.0
    source: str = "corpus"
    error: Optional[str] = None


class LatencyReport(BaseModel):
    e2e: Optional[dict] = None
    all_stats: dict = {}


@app.on_event("startup")
async def startup():
    global vector_store, rag_pipeline, stt_module, latency_tracker

    logger.info("Initializing RAG pipeline components...")

    latency_tracker = LatencyTracker()
    stt_module = ElevenLabsSTT(config.stt)
    guardrails = Guardrails(config.guardrails)

    vector_store = VectorStore(config.vector_store)
    try:
        vector_store.load(str(INDEX_DIR))
        logger.info(f"Loaded existing index with {vector_store.size} vectors")
    except FileNotFoundError:
        logger.warning("No index found - run: python -m data.ingest")

    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        guardrails=guardrails,
        latency_tracker=latency_tracker,
        config=config.rag,
    )

    logger.info("Pipeline ready!")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return {"service": "Voice-Enabled RAG", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "index_loaded": vector_store is not None and vector_store.size > 0,
        "index_size": vector_store.size if vector_store else 0,
    }


@app.post("/query", response_model=TextQueryResponse)
async def text_query(req: TextQuery):
    """Text-based query endpoint."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    timer = latency_tracker.start("text_query")
    response = await rag_pipeline.process(query=req.query)
    timer.stop()

    return TextQueryResponse(
        answer=response.answer,
        status=response.status.value,
        query=response.query,
        transcribed_text=response.transcribed_text,
        contexts=response.contexts,
        scores=response.scores,
        guardrail=response.guardrail,
        latency_ms=response.latency_ms,
        retrieval_ms=response.retrieval_ms,
        llm_ms=response.llm_ms,
        web_search_ms=response.web_search_ms,
        source=response.source,
        error=response.error,
    )


@app.post("/voice-query", response_model=TextQueryResponse)
async def voice_query(
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
    top_k: int = Form(default=8),
):
    """Voice-based query endpoint. Upload audio file for transcription + RAG."""
    if not rag_pipeline or not stt_module:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    timer = latency_tracker.start("voice_query")
    audio_bytes = await audio.read()
    filename = audio.filename or "audio.wav"
    response = await rag_pipeline.process(
        audio_bytes=audio_bytes,
        stt_module=stt_module,
        language_code=language,
        filename=filename,
    )
    timer.stop()

    return TextQueryResponse(
        answer=response.answer,
        status=response.status.value,
        query=response.query,
        transcribed_text=response.transcribed_text,
        contexts=response.contexts,
        scores=response.scores,
        guardrail=response.guardrail,
        latency_ms=response.latency_ms,
        retrieval_ms=response.retrieval_ms,
        llm_ms=response.llm_ms,
        web_search_ms=response.web_search_ms,
        source=response.source,
        error=response.error,
    )


@app.get("/analytics/latency")
async def latency_report():
    """Get latency analytics (P50/P70/P100)."""
    if not latency_tracker:
        return LatencyReport()

    stats = latency_tracker.get_all_stats()
    e2e = stats.pop("e2e", None)
    return LatencyReport(e2e=e2e, all_stats=stats)


@app.get("/analytics/report")
async def latency_text_report():
    """Get human-readable latency report."""
    if not latency_tracker:
        return {"report": "No data yet"}
    return {"report": latency_tracker.get_percentile_report()}


@app.post("/analytics/benchmark")
async def run_benchmark(queries: Optional[int] = 50):
    """Run latency benchmark with sample queries."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    sample_queries = [
        "What is machine learning?",
        "How does a neural network work?",
        "Explain natural language processing",
        "What is deep learning?",
        "How does speech recognition work?",
        "What is transfer learning?",
        "What is a vector database?",
        "How does FAISS similarity search work?",
        "What is attention mechanism in transformers?",
        "What is few-shot learning?",
        "What is the capital of France?",
        "Who invented the telephone?",
        "What is photosynthesis?",
        "Explain quantum computing",
        "What is blockchain technology?",
        "How does DNA replication work?",
        "What is the theory of relativity?",
        "Explain cloud computing",
        "What is reinforcement learning?",
        "How do convolutional neural networks work?",
        "What is natural language generation?",
        "What is the Internet of Things?",
        "How does gradient descent work?",
        "What is a GAN?",
        "Explain unsupervised learning",
        "What is sentiment analysis?",
        "How does information retrieval work?",
        "What is knowledge graph?",
        "Explain recommendation systems",
        "What is prompt engineering?",
        "How do transformers architecture work?",
        "What is bag of words model?",
        "Explain TF-IDF weighting",
        "What is word embedding?",
        "How does Word2Vec work?",
        "What is BERT model?",
        "Explain GPT architecture",
        "What is Retrieval Augmented Generation?",
        "How does text classification work?",
        "What is named entity recognition?",
        "How does speech synthesis work?",
        "What is the Internet today?",
        "Who is the president of India?",
        "What is climate change?",
        "How does the stock market work?",
        "What is cryptocurrency?",
        "Explain data parallelism",
        "What is model fine-tuning?",
        "How does regularization work?",
        "What is the vanishing gradient problem?",
    ]

    n = min(queries, len(sample_queries))
    latency_tracker.clear()

    results = []
    for i in range(n):
        q = sample_queries[i % len(sample_queries)]
        timer = latency_tracker.start("e2e")
        response = await rag_pipeline.process(query=q)
        timer.stop()
        results.append({
            "query": q,
            "latency_ms": round(timer.elapsed_ms, 1),
            "retrieval_ms": round(response.retrieval_ms, 1),
            "llm_ms": round(response.llm_ms, 1),
            "web_search_ms": round(response.web_search_ms, 1),
            "source": response.source,
            "status": response.status.value,
        })

    stats = latency_tracker.get_stats("e2e")
    return {
        "total_queries": n,
        "benchmark_results": results,
        "stats": stats,
        "report": latency_tracker.get_percentile_report(),
    }


@app.get("/strategies")
async def available_strategies():
    """List available chunking strategies."""
    return {
        "strategies": [
            "fixed_size",
            "sliding_window",
            "sentence",
            "semantic",
            "metadata_aware",
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)
