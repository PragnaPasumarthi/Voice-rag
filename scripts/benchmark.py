"""
Quick benchmark script to validate latency targets.
Run after ingestion: python scripts/benchmark.py
"""
import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import config, INDEX_DIR
from backend.vectorstore.faiss_store import VectorStore
from backend.rag.pipeline import RAGPipeline
from backend.rag.guardrails import Guardrails
from backend.analytics.latency import LatencyTracker


QUERIES = [
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
    "How does reinforcement learning work?",
    "What is computer vision?",
    "Explain text-to-speech technology",
    "What is word tokenization?",
    "How does language model generate text?",
]


def run_benchmark():
    print("=" * 60)
    print("  Voice-Enabled RAG - Latency Benchmark")
    print("=" * 60)

    print("\n[1] Loading vector store...")
    store = VectorStore(config.vector_store)
    try:
        store.load(str(INDEX_DIR))
        print(f"    Index loaded: {store.size} vectors")
    except FileNotFoundError:
        print("    ERROR: Index not found. Run: python -m data.ingest")
        return

    print("\n[2] Initializing pipeline...")
    guardrails = Guardrails(config.guardrails)
    tracker = LatencyTracker()
    pipeline = RAGPipeline(
        vector_store=store,
        guardrails=guardrails,
        latency_tracker=tracker,
        config=config.rag,
    )

    print("\n[3] Running benchmark (50 queries)...")
    import asyncio

    async def _run():
        results = []
        for i in range(50):
            q = QUERIES[i % len(QUERIES)]
            timer = tracker.start("e2e")
            response = await pipeline.process(query=q)
            timer.stop()
            results.append({
                "query": q,
                "latency_ms": timer.elapsed_ms,
                "status": response.status.value,
            })
        return results

    results = asyncio.run(_run())

    print("\n[4] Results:")
    print("-" * 60)
    latencies = [r["latency_ms"] for r in results]
    latencies_sorted = sorted(latencies)

    print(f"  Samples:     {len(results)}")
    print(f"  Mean:        {statistics.mean(latencies):.1f} ms")
    print(f"  Median:      {statistics.median(latencies):.1f} ms")
    print(f"  P50:         {latencies_sorted[int(len(latencies_sorted) * 0.5)]:.1f} ms")
    print(f"  P70:         {latencies_sorted[int(len(latencies_sorted) * 0.7)]:.1f} ms")
    print(f"  P90:         {latencies_sorted[int(len(latencies_sorted) * 0.9)]:.1f} ms")
    print(f"  P100:        {max(latencies):.1f} ms")
    print(f"  Min:         {min(latencies):.1f} ms")
    print(f"  Max:         {max(latencies):.1f} ms")
    print("-" * 60)

    success = sum(1 for r in results if r["status"] == "success")
    print(f"  Success:     {success}/{len(results)} ({100*success/len(results):.0f}%)")

    under_200 = sum(1 for l in latencies if l < 200)
    print(f"  Under 200ms: {under_200}/{len(results)} ({100*under_200/len(results):.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
