"""
Data Ingestion for VoiceRAG.
Tries MSMARCO-XI from HuggingFace, falls back to local corpus.jsonl.
"""
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import config, DATA_DIR, INDEX_DIR
from backend.chunking.engine import ChunkingEngine
from backend.vectorstore.faiss_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CORPUS_FILE = DATA_DIR / "corpus.jsonl"


def try_load_real_dataset(max_docs: int = 2000) -> List[Dict[str, Any]]:
    """Try loading MSMARCO-XI from HuggingFace."""
    try:
        from datasets import load_dataset
        documents = []
        for lang in ["hi", "bn", "ta", "te", "mr", "kn"]:
            if len(documents) >= max_docs:
                break
            logger.info(f"Attempting MSMARCO-XI {lang} split...")
            try:
                ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split="validation", streaming=True)
                for i, item in enumerate(ds):
                    if len(documents) >= max_docs:
                        break
                    passages_data = item.get("passages", {})
                    eng_passages = passages_data.get("English_passages", [])
                    selected = passages_data.get("is_selected", [])
                    for j, passage in enumerate(eng_passages):
                        if len(documents) >= max_docs:
                            break
                        text = str(passage).strip() if passage else ""
                        if len(text) < 30:
                            continue
                        is_sel = selected[j] if j < len(selected) else 0
                        documents.append({
                            "id": f"msmarco_{lang}_{len(documents)}",
                            "text": text,
                            "metadata": {
                                "source": "MSMARCO-XI",
                                "language": lang,
                                "query": item.get("Eng_Query", ""),
                                "query_type": item.get("query_type", ""),
                                "is_relevant": bool(is_sel),
                            },
                        })
            except Exception as e:
                logger.warning(f"Failed {lang}: {e}")
        if documents:
            logger.info(f"Loaded {len(documents)} real MSMARCO-XI documents")
        return documents
    except Exception as e:
        logger.error(f"Real dataset unavailable: {e}")
        return []


def load_corpus(max_docs: int = 500) -> List[Dict[str, Any]]:
    """Load corpus from local JSONL file."""
    if not CORPUS_FILE.exists():
        return []
    documents = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                documents.append(entry)
                if len(documents) >= max_docs:
                    break
            except json.JSONDecodeError:
                continue
    logger.info(f"Loaded {len(documents)} documents from corpus")
    return documents


def main():
    """Full ingestion pipeline: load -> chunk -> index -> save."""
    start = time.time()

    documents = try_load_real_dataset(max_docs=2000)

    if not documents:
        documents = load_corpus(max_docs=500)

    if not documents:
        logger.error("No documents found. Run: python data/build_corpus.py")
        return

    logger.info("Chunking documents with multiple strategies...")
    engine = ChunkingEngine(config.chunking)
    chunks = engine.chunk_batch(documents, strategies=["fixed_size", "sliding_window", "sentence", "metadata_aware"])
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

    logger.info("Building FAISS vector index...")
    store = VectorStore(config.vector_store)
    chunk_dicts = [c.to_dict() for c in chunks]
    store.add_chunks(chunk_dicts)
    logger.info(f"Indexed {store.size} vectors (dimension={config.vector_store.dimension})")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save(str(INDEX_DIR))
    logger.info(f"Index saved to {INDEX_DIR}")

    elapsed = time.time() - start
    logger.info(f"Ingestion complete in {elapsed:.1f}s")
    logger.info(f"Documents: {len(documents)} | Chunks: {len(chunks)} | Vectors: {store.size}")


if __name__ == "__main__":
    main()
