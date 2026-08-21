"""
Data Ingestion Script for MSMARCO-XI dataset from HuggingFace.
Downloads, preprocesses, chunks, and indexes the dataset into FAISS.
"""
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import config, DATA_DIR, INDEX_DIR
from backend.chunking.engine import ChunkingEngine
from backend.vectorstore.faiss_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(subset_size: int = 2000) -> List[Dict[str, Any]]:
    """
    Load MSMARCO-XI from HuggingFace datasets.
    Returns list of dicts with 'text' and 'id' keys.
    """
    from datasets import load_dataset

    logger.info("Loading MSMARCO-XI dataset from HuggingFace...")
    try:
        ds = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)
        documents = []
        for i, item in enumerate(ds):
            if i >= subset_size:
                break

            text = item.get("passage", item.get("text", item.get("document", "")))
            if not text or len(str(text).strip()) < 30:
                continue

            doc_id = f"msmarco_{i}"
            metadata = {
                "source": "MSMARCO-XI",
                "query_id": item.get("query_id", ""),
                "query": item.get("query", ""),
            }

            documents.append({
                "id": doc_id,
                "text": str(text).strip(),
                "metadata": metadata,
            })

        logger.info(f"Loaded {len(documents)} documents from MSMARCO-XI")
        return documents

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Generating synthetic demo data instead...")
        return generate_demo_data(subset_size)


def generate_demo_data(n: int = 2000) -> List[Dict[str, Any]]:
    """Generate synthetic Q&A data for demo purposes."""
    topics = [
        ("What is machine learning?",
         "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data, learn from it, and make predictions."),
        ("How does neural network work?",
         "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information using connectionist approaches. Each neuron receives inputs, applies weights and biases, and produces an output through an activation function."),
        ("What is natural language processing?",
         "Natural language processing (NLP) is a subfield of AI that focuses on the interaction between computers and human language. It involves processing and understanding text or speech data, enabling machines to comprehend, interpret, and generate human language."),
        ("What is deep learning?",
         "Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input. It has achieved breakthrough results in image recognition, speech processing, and natural language understanding."),
        ("How does computer vision work?",
         "Computer vision is a field of AI that trains computers to interpret and understand visual information from the world. It uses digital images from cameras and videos to extract meaningful information and make decisions based on that data."),
        ("What is reinforcement learning?",
         "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. The agent learns through trial and error, receiving feedback in the form of rewards or penalties."),
        ("What is transformers in AI?",
         "Transformers are a deep learning architecture introduced in 2017 that revolutionized NLP. They use self-attention mechanisms to process input data in parallel, making them highly efficient for sequence-to-sequence tasks like translation, summarization, and text generation."),
        ("What is transfer learning?",
         "Transfer learning is a technique where a model developed for one task is reused as the starting point for a model on a second task. It significantly reduces training time and data requirements by leveraging knowledge learned from previous tasks."),
        ("How does text-to-speech work?",
         "Text-to-speech (TTS) technology converts written text into spoken audio. Modern TTS systems use deep learning models to generate natural-sounding speech. The process involves text analysis, linguistic processing, and audio synthesis to produce human-like speech output."),
        ("What is speech recognition?",
         "Speech recognition is the technology that enables computers to identify and process human speech. It converts audio signals into text using acoustic and language models. Modern systems use deep learning to achieve high accuracy across multiple languages and accents."),
        ("What is retrieval augmented generation?",
         "Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them as context for a language model to generate more accurate and grounded responses."),
        ("What is vector database?",
         "A vector database stores data as high-dimensional vectors (embeddings) and enables efficient similarity search. It uses algorithms like FAISS, Annoy, or HNSW to find nearest neighbors, making it ideal for semantic search, recommendation systems, and RAG pipelines."),
        ("What is embedding in NLP?",
         "Embeddings are dense vector representations of text that capture semantic meaning. Words or sentences with similar meanings are mapped to nearby points in the embedding space. Popular models include Word2Vec, GloVe, and sentence transformers like all-MiniLM-L6-v2."),
        ("How does FAISS work?",
         "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It provides algorithms for searching sets of vectors of any size, including methods that are highly efficient for large-scale search."),
        ("What is chunking in RAG?",
         "Chunking is the process of splitting documents into smaller, manageable pieces for indexing and retrieval. Common strategies include fixed-size chunking, semantic chunking, sentence-based chunking, and overlap-aware chunking. Good chunking improves retrieval quality."),
        ("What is cosine similarity?",
         "Cosine similarity measures the similarity between two non-zero vectors by computing the cosine of the angle between them. In NLP, it is commonly used to measure the semantic similarity between text embeddings, where values closer to 1 indicate higher similarity."),
        ("What is attention mechanism?",
         "The attention mechanism is a technique that allows neural networks to focus on specific parts of the input when producing output. Self-attention, the foundation of transformers, computes relationships between all positions in a sequence simultaneously."),
        ("What is word tokenization?",
         "Tokenization is the process of breaking text into individual units called tokens. These can be words, subwords, or characters. Common algorithms include BPE (Byte Pair Encoding), WordPiece, and SentencePiece, each with different trade-offs for vocabulary size and coverage."),
        ("How does language model work?",
         "Language models predict the probability of word sequences. They learn statistical patterns from text data and can generate coherent text. Modern large language models (LLMs) like GPT use transformer architecture to capture long-range dependencies in text."),
        ("What is few-shot learning?",
         "Few-shot learning is a machine learning paradigm where a model learns to make accurate predictions from very few training examples. It leverages prior knowledge and meta-learning techniques to generalize from limited data, similar to how humans learn from small examples."),
    ]

    documents = []
    idx = 0
    for i in range(n):
        topic_idx = i % len(topics)
        query, passage = topics[topic_idx]
        # Slight variation
        suffix = f" This relates to the broader field of artificial intelligence and computer science research." if i % 3 == 0 else ""
        documents.append({
            "id": f"demo_{idx}",
            "text": passage + suffix,
            "metadata": {
                "source": "synthetic",
                "query": query,
            },
        })
        idx += 1

    logger.info(f"Generated {len(documents)} synthetic documents")
    return documents


def main():
    """Full ingestion pipeline: load -> chunk -> index -> save."""
    start = time.time()

    # Step 1: Use demo data (fast and reliable for demo)
    documents = generate_demo_data(200)
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return

    # Step 2: Chunk with multiple strategies (use fewer for speed)
    logger.info("Chunking documents with multiple strategies...")
    engine = ChunkingEngine(config.chunking)
    chunks = engine.chunk_batch(documents, strategies=["fixed_size", "sentence"])
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

    # Step 3: Build vector store
    logger.info("Building FAISS vector index...")
    store = VectorStore(config.vector_store)
    chunk_dicts = [c.to_dict() for c in chunks]
    store.add_chunks(chunk_dicts)
    logger.info(f"Indexed {store.size} vectors (dimension={config.vector_store.dimension})")

    # Step 4: Save
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save(str(INDEX_DIR))
    logger.info(f"Index saved to {INDEX_DIR}")

    elapsed = time.time() - start
    logger.info(f"Ingestion complete in {elapsed:.1f}s")
    logger.info(f"Documents: {len(documents)} | Chunks: {len(chunks)} | Vectors: {store.size}")


if __name__ == "__main__":
    main()
