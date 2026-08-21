"""
Configuration for Voice-Enabled RAG Pipeline
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "data" / "faiss_index"


@dataclass
class ChunkingConfig:
    fixed_size: int = 256
    fixed_overlap: int = 64
    sliding_window: int = 300
    sliding_step: int = 100
    semantic_threshold: float = 0.5
    metadata_max_tokens: int = 256


@dataclass
class VectorStoreConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    index_type: str = "IVFFlat"
    nprobe: int = 10
    top_k: int = 8


@dataclass
class STTConfig:
    model_id: str = "scribe_v1"
    language: str = "en"
    supported_languages: list = field(default_factory=lambda: [
        "en", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "or", "pa",
        "es", "fr", "de", "pt", "ja", "ko", "zh", "ar", "ru", "it",
    ])


@dataclass
class RAGConfig:
    nvidia_api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "meta/llama-3.1-8b-instruct"
    max_tokens: int = 512
    temperature: float = 0.1
    max_retries: int = 3
    retry_delay: float = 0.5

    @property
    def has_llm(self) -> bool:
        return bool(self.nvidia_api_key)


@dataclass
class GuardrailsConfig:
    max_relevance_score: float = 0.15
    hallucination_threshold: float = 0.6
    blocked_keywords: list = field(default_factory=lambda: [
        "hack", "exploit", "bypass", "jailbreak", "ignore previous"
    ])
    topic_keywords: list = field(default_factory=lambda: [
        "what", "how", "why", "when", "where", "who", "which",
        "explain", "describe", "tell", "define", "list", "compare",
        "advantage", "disadvantage", "difference", "example", "mean"
    ])


@dataclass
class AnalyticsConfig:
    warmup_queries: int = 5
    benchmark_queries: int = 50


@dataclass
class AppConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    host: str = "0.0.0.0"
    port: int = 8000


config = AppConfig()
