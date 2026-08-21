"""
Multi-Strategy Chunking Engine
Implements 5 distinct chunking strategies with adaptive selection.
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import nltk
from nltk.tokenize import sent_tokenize

from .strategies import (
    FixedSizeChunker,
    SemanticChunker,
    SlidingWindowChunker,
    MetadataAwareChunker,
    SentenceChunker,
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


@dataclass
class Chunk:
    text: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    strategy: str = ""
    start_idx: int = 0
    end_idx: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "strategy": self.strategy,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
        }


class ChunkingEngine:
    """
    Unified chunking engine that applies multiple strategies and
    produces a merged, deduplicated chunk index.
    
    Strategies:
    1. Fixed-size with overlap
    2. Sliding window (character-level)
    3. Sentence-aware
    4. Semantic (embedding-based similarity clustering)
    5. Metadata-aware (preserves document structure)
    """

    def __init__(self, config=None):
        from ..config import ChunkingConfig
        self.config = config or ChunkingConfig()

        self.strategies = {
            "fixed_size": FixedSizeChunker(
                chunk_size=self.config.fixed_size,
                overlap=self.config.fixed_overlap,
            ),
            "sliding_window": SlidingWindowChunker(
                window_size=self.config.sliding_window,
                step=self.config.sliding_step,
            ),
            "sentence": SentenceChunker(max_sentences=4),
            "semantic": SemanticChunker(
                threshold=self.config.semantic_threshold,
            ),
            "metadata_aware": MetadataAwareChunker(
                max_tokens=self.config.metadata_max_tokens,
            ),
        }

    def chunk(
        self,
        text: str,
        doc_id: str = "doc_0",
        metadata: Optional[Dict[str, Any]] = None,
        strategies: Optional[List[str]] = None,
    ) -> List[Chunk]:
        """
        Apply chunking strategies and return merged chunks.
        
        When multiple strategies are specified, chunks from each strategy
        are combined and deduplicated based on text similarity.
        """
        if not text or not text.strip():
            return []

        meta = metadata or {}
        meta["doc_id"] = doc_id

        active = strategies or list(self.strategies.keys())
        all_chunks: List[Chunk] = []

        for strat_name in active:
            if strat_name not in self.strategies:
                continue

            strategy = self.strategies[strat_name]
            raw_chunks = strategy.chunk(text, meta)

            for i, chunk_text in enumerate(raw_chunks):
                chunk = Chunk(
                    text=chunk_text.strip(),
                    chunk_id=f"{doc_id}_{strat_name}_{i:04d}",
                    metadata={**meta, "strategy": strat_name},
                    strategy=strat_name,
                    start_idx=text.find(chunk_text[:50]) if len(chunk_text) > 0 else 0,
                )
                if chunk.text and len(chunk.text) > 20:
                    all_chunks.append(chunk)

        merged = self._merge_and_deduplicate(all_chunks)
        return merged

    def _merge_and_deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        """Remove near-duplicate chunks across strategies."""
        if not chunks:
            return []

        seen = set()
        unique = []
        for chunk in chunks:
            normalized = self._normalize(chunk.text)
            key = normalized[:120]
            if key not in seen:
                seen.add(key)
                unique.append(chunk)

        return unique

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def chunk_batch(
        self,
        documents: List[Dict[str, Any]],
        strategies: Optional[List[str]] = None,
    ) -> List[Chunk]:
        """Chunk a batch of documents."""
        all_chunks = []
        for doc in documents:
            doc_id = doc.get("id", f"doc_{len(all_chunks)}")
            text = doc.get("text", doc.get("passage", doc.get("content", "")))
            meta = doc.get("metadata", {})
            chunks = self.chunk(text, doc_id=doc_id, metadata=meta, strategies=strategies)
            all_chunks.extend(chunks)
        return all_chunks
