"""
Individual Chunking Strategies
Each strategy implements a common interface: chunk(text, metadata) -> List[str]
"""
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[str]:
        pass


class FixedSizeChunker(BaseChunker):
    """Fixed-size character chunking with configurable overlap."""

    def __init__(self, chunk_size: int = 256, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += self.chunk_size - self.overlap
        return chunks


class SlidingWindowChunker(BaseChunker):
    """Sliding window chunking with sentence-boundary awareness."""

    def __init__(self, window_size: int = 300, step: int = 100):
        self.window_size = window_size
        self.step = step

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.window_size, len(text))
            chunk = text[start:end]
            if end < len(text):
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                boundary = max(last_period, last_newline)
                if boundary > self.window_size * 0.4:
                    chunk = text[start:start + boundary + 1]
                    end = start + boundary + 1
            if chunk.strip():
                chunks.append(chunk)
            start += self.step
            if start >= len(text):
                break
        return chunks


class SentenceChunker(BaseChunker):
    """Sentence-aware chunking that groups sentences up to a max count."""

    def __init__(self, max_sentences: int = 4):
        self.max_sentences = max_sentences

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[str]:
        if not text:
            return []
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        for i in range(0, len(sentences), self.max_sentences):
            group = sentences[i:i + self.max_sentences]
            chunk = " ".join(group)
            if chunk.strip():
                chunks.append(chunk)
        return chunks


class SemanticChunker(BaseChunker):
    """
    Semantic chunking using embedding similarity.
    Falls back to sentence chunking if sentence-transformers unavailable.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                return None
        return self._model

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[str]:
        if not text:
            return []

        model = self._get_model()
        if model is None:
            fallback = SentenceChunker(max_sentences=3)
            return fallback.chunk(text, metadata)

        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) <= 2:
            return [text]

        embeddings = model.encode(sentences, show_progress_bar=False)

        chunks = []
        current_group = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = self._cosine_sim(embeddings[i - 1], embeddings[i])
            if sim < self.threshold:
                chunk = " ".join(current_group)
                if chunk.strip():
                    chunks.append(chunk)
                current_group = [sentences[i]]
            else:
                current_group.append(sentences[i])

        if current_group:
            chunk = " ".join(current_group)
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    @staticmethod
    def _cosine_sim(a, b):
        import numpy as np
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return dot / (norm + 1e-8)


class MetadataAwareChunker(BaseChunker):
    """
    Splits text while preserving metadata regions (headers, titles, etc.).
    Identifies structural markers and creates chunks that keep them together.
    """

    def __init__(self, max_tokens: int = 256):
        self.max_tokens = max_tokens
        self._pattern = re.compile(
            r'(?:^|\n)(#{1,6}\s+.+|(?:[A-Z][A-Za-z\s]{2,30}\n[-=]{3,})|(?:\d+\.\s+.+))',
            re.MULTILINE
        )

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[str]:
        if not text:
            return []

        regions = self._find_regions(text)
        if not regions:
            return self._fallback_chunk(text)

        chunks = []
        for header, body in regions:
            combined = f"{header}\n{body}".strip()
            if self._estimate_tokens(combined) <= self.max_tokens:
                chunks.append(combined)
            else:
                sub_chunks = self._split_region(combined)
                chunks.extend(sub_chunks)

        return chunks

    def _find_regions(self, text: str):
        matches = list(self._pattern.finditer(text))
        if not matches:
            return []

        regions = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            header = match.group().strip()
            body = text[match.end():end].strip()
            if body:
                regions.append((header, body))
        return regions

    def _split_region(self, text: str):
        words = text.split()
        chunks = []
        current = []
        for word in words:
            current.append(word)
            if len(current) >= self.max_tokens:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _fallback_chunk(self, text: str):
        words = text.split()
        chunks = []
        current = []
        for word in words:
            current.append(word)
            if len(current) >= self.max_tokens:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text.split()) * 1.3
