"""
Guardrails for RAG pipeline.
Handles off-topic queries, unsafe inputs, hallucination detection, and grounding checks.
"""
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    passed: bool
    reason: str
    severity: str  # "block", "warn", "pass"
    details: Dict[str, Any] = None


class Guardrails:
    """
    Multi-layered guardrail system:
    1. Input validation (length, encoding)
    2. Safety filter (blocked keywords, injection patterns)
    3. Topic relevance check
    4. Hallucination detection (grounding in retrieved context)
    5. Response quality check
    """

    def __init__(self, config=None):
        from ..config import GuardrailsConfig
        self.config = config or GuardrailsConfig()
        self._injection_patterns = [
            r"ignore\s+(previous|all|above)\s+(instructions?|prompts?|rules?)",
            r"you\s+are\s+now\s+",
            r"pretend\s+(you|that|to)\s+",
            r"act\s+as\s+if\s+",
            r"disregard\s+",
            r"forget\s+(everything|all|your)\s+",
            r"system\s*:\s*",
            r"override\s+",
            r"bypass\s+",
            r"jailbreak",
        ]

    def check_input(self, query: str) -> GuardrailResult:
        """Run all input guardrails."""
        checks = [
            self._check_empty(query),
            self._check_length(query),
            self._check_safety(query),
            self._check_injection(query),
            self._check_topic_relevance(query),
        ]

        for result in checks:
            if not result.passed:
                return result

        return GuardrailResult(passed=True, reason="All guardrails passed", severity="pass")

    def _check_empty(self, query: str) -> GuardrailResult:
        if not query or not query.strip():
            return GuardrailResult(
                passed=False,
                reason="Empty or whitespace-only query",
                severity="block",
            )
        return GuardrailResult(passed=True, reason="", severity="pass")

    def _check_length(self, query: str) -> GuardrailResult:
        if len(query.strip()) < 3:
            return GuardrailResult(
                passed=False,
                reason="Query too short to be meaningful",
                severity="block",
            )
        if len(query) > 10000:
            return GuardrailResult(
                passed=False,
                reason="Query exceeds maximum length (10,000 characters)",
                severity="block",
            )
        return GuardrailResult(passed=True, reason="", severity="pass")

    def _check_safety(self, query: str) -> GuardrailResult:
        lower = query.lower()
        for keyword in self.config.blocked_keywords:
            if keyword.lower() in lower:
                return GuardrailResult(
                    passed=False,
                    reason=f"Query contains blocked content: '{keyword}'",
                    severity="block",
                    details={"blocked_keyword": keyword},
                )
        return GuardrailResult(passed=True, reason="", severity="pass")

    def _check_injection(self, query: str) -> GuardrailResult:
        lower = query.lower()
        for pattern in self._injection_patterns:
            if re.search(pattern, lower):
                return GuardrailResult(
                    passed=False,
                    reason="Potential prompt injection detected",
                    severity="block",
                    details={"pattern": pattern},
                )
        return GuardrailResult(passed=True, reason="", severity="pass")

    def _check_topic_relevance(self, query: str) -> GuardrailResult:
        lower = query.lower()
        has_question_word = any(kw in lower for kw in self.config.topic_keywords)
        if not has_question_word and len(query.split()) > 5:
            return GuardrailResult(
                passed=True,
                reason="Query may be off-topic but proceeding with caution",
                severity="warn",
            )
        return GuardrailResult(passed=True, reason="", severity="pass")

    def check_grounding(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> GuardrailResult:
        """
        Check if the generated answer is grounded in the retrieved contexts.
        Uses token overlap as a proxy for grounding.
        """
        if not answer or not answer.strip():
            return GuardrailResult(
                passed=False,
                reason="Empty answer generated",
                severity="block",
            )

        answer_tokens = set(answer.lower().split())
        context_tokens = set()
        for ctx in contexts:
            context_tokens.update(ctx.lower().split())

        if not context_tokens:
            return GuardrailResult(
                passed=False,
                reason="No context available for grounding check",
                severity="warn",
            )

        overlap = answer_tokens & context_tokens
        overlap_ratio = len(overlap) / max(len(answer_tokens), 1)

        if overlap_ratio < self.config.hallucination_threshold:
            return GuardrailResult(
                passed=False,
                reason=f"Answer may be hallucinated (grounding: {overlap_ratio:.1%})",
                severity="warn",
                details={
                    "overlap_ratio": overlap_ratio,
                    "threshold": self.config.hallucination_threshold,
                },
            )

        return GuardrailResult(
            passed=True,
            reason=f"Answer grounded in context ({overlap_ratio:.1%} overlap)",
            severity="pass",
            details={"overlap_ratio": overlap_ratio},
        )

    def check_relevance(self, query: str, contexts: List[str], scores: List[float]) -> GuardrailResult:
        """Check if retrieved contexts are relevant to the query."""
        if not scores:
            return GuardrailResult(
                passed=False,
                reason="No retrieval results found",
                severity="block",
            )

        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        if max_score < self.config.max_relevance_score:
            return GuardrailResult(
                passed=False,
                reason=f"Low relevance scores (max: {max_score:.3f}, avg: {avg_score:.3f})",
                severity="block",
                details={"max_score": max_score, "avg_score": avg_score},
            )

        return GuardrailResult(
            passed=True,
            reason=f"Context relevant (max: {max_score:.3f}, avg: {avg_score:.3f})",
            severity="pass",
            details={"max_score": max_score, "avg_score": avg_score},
        )
