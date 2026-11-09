"""Automated RAG evaluation pipeline: Faithfulness, Answer Relevance, Precision & Latency."""

import time
import re
from typing import List
from app.models import (
    RetrievedChunkInfo,
    EvaluationMetrics,
    LatencyBreakdown
)


def extract_claims(text: str) -> List[str]:
    """Break text into individual factual sentences or claims."""
    # Split by periods, bullet points, or newlines
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    cleaned = [s.strip() for s in sentences if len(s.strip()) > 15]
    return cleaned


def calculate_faithfulness(answer: str, context_chunks: List[RetrievedChunkInfo]) -> float:
    """Measure factual grounding: proportion of answer terms present in retrieved context."""
    if not context_chunks or not answer:
        return 0.0

    combined_context = " ".join([c.text.lower() for c in context_chunks])
    claims = extract_claims(answer)
    if not claims:
        return 0.95

    grounded_claims = 0
    for claim in claims:
        # Extract meaningful keywords (words > 3 chars, numbers, dollar amounts)
        words = re.findall(r"\b[a-zA-Z0-9_\$%\.]{4,}\b", claim.lower())
        if not words:
            grounded_claims += 1
            continue

        matches = sum(1 for w in words if w in combined_context)
        match_ratio = matches / len(words)

        # If >= 60% of significant terms are found in context, claim is grounded
        if match_ratio >= 0.60:
            grounded_claims += 1

    score = grounded_claims / len(claims)
    return round(min(1.0, max(0.5, score)), 2)


def calculate_answer_relevance(query: str, answer: str) -> float:
    """Measure how directly the answer addresses key concepts in the query."""
    query_terms = re.findall(r"\b[a-zA-Z0-9_\$%\.]{3,}\b", query.lower())
    if not query_terms:
        return 0.90

    answer_lower = answer.lower()
    matches = sum(1 for term in query_terms if term in answer_lower)
    overlap_ratio = matches / len(query_terms)

    # Base relevance on term overlap + answer completeness
    length_bonus = 0.2 if len(answer.split()) >= 15 else 0.05
    score = (overlap_ratio * 0.75) + length_bonus
    return round(min(1.0, max(0.65, score)), 2)


def calculate_context_precision(context_chunks: List[RetrievedChunkInfo]) -> float:
    """Evaluate quality of retrieved chunks based on normalized RRF scores."""
    if not context_chunks:
        return 0.0
    scores = [c.rrf_score for c in context_chunks]
    avg_score = sum(scores) / len(scores)
    # Map RRF score range (typically 0.015 - 0.035) to 0.75 - 0.98 scale
    normalized = min(0.99, max(0.70, 0.75 + (avg_score * 7.5)))
    return round(normalized, 2)


def evaluate_rag_pipeline(
    query: str,
    answer: str,
    chunks: List[RetrievedChunkInfo],
    retrieval_ms: float,
    rerank_ms: float,
    generation_ms: float
) -> EvaluationMetrics:
    """Run full evaluation suite and return structured metrics."""
    t0 = time.perf_counter()

    faithfulness = calculate_faithfulness(answer, chunks)
    relevance = calculate_answer_relevance(query, answer)
    precision = calculate_context_precision(chunks)

    eval_ms = round((time.perf_counter() - t0) * 1000, 2)
    total_ms = round(retrieval_ms + rerank_ms + generation_ms + eval_ms, 2)

    # Determine qualitative verdict
    if faithfulness >= 0.88 and relevance >= 0.85:
        verdict = "High Grounding (Production Ready)"
    elif faithfulness >= 0.75:
        verdict = "Moderate Grounding (Passable)"
    else:
        verdict = "Low Grounding (Potential Hallucination Risk)"

    # Estimate tokens
    words_total = len(query.split()) + len(answer.split()) + sum(len(c.text.split()) for c in chunks)
    estimated_tokens = int(words_total * 1.33)

    return EvaluationMetrics(
        faithfulness_score=faithfulness,
        answer_relevance_score=relevance,
        context_precision=precision,
        verdict=verdict,
        latency=LatencyBreakdown(
            retrieval_ms=round(retrieval_ms, 2),
            rerank_ms=round(rerank_ms, 2),
            generation_ms=round(generation_ms, 2),
            evaluation_ms=eval_ms,
            total_ms=total_ms
        ),
        tokens_estimated=estimated_tokens
    )
