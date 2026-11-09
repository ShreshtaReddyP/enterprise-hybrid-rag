"""Pydantic schemas and data models for Enterprise Hybrid RAG."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int
    text: str
    embedding: Optional[List[float]] = None
    char_length: int = 0


class Citation(BaseModel):
    citation_id: int
    source_file: str
    page_number: int
    snippet: str
    relevance_score: float


class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    evaluation_ms: float
    total_ms: float


class EvaluationMetrics(BaseModel):
    faithfulness_score: float = Field(..., description="0.0 - 1.0: Factual grounding in context (low hallucination)")
    answer_relevance_score: float = Field(..., description="0.0 - 1.0: How directly it addresses the query")
    context_precision: float = Field(..., description="0.0 - 1.0: Proportion of retrieved chunks that are relevant")
    verdict: str = Field(..., description="'High Grounding', 'Moderate Grounding', or 'Low Grounding'")
    latency: LatencyBreakdown
    tokens_estimated: int = 0


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="User question or research query")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of re-ranked chunks to provide LLM")
    include_evaluation: bool = Field(default=True, description="Run automated RAG evaluation metrics")


class RetrievedChunkInfo(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int
    text: str
    dense_score: float
    sparse_score: float
    rrf_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunkInfo]
    evaluation: Optional[EvaluationMetrics] = None
    model_used: str = "gemini-2.5-flash"


class DocumentInfo(BaseModel):
    filename: str
    total_pages: int
    total_chunks: int
    uploaded_at: str


class DocumentStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[DocumentInfo]
