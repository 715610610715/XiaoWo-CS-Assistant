from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class KnowledgeItem:
    id: str
    category: str
    question: str
    answer: str
    keywords: tuple[str, ...] = ()
    test_query: str = ""


@dataclass(frozen=True)
class SearchResult:
    item: KnowledgeItem
    score: float


@dataclass(frozen=True)
class ChatReply:
    text: str
    confidence: float
    category: str
    source_id: str | None
    suggestions: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    top1_accuracy: float
    top3_accuracy: float
    mrr: float
    sample_count: int
