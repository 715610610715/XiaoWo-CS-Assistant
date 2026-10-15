from __future__ import annotations

import math
from collections import Counter

from app.models import KnowledgeItem, SearchResult
from app.nlp.tokenizer import tokenize


class TfidfRetriever:
    """轻量级 TF-IDF 余弦检索器；只负责召回，不依赖 GUI 或存储。"""

    def __init__(self, ngram: int = 2, keyword_weight: float = 2.2):
        self.ngram = ngram
        self.keyword_weight = keyword_weight
        self._items: list[KnowledgeItem] = []
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] = []

    def fit(self, items: list[KnowledgeItem]) -> "TfidfRetriever":
        if not items:
            raise ValueError("训练数据不能为空")
        self._items = list(items)
        documents = [self._tokens_for_item(item) for item in items]
        document_frequency: Counter[str] = Counter()
        for tokens in documents:
            document_frequency.update(set(tokens))
        size = len(documents)
        self._idf = {token: math.log((size + 1) / (frequency + 1)) + 1 for token, frequency in document_frequency.items()}
        self._vectors = [self._vectorize(tokens) for tokens in documents]
        return self

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not self._items:
            raise RuntimeError("检索器尚未训练")
        query_vector = self._vectorize(tokenize(query, self.ngram))
        if not query_vector:
            return []
        ranked = [SearchResult(item, self._cosine(query_vector, vector)) for item, vector in zip(self._items, self._vectors)]
        ranked.sort(key=lambda result: result.score, reverse=True)
        return ranked[:max(1, top_k)]

    def _tokens_for_item(self, item: KnowledgeItem) -> list[str]:
        base = tokenize(item.question, self.ngram)
        for keyword in item.keywords:
            base.extend(tokenize(keyword, self.ngram) * max(1, round(self.keyword_weight)))
        return base

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(token for token in tokens if token in self._idf)
        vector = {token: (1 + math.log(count)) * self._idf[token] for token, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {token: value / norm for token, value in vector.items()}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(token, 0.0) for token, value in left.items())
