from __future__ import annotations

from app.models import EvaluationResult, KnowledgeItem
from app.nlp.retriever import TfidfRetriever


class EvaluationService:
    """使用独立测试问法比较参数，避免以训练问题原文虚高评估结果。"""

    VARIANTS = (
        ("字符 1-gram", 1, 1.0),
        ("字符 2-gram", 2, 1.0),
        ("2-gram + 关键词", 2, 2.2),
    )

    def evaluate_all(self, items: list[KnowledgeItem]) -> list[EvaluationResult]:
        return [self._evaluate(items, name, ngram, weight) for name, ngram, weight in self.VARIANTS]

    @staticmethod
    def _evaluate(items: list[KnowledgeItem], name: str, ngram: int, keyword_weight: float) -> EvaluationResult:
        retriever = TfidfRetriever(ngram=ngram, keyword_weight=keyword_weight).fit(items)
        tests = [(item, item.test_query) for item in items if item.test_query]
        top1 = top3 = 0
        reciprocal_rank = 0.0
        for expected, query in tests:
            results = retriever.search(query, top_k=3)
            ids = [result.item.id for result in results]
            if ids and ids[0] == expected.id:
                top1 += 1
            if expected.id in ids:
                top3 += 1
                reciprocal_rank += 1 / (ids.index(expected.id) + 1)
        count = len(tests)
        if count == 0:
            return EvaluationResult(name, 0.0, 0.0, 0.0, 0)
        return EvaluationResult(name, top1 / count, top3 / count, reciprocal_rank / count, count)
