from __future__ import annotations

import csv
from pathlib import Path

from app.models import KnowledgeItem


class KnowledgeDataError(ValueError):
    pass


class CsvKnowledgeRepository:
    REQUIRED_COLUMNS = {"id", "category", "question", "answer", "keywords", "test_query"}

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> list[KnowledgeItem]:
        if not self.path.exists():
            raise KnowledgeDataError(f"知识库不存在：{self.path}")
        with self.path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise KnowledgeDataError(f"知识库缺少字段：{', '.join(sorted(missing))}")
            items: list[KnowledgeItem] = []
            seen: set[str] = set()
            for line_number, row in enumerate(reader, start=2):
                item_id = (row["id"] or "").strip()
                question = (row["question"] or "").strip()
                answer = (row["answer"] or "").strip()
                if not item_id or not question or not answer:
                    raise KnowledgeDataError(f"第 {line_number} 行存在空的 id、question 或 answer")
                if item_id in seen:
                    raise KnowledgeDataError(f"第 {line_number} 行 id 重复：{item_id}")
                seen.add(item_id)
                items.append(KnowledgeItem(
                    id=item_id,
                    category=(row["category"] or "其他").strip(),
                    question=question,
                    answer=answer,
                    keywords=tuple(word.strip() for word in (row["keywords"] or "").split("|") if word.strip()),
                    test_query=(row["test_query"] or "").strip(),
                ))
        if not items:
            raise KnowledgeDataError("知识库没有有效数据")
        return items
