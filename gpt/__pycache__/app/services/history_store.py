from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonlStore:
    """追加式 JSONL 存储，聊天服务无需感知文件格式细节。"""

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"created_at": datetime.now().isoformat(timespec="seconds"), **record}
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def load(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records[-limit:]

    def clear(self) -> None:
        if self.path.exists():
            self.path.write_text("", encoding="utf-8")
