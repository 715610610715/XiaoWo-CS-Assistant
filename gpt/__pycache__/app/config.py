from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    app_name: str
    data_path: Path
    history_path: Path
    feedback_path: Path
    similarity_threshold: float
    top_k: int
    use_gpt2: bool
    model_path: Path
    vocab_path: Path
    max_new_tokens: int
    temperature: float
    top_p: float


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or PROJECT_ROOT / "config" / "settings.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    return AppConfig(
        app_name=raw.get("app_name", "联通小沃智能客服"),
        data_path=resolve(raw["data_path"]),
        history_path=resolve(raw["history_path"]),
        feedback_path=resolve(raw["feedback_path"]),
        similarity_threshold=float(raw.get("similarity_threshold", 0.16)),
        top_k=int(raw.get("top_k", 3)),
        use_gpt2=bool(raw.get("use_gpt2", False)),
        model_path=resolve(raw.get("model_path", "../mymodel")),
        vocab_path=resolve(raw.get("vocab_path", "../资料数据和配置文件/vocab/vocab.txt")),
        max_new_tokens=int(raw.get("max_new_tokens", 120)),
        temperature=float(raw.get("temperature", 0.8)),
        top_p=float(raw.get("top_p", 0.9)),
    )
