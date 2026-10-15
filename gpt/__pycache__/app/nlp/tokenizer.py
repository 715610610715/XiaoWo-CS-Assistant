from __future__ import annotations

import re


NORMALIZE_PATTERN = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")


def normalize(text: str) -> str:
    return NORMALIZE_PATTERN.sub("", text.lower())


def tokenize(text: str, ngram: int = 2) -> list[str]:
    """字符 n-gram 对中文短问句稳定，且不依赖外部分词包。"""
    cleaned = normalize(text)
    if not cleaned:
        return []
    tokens = list(cleaned)
    if ngram > 1 and len(cleaned) >= ngram:
        tokens.extend(cleaned[index:index + ngram] for index in range(len(cleaned) - ngram + 1))
    return tokens
