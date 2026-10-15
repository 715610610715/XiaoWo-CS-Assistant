"""下载并统一客服语料为通用 conversational JSONL。

本脚本设计为在可联网的云服务器运行。本地内置领域知识始终加入训练集；
外部数据只从许可清晰且显式列入白名单的数据源加载。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "你是中国联通小沃客服。回答应准确、简洁、礼貌并保护隐私；"
    "不编造资费、优惠、覆盖或办理结果，存在地区和时效差异时引导用户通过中国联通App或10010核实。"
)
SENSITIVE_PATTERNS = (
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号码]"),
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "[身份证号]"),
    (re.compile(r"(?<!\d)\d{15,19}(?!\d)"), "[账号]"),
)
STALE_OR_LOW_QUALITY = re.compile(
    r"20(?:0\d|1\d|2[0-4])年|\d+(?:\.\d+)?元|省内流量|2G网络|3G网络|"
    r"冰激凌套餐|腾讯王卡|iphone|ipad|苹果[45678xX]|三星|小米|红米|魅族|酷派|"
    r"刷机|破解|卡贴|越狱|手机型号|大神|不知道|不清楚|百度一下|自己百度|"
    r"应该是|好像|可能是|百分之百|胡说|问问客服|详询当地|咨询当地"
)
DOMAIN_TERMS = re.compile(
    r"联通|话费|流量|套餐|宽带|手机卡|SIM|号码|营业厅|漫游|长途|欠费|充值|"
    r"停机|复机|销户|信号|网络|通话|短信|上网|4G|5G|副卡|发票|实名|补卡|光猫"
)


def clean_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    for pattern, replacement in SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def local_knowledge_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            examples.append({
                "id": f"local-{row['id']}",
                "group_id": f"local-{row['id']}",
                "source": "local-telecom-kb",
                "category": row["category"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": clean_text(row["question"])},
                    {"role": "assistant", "content": clean_text(row["answer"])},
                ],
            })
            if row.get("test_query"):
                examples.append({
                    "id": f"local-aug-{row['id']}",
                    "group_id": f"local-{row['id']}",
                    "source": "local-telecom-kb",
                    "category": row["category"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": clean_text(row["test_query"])},
                        {"role": "assistant", "content": clean_text(row["answer"])},
                    ],
                })
    return examples


def legacy_telecom_questions(path: Path, max_samples: int, seed: int) -> list[dict[str, Any]]:
    """抽取真实用户问法；旧回答可能失效，严禁默认混入生成式 SFT。"""
    if not path.exists():
        print(f"提示：未找到联通 20.3 万问答，跳过：{path}")
        return []
    randomizer = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    accepted = 0
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row_index, row in enumerate(csv.DictReader(file)):
            if clean_text(row.get("is_best")) != "1":
                continue
            question = clean_text(row.get("title"))
            answer = clean_text(row.get("reply"))
            normalized_question = re.sub(r"\W+", "", question.lower())
            if not (4 <= len(question) <= 100 and 20 <= len(answer) <= 500):
                continue
            combined = question + " " + answer
            if (
                not normalized_question
                or normalized_question in seen_questions
                or "###" in combined
                or not DOMAIN_TERMS.search(question)
                or STALE_OR_LOW_QUALITY.search(combined)
            ):
                continue
            seen_questions.add(normalized_question)
            answer = answer.replace("联通手机营业厅客户端", "中国联通App").replace("联通手机营业厅", "中国联通App")
            example = {
                "id": f"legacy-{row_index:06d}",
                "source": "liantongzhidao-filtered",
                "text": question,
            }
            accepted += 1
            if len(reservoir) < max_samples:
                reservoir.append(example)
            else:
                replacement = randomizer.randrange(accepted)
                if replacement < max_samples:
                    reservoir[replacement] = example
    print(f"联通历史用户问法：过滤后 {accepted} 条，抽样保留 {len(reservoir)} 条（不使用旧回答做SFT）")
    return reservoir


def _speaker_role(speaker: str) -> str | None:
    value = speaker.strip().lower()
    if value in {"客户", "用户", "user", "client", "customer"}:
        return "user"
    if value in {"客服", "坐席", "assistant", "agent"}:
        return "assistant"
    return None


def normalize_dialogue(turns: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in turns:
        role = _speaker_role(str(turn.get("speaker", "")))
        content = clean_text(turn.get("text", ""))
        if not role or not content:
            continue
        if len(messages) > 1 and messages[-1]["role"] == role:
            messages[-1]["content"] += " " + content
        else:
            messages.append({"role": role, "content": content})
    while len(messages) > 1 and messages[-1]["role"] != "assistant":
        messages.pop()
    return messages if len(messages) >= 3 else []


def load_dianjin(max_samples: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("缺少 datasets，请先运行 pip install -r requirements-cloud.txt") from error
    dataset = load_dataset("DianJin/DianJin-CSC-Data", split="train")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        messages = normalize_dialogue(row.get("dialogue") or [])
        if messages:
            examples.append({
                "id": f"dianjin-{index:06d}",
                "group_id": f"dianjin-{index:06d}",
                "source": "DianJin-CSC",
                "category": clean_text(row.get("source", "通用客服")),
                "messages": messages,
            })
    return examples


def split_examples(examples: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    """按会话/知识点分组切分，避免同义问法泄漏到不同集合。"""
    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault(example["group_id"], []).append(example)
    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    valid_size = max(1, round(len(group_ids) * 0.05))
    test_size = max(1, round(len(group_ids) * 0.05))
    split_ids = {
        "train": group_ids[: len(group_ids) - valid_size - test_size],
        "validation": group_ids[len(group_ids) - valid_size - test_size: len(group_ids) - test_size],
        "test": group_ids[len(group_ids) - test_size:],
    }
    return {name: [row for group_id in ids for row in groups[group_id]] for name, ids in split_ids.items()}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建云端 SFT 客服数据")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--skip-external", action="store_true", help="仅处理本地领域知识")
    parser.add_argument("--max-external", type=int, default=None, help="调试时限制外部样本数")
    parser.add_argument("--skip-legacy", action="store_true", help="不使用项目附带的 20.3 万联通问答")
    parser.add_argument("--max-legacy", type=int, default=30000, help="旧联通问答通过质量过滤后的最大抽样数")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    examples = local_knowledge_examples(ROOT / "data" / "knowledge_base.csv")
    domain_questions: list[dict[str, Any]] = []
    if not args.skip_legacy:
        legacy_path = ROOT.parent / "资料数据和配置文件" / "data" / "liantongzhidao_filter.csv"
        domain_questions = legacy_telecom_questions(legacy_path, args.max_legacy, args.seed)
    if not args.skip_external:
        examples.extend(load_dianjin(args.max_external))
    splits = split_examples(examples, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(args.output / f"{name}.jsonl", rows)
    write_jsonl(args.output / "domain_questions.jsonl", domain_questions)

    stats = {
        "total": len(examples),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "sources": Counter(row["source"] for row in examples),
        "domain_questions": len(domain_questions),
        "seed": args.seed,
    }
    (args.output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
