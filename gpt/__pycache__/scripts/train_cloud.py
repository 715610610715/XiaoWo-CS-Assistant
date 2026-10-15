"""基于项目 GPT-2 配置在云端训练中文小沃客服模型。

数据编码遵循课程方案：[CLS] 上下文/问题 [SEP] 客服回答 [SEP]。
默认直接构建小型 GPT2LMHeadModel 从零训练；也可用 --init-model 继续训练已有权重。

=== 与原版差异 ===
原版硬编码依赖 ../资料数据和配置文件/config/config.json 和 ../vocab/vocab.txt，
这两个文件不在代码仓库中，导致云服务器上无法运行。

修改内容：
  1. --vocab 默认值改为 "bert-base-chinese"，优先尝试本地文件，不存在则从 HuggingFace 加载
  2. --config 默认值改为 ""，优先尝试本地文件/HuggingFace，都没有则用默认小型 GPT-2 配置
  3. 模型默认使用 6 层 / 768 维 / 12 头的小型 GPT-2，适合 ~1.3 万条训练数据
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def build_training_samples(dataset: Any, tokenizer: Any, max_length: int) -> list[dict[str, list[int]]]:
    samples: list[dict[str, list[int]]] = []
    for row in dataset:
        history: list[dict[str, str]] = []
        for message in row["messages"]:
            role = message.get("role")
            content = str(message.get("content", "")).strip()
            if role == "system" or not content:
                continue
            if role == "assistant" and history:
                context = "[SEP]".join(item["content"] for item in history[-6:])
                prompt_ids = tokenizer.encode(context, add_special_tokens=False)
                answer_ids = tokenizer.encode(content, add_special_tokens=False)
                # 优先保留完整回答，过长时从左侧裁剪历史上下文。
                answer_ids = answer_ids[: max_length - 3]
                available_prompt = max(0, max_length - len(answer_ids) - 3)
                prompt_ids = prompt_ids[-available_prompt:]
                input_ids = [tokenizer.cls_token_id] + prompt_ids + [tokenizer.sep_token_id] + answer_ids + [tokenizer.sep_token_id]
                answer_start = 2 + len(prompt_ids)
                labels = [-100] * answer_start + answer_ids + [tokenizer.sep_token_id]
                samples.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels})
            history.append({"role": role, "content": content})
    return samples


@dataclass
class DialogueCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="云端 GPT-2 中文客服训练")
    # ── 修改 ①：默认值不再硬编码本地路径 ──
    parser.add_argument("--config", type=str, default="",
                        help="GPT-2 config：留空=默认小模型，或传 HuggingFace 模型名/本地 json 路径")
    parser.add_argument("--vocab", type=str, default="bert-base-chinese",
                        help="Tokenizer：默认 bert-base-chinese，或传本地 vocab.txt 路径")
    # ── 以下参数保持不变 ──
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "xiaowo-gpt2")
    parser.add_argument("--init-model", type=str, default=None, help="可选：从已有 GPT-2 模型目录继续训练")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--resume-checkpoint", default=None, help="Trainer checkpoint 路径或 true")
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset, load_dataset
        from transformers import BertTokenizerFast, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, set_seed
    except ImportError as error:
        raise SystemExit("请先运行 pip install -r requirements-cloud.txt") from error

    set_seed(2026)

    # ── 修改 ②：Tokenizer 初始化，优先本地文件，回退到 HuggingFace ──
    vocab_path = Path(args.vocab)
    if vocab_path.exists():
        print(f"从本地文件加载 tokenizer: {vocab_path}")
        tokenizer = BertTokenizerFast(
            str(vocab_path),
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
    else:
        print(f"从 HuggingFace 加载 tokenizer: {args.vocab}")
        tokenizer = BertTokenizerFast.from_pretrained(args.vocab)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "[PAD]"

    print(f"Tokenizer 加载完成，词表大小: {len(tokenizer)}")

    # 加载数据
    raw = load_dataset("json", data_files={
        "train": str(args.data_dir / "train.jsonl"),
        "validation": str(args.data_dir / "validation.jsonl"),
    })
    train_samples = build_training_samples(raw["train"], tokenizer, args.max_length)
    valid_samples = build_training_samples(raw["validation"], tokenizer, args.max_length)
    if not train_samples or not valid_samples:
        raise SystemExit("训练集或验证集没有可用的 assistant 回答")
    train_dataset = Dataset.from_list(train_samples)
    valid_dataset = Dataset.from_list(valid_samples)
    print(f"训练样本: {len(train_samples)}, 验证样本: {len(valid_samples)}")

    # ── 修改 ③：模型初始化，支持本地文件 / HuggingFace / 默认小模型 ──
    if args.init_model:
        print(f"从已有模型继续训练: {args.init_model}")
        model = GPT2LMHeadModel.from_pretrained(args.init_model)
        if model.config.vocab_size != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
    else:
        config_path = Path(args.config)
        if args.config and config_path.suffix == ".json" and config_path.exists():
            # 本地 config.json 文件
            print(f"从本地文件加载 config: {config_path}")
            config = GPT2Config.from_json_file(str(config_path))
        elif args.config and config_path.exists():
            # 本地模型目录
            print(f"从本地目录加载 config: {config_path}")
            config = GPT2Config.from_pretrained(str(config_path))
        elif args.config:
            # HuggingFace 模型名
            print(f"从 HuggingFace 加载 config: {args.config}")
            config = GPT2Config.from_pretrained(args.config)
        else:
            # 默认：小型 GPT-2，适合 ~1.3 万条训练数据
            print("使用默认小型 GPT-2 配置（6 层 / 768 维 / 12 头）")
            config = GPT2Config(
                n_positions=512,
                n_ctx=512,
                n_embd=768,
                n_layer=6,
                n_head=12,
            )
        config.vocab_size = len(tokenizer)
        config.bos_token_id = tokenizer.cls_token_id
        config.eos_token_id = tokenizer.sep_token_id
        config.pad_token_id = tokenizer.pad_token_id
        model = GPT2LMHeadModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.1f}M")
    model.config.use_cache = False

    # 训练配置
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=True,
        report_to="tensorboard",
        seed=2026,
        data_seed=2026,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DialogueCollator(tokenizer.pad_token_id),
    )

    resume: str | bool | None = args.resume_checkpoint
    if isinstance(resume, str) and resume.lower() == "true":
        resume = True
    train_result = trainer.train(resume_from_checkpoint=resume)
    metrics = trainer.evaluate()
    metrics["perplexity"] = math.exp(metrics["eval_loss"]) if metrics.get("eval_loss", 100) < 20 else float("inf")
    metrics.update({"train_samples": len(train_samples), "validation_samples": len(valid_samples)})

    # 保存最佳模型
    final_dir = args.output / "best_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    trainer.save_state()
    (args.output / "final_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"训练完成，GPT-2 模型：{final_dir}")


if __name__ == "__main__":
    main()
