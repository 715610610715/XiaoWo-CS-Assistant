from __future__ import annotations

import json
from pathlib import Path
from threading import Lock


class Gpt2Generator:
    """课程 GPT-2 模型适配器；延迟加载，业务服务不依赖 Transformers。"""

    def __init__(
        self,
        model_path: Path,
        vocab_path: Path,
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ):
        self.model_path = Path(model_path)
        self.vocab_path = Path(vocab_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None
        self._device = None
        self._lock = Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"GPT-2 模型不存在：{self.model_path}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"GPT-2 词表不存在：{self.vocab_path}")
        try:
            import torch
            from transformers import BertTokenizerFast, GPT2Config, GPT2LMHeadModel
        except ImportError as error:
            raise RuntimeError("使用 GPT-2 客户端需要安装 torch 和 transformers") from error

        tokenizer = BertTokenizerFast(
            str(self.vocab_path),
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_data = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
        config_data.update({
            "bos_token_id": tokenizer.cls_token_id,
            "eos_token_id": tokenizer.sep_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        })
        config = GPT2Config.from_dict(config_data)
        model = GPT2LMHeadModel.from_pretrained(str(self.model_path), config=config).to(device)
        model.generation_config.bos_token_id = tokenizer.cls_token_id
        model.generation_config.eos_token_id = tokenizer.sep_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        self._tokenizer, self._model, self._device = tokenizer, model, device

    def prepare(self) -> None:
        """供 GUI 在后台预热模型。"""
        with self._lock:
            self._load()

    def generate(self, question: str) -> str:
        with self._lock:
            self._load()
            import torch

            tokenizer, model = self._tokenizer, self._model
            prompt = [tokenizer.cls_token_id] + tokenizer.encode(question, add_special_tokens=False) + [tokenizer.sep_token_id]
            input_ids = torch.tensor([prompt], dtype=torch.long, device=self._device)
            attention_mask = torch.ones_like(input_ids)
            for do_sample in (True, False):
                generation = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": self.max_new_tokens,
                    "min_new_tokens": min(5, self.max_new_tokens),
                    "do_sample": do_sample,
                    "repetition_penalty": 1.1,
                    "no_repeat_ngram_size": 3,
                    "eos_token_id": tokenizer.sep_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if do_sample:
                    generation.update({"temperature": self.temperature, "top_p": self.top_p})
                with torch.inference_mode():
                    output = model.generate(**generation)
                answer_ids = output[0, len(prompt):].tolist()
                if tokenizer.sep_token_id in answer_ids:
                    answer_ids = answer_ids[:answer_ids.index(tokenizer.sep_token_id)]
                answer = tokenizer.decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ", "").strip()
                if answer:
                    return answer
            return ""
