from __future__ import annotations

import tkinter as tk
from pathlib import Path

from app.config import AppConfig, load_config
from app.data import CsvKnowledgeRepository
from app.gui import MainWindow
from app.models import KnowledgeItem
from app.nlp import TfidfRetriever
from app.services import ChatService, EvaluationService
from app.services.history_store import JsonlStore
from app.services.gpt2_generator import Gpt2Generator


class CustomerServiceApplication:
    def __init__(self, config: AppConfig):
        self.config = config
        self.root = tk.Tk()
        self.data_path = config.data_path
        self.items: list[KnowledgeItem] = []
        self.chat_service: ChatService
        self._load_services(self.data_path)
        self.window = MainWindow(
            root=self.root,
            config=config,
            items=self.items,
            chat_service=self.chat_service,
            evaluation_service=EvaluationService(),
        )

    def _load_services(self, path: Path) -> None:
        self.items = CsvKnowledgeRepository(path).load()
        retriever = TfidfRetriever(ngram=2, keyword_weight=2.2).fit(self.items)
        generator = None
        if self.config.use_gpt2:
            generator = Gpt2Generator(
                model_path=self.config.model_path,
                vocab_path=self.config.vocab_path,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        self.chat_service = ChatService(
            retriever=retriever,
            history_store=JsonlStore(self.config.history_path),
            feedback_store=JsonlStore(self.config.feedback_path),
            threshold=self.config.similarity_threshold,
            top_k=self.config.top_k,
            generator=generator,
        )

    def import_knowledge(self, path: Path) -> tuple[list[KnowledgeItem], ChatService]:
        self._load_services(path)
        self.data_path = path
        return self.items, self.chat_service

    def run(self) -> None:
        self.window.show()


def build_application(config_path: Path | None = None) -> CustomerServiceApplication:
    return CustomerServiceApplication(load_config(config_path))
