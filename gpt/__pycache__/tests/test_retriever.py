import tempfile
import unittest
from pathlib import Path

from app.config import PROJECT_ROOT
from app.data import CsvKnowledgeRepository
from app.nlp import TfidfRetriever
from app.services import ChatService, EvaluationService
from app.services.history_store import JsonlStore


class RetrieverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.items = CsvKnowledgeRepository(PROJECT_ROOT / "data" / "knowledge_base.csv").load()

    def test_common_question_retrieves_correct_intent(self) -> None:
        retriever = TfidfRetriever().fit(self.items)
        result = retriever.search("我还剩下多少流量", top_k=1)[0]
        self.assertEqual(result.item.id, "DATA001")

    def test_chat_logs_reply_without_gui_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history = JsonlStore(Path(directory) / "history.jsonl")
            feedback = JsonlStore(Path(directory) / "feedback.jsonl")
            service = ChatService(TfidfRetriever().fit(self.items), history, feedback)
            reply = service.reply("手机卡丢失了")
            self.assertEqual(reply.source_id, "SIM001")
            self.assertEqual(len(history.load()), 1)

    def test_tuned_model_reaches_reasonable_accuracy(self) -> None:
        result = EvaluationService().evaluate_all(self.items)[-1]
        self.assertGreaterEqual(result.top1_accuracy, 0.75)
        self.assertGreaterEqual(result.top3_accuracy, 0.90)

    def test_chat_can_use_decoupled_gpt2_generator(self) -> None:
        class FakeGenerator:
            def generate(self, question: str) -> str:
                return f"模型回答：{question}"

        with tempfile.TemporaryDirectory() as directory:
            service = ChatService(
                TfidfRetriever().fit(self.items),
                JsonlStore(Path(directory) / "history.jsonl"),
                JsonlStore(Path(directory) / "feedback.jsonl"),
                generator=FakeGenerator(),
            )
            reply = service.reply("怎么查询剩余流量")
            self.assertEqual(reply.text, "模型回答：怎么查询剩余流量")
            free_reply = service.reply("你好，可以介绍一下你自己吗")
            self.assertEqual(free_reply.text, "模型回答：你好，可以介绍一下你自己吗")


if __name__ == "__main__":
    unittest.main()
