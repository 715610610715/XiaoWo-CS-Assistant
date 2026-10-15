from __future__ import annotations

from typing import Protocol

from app.models import ChatReply
from app.nlp.retriever import TfidfRetriever
from app.services.history_store import JsonlStore


FALLBACK = (
    "抱歉，我暂时没有找到足够可靠的答案。您可以换一种说法，"
    "或通过中国联通 App 在线客服、10010 人工客服确认。涉及资费、活动和办理资格时，请以当地最新页面为准。"
)


class AnswerGenerator(Protocol):
    def generate(self, question: str) -> str: ...


class ChatService:
    def __init__(
        self,
        retriever: TfidfRetriever,
        history_store: JsonlStore,
        feedback_store: JsonlStore,
        threshold: float = 0.16,
        top_k: int = 3,
        generator: AnswerGenerator | None = None,
    ):
        self.retriever = retriever
        self.history_store = history_store
        self.feedback_store = feedback_store
        self.threshold = threshold
        self.top_k = top_k
        self.generator = generator

    def reply(self, question: str) -> ChatReply:
        query = question.strip()
        if not query:
            raise ValueError("请输入您的问题")
        results = self.retriever.search(query, self.top_k)
        best = results[0] if results else None
        matched = best is not None and best.score >= self.threshold
        suggestions = tuple(result.item.question for result in results[1:] if result.score >= self.threshold * 0.65)
        generated = ""
        if self.generator:
            try:
                generated = self.generator.generate(query)
            except Exception:
                # 模型目录缺失或运行环境不可用时，GUI 仍可用检索答案工作。
                generated = ""
        if generated:
            reply = ChatReply(
                generated,
                best.score if best else 0.0,
                best.item.category if matched else "GPT-2 智能问答",
                best.item.id if matched else None,
                suggestions if matched else (),
            )
        elif matched:
            reply = ChatReply(best.item.answer, best.score, best.item.category, best.item.id, suggestions)
        else:
            reply = ChatReply(FALLBACK, best.score if best else 0.0, "未识别", None)
        self.history_store.append({
            "question": query,
            "answer": reply.text,
            "confidence": round(reply.confidence, 4),
            "category": reply.category,
            "source_id": reply.source_id,
        })
        return reply

    def prepare(self) -> None:
        prepare = getattr(self.generator, "prepare", None)
        if callable(prepare):
            prepare()

    def add_feedback(self, source_id: str | None, question: str, helpful: bool) -> None:
        self.feedback_store.append({"source_id": source_id, "question": question, "helpful": helpful})
