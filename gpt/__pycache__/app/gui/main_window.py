from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from app.config import AppConfig
from app.models import ChatReply, EvaluationResult, KnowledgeItem
from app.services import ChatService, EvaluationService


class MainWindow:
    COLORS = {
        "navy": "#102A56",
        "primary": "#246BFD",
        "primary_hover": "#1854D8",
        "accent": "#2EC5CE",
        "background": "#EEF3FA",
        "surface": "#FFFFFF",
        "panel": "#F7F9FC",
        "border": "#DCE5F2",
        "text": "#17233C",
        "muted": "#6B7A90",
        "user": "#DFEAFF",
        "bot": "#FFFFFF",
        "success": "#16A36A",
        "warning": "#F59E0B",
    }
    QUICK_QUESTIONS = (
        "怎么查询剩余流量？",
        "手机突然没有信号怎么办？",
        "宽带断网怎么排查？",
        "套餐可以在线变更吗？",
        "如何联系人工客服？",
    )

    def __init__(
        self,
        root: tk.Tk,
        config: AppConfig,
        items: list[KnowledgeItem],
        chat_service: ChatService,
        evaluation_service: EvaluationService,
    ):
        self.root = root
        self.config = config
        self.items = items
        self.chat_service = chat_service
        self.evaluation_service = evaluation_service
        self.last_question = ""
        self.last_reply: ChatReply | None = None
        self.suggestion_buttons: list[ttk.Button] = []
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.model_ready = False
        self._configure_root()
        self._configure_styles()
        self._build_layout()
        self._write_bot("您好，我是小沃智能客服。请输入问题，或从左侧选择常用服务。")
        self.root.after(150, self._start_model_prepare)
        self.root.after(100, self._poll_events)

    def _configure_root(self) -> None:
        self.root.title(self.config.app_name)
        self.root.geometry("1180x790")
        self.root.minsize(980, 680)
        self.root.configure(bg=self.COLORS["background"])

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background=self.COLORS["background"])
        style.configure("Surface.TFrame", background=self.COLORS["surface"])
        style.configure("Panel.TFrame", background=self.COLORS["panel"])
        style.configure(
            "Primary.TButton",
            background=self.COLORS["primary"],
            foreground="white",
            padding=(20, 11),
            borderwidth=0,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.COLORS["primary_hover"]), ("disabled", "#9BB7ED")],
        )
        style.configure(
            "Secondary.TButton",
            background="#EAF0F8",
            foreground=self.COLORS["navy"],
            padding=(13, 9),
            borderwidth=0,
        )
        style.configure(
            "Chip.TButton",
            background="#E8F0FF",
            foreground=self.COLORS["primary_hover"],
            padding=(10, 6),
            borderwidth=0,
        )
        style.configure("TNotebook", background=self.COLORS["background"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=self.COLORS["background"],
            foreground=self.COLORS["muted"],
            padding=(22, 10),
            font=("Microsoft YaHei UI", 10),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.COLORS["surface"])],
            foreground=[("selected", self.COLORS["primary"])],
        )
        style.configure("Treeview", rowheight=30, font=("Microsoft YaHei UI", 9))
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg=self.COLORS["navy"], height=96)
        header.pack(fill="x")
        header.pack_propagate(False)

        logo = tk.Canvas(header, width=54, height=54, bg=self.COLORS["navy"], highlightthickness=0)
        logo.create_oval(3, 3, 51, 51, fill=self.COLORS["primary"], outline="")
        logo.create_text(27, 27, text="沃", fill="white", font=("Microsoft YaHei UI", 19, "bold"))
        logo.pack(side="left", padx=(28, 14), pady=20)

        title_box = tk.Frame(header, bg=self.COLORS["navy"])
        title_box.pack(side="left", pady=19)
        tk.Label(
            title_box,
            text="联通小沃智能客服",
            bg=self.COLORS["navy"],
            fg="white",
            font=("Microsoft YaHei UI", 20, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="GPT-2 中文客服模型 · 本地安全运行",
            bg=self.COLORS["navy"],
            fg="#B9C8E3",
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", pady=(3, 0))

        self.status_label = tk.Label(
            header,
            text="  ●  GPT-2 模型预热中  ",
            bg="#183C70",
            fg="#FFD690",
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=10,
            pady=7,
        )
        self.status_label.pack(side="right", padx=30)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=22, pady=(16, 22))
        self.chat_tab = ttk.Frame(self.notebook, style="Surface.TFrame")
        self.evaluation_tab = ttk.Frame(self.notebook, style="Surface.TFrame")
        self.about_tab = ttk.Frame(self.notebook, style="Surface.TFrame")
        self.notebook.add(self.chat_tab, text="  智能客服  ")
        self.notebook.add(self.evaluation_tab, text="  模型评估  ")
        self.notebook.add(self.about_tab, text="  系统说明  ")
        self._build_chat_tab()
        self._build_evaluation_tab()
        self._build_about_tab()

    def _build_chat_tab(self) -> None:
        sidebar = tk.Frame(self.chat_tab, bg=self.COLORS["panel"], width=240)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Label(
            sidebar,
            text="常用服务",
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 13, "bold"),
        ).pack(anchor="w", padx=20, pady=(24, 5))
        tk.Label(
            sidebar,
            text="点击即可快速提问",
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", padx=20, pady=(0, 14))
        for question in self.QUICK_QUESTIONS:
            button = tk.Button(
                sidebar,
                text=question,
                command=lambda value=question: self._ask_suggestion(value),
                bg="white",
                fg=self.COLORS["navy"],
                activebackground="#E8F0FF",
                activeforeground=self.COLORS["primary"],
                relief="flat",
                bd=0,
                anchor="w",
                padx=12,
                pady=10,
                wraplength=185,
                justify="left",
                cursor="hand2",
                font=("Microsoft YaHei UI", 9),
            )
            button.pack(fill="x", padx=16, pady=4)
        privacy = tk.Frame(sidebar, bg="#E9F8F4")
        privacy.pack(side="bottom", fill="x", padx=16, pady=18)
        tk.Label(
            privacy,
            text="✓  隐私保护",
            bg="#E9F8F4",
            fg=self.COLORS["success"],
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(10, 2))
        tk.Label(
            privacy,
            text="对话与模型均在本机运行",
            bg="#E9F8F4",
            fg="#477568",
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="w", padx=12, pady=(0, 10))

        main = tk.Frame(self.chat_tab, bg=self.COLORS["surface"])
        main.pack(side="left", fill="both", expand=True)
        conversation_header = tk.Frame(main, bg=self.COLORS["surface"], height=55)
        conversation_header.pack(fill="x", padx=22, pady=(14, 0))
        conversation_header.pack_propagate(False)
        tk.Label(
            conversation_header,
            text="在线对话",
            bg=self.COLORS["surface"],
            fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 13, "bold"),
        ).pack(side="left", pady=12)
        tk.Label(
            conversation_header,
            text=" GPT-2 ",
            bg="#E8F0FF",
            fg=self.COLORS["primary"],
            font=("Microsoft YaHei UI", 8, "bold"),
            padx=7,
            pady=3,
        ).pack(side="left", padx=10, pady=14)

        conversation = tk.Frame(main, bg=self.COLORS["panel"], highlightbackground=self.COLORS["border"], highlightthickness=1)
        conversation.pack(fill="both", expand=True, padx=22, pady=(0, 12))
        self.chat_text = tk.Text(
            conversation,
            wrap="word",
            state="disabled",
            relief="flat",
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 10),
            padx=20,
            pady=16,
            spacing1=5,
            spacing3=8,
            cursor="arrow",
        )
        scrollbar = ttk.Scrollbar(conversation, orient="vertical", command=self.chat_text.yview)
        self.chat_text.configure(yscrollcommand=scrollbar.set)
        self.chat_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.chat_text.tag_configure("user_name", foreground=self.COLORS["primary"], font=("Microsoft YaHei UI", 9, "bold"), justify="right")
        self.chat_text.tag_configure("user", background=self.COLORS["user"], lmargin1=180, lmargin2=180, rmargin=10, justify="right", spacing1=6, spacing3=9)
        self.chat_text.tag_configure("bot_name", foreground=self.COLORS["success"], font=("Microsoft YaHei UI", 9, "bold"))
        self.chat_text.tag_configure("bot", background=self.COLORS["bot"], lmargin1=10, lmargin2=10, rmargin=180, spacing1=6, spacing3=9)
        self.chat_text.tag_configure("meta", foreground=self.COLORS["muted"], font=("Microsoft YaHei UI", 8))

        input_panel = tk.Frame(main, bg=self.COLORS["surface"])
        input_panel.pack(fill="x", padx=22, pady=(0, 16))
        self.suggestion_frame = tk.Frame(input_panel, bg=self.COLORS["surface"])
        self.suggestion_frame.pack(fill="x", pady=(0, 7))
        input_row = tk.Frame(input_panel, bg=self.COLORS["surface"])
        input_row.pack(fill="x")
        self.question_entry = tk.Entry(
            input_row,
            font=("Microsoft YaHei UI", 11),
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=self.COLORS["border"],
            highlightcolor=self.COLORS["primary"],
        )
        self.question_entry.pack(side="left", fill="x", expand=True, ipady=10)
        self.question_entry.bind("<Return>", self._on_send)
        self.send_button = ttk.Button(input_row, text="发送问题", style="Primary.TButton", command=self._on_send)
        self.send_button.pack(side="left", padx=(10, 0))
        ttk.Button(input_row, text="清空对话", style="Secondary.TButton", command=self._clear_chat).pack(side="left", padx=(8, 0))
        self.input_hint = tk.Label(
            input_panel,
            text="Enter 发送 · 首次使用需等待 GPT-2 模型预热",
            bg=self.COLORS["surface"],
            fg=self.COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
        )
        self.input_hint.pack(anchor="w", pady=(6, 0))

    def _build_evaluation_tab(self) -> None:
        frame = ttk.Frame(self.evaluation_tab, style="Surface.TFrame", padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="模型参数对比",
            background=self.COLORS["surface"],
            foreground=self.COLORS["text"],
            font=("Microsoft YaHei UI", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="比较不同文本向量配置的 Top-1、Top-3 与 MRR；GPT-2 训练损失可通过云端 TensorBoard 查看。",
            background=self.COLORS["surface"],
            foreground=self.COLORS["muted"],
        ).pack(anchor="w", pady=(4, 10))
        ttk.Button(frame, text="重新评估", style="Primary.TButton", command=self._run_evaluation).pack(anchor="e")
        self.chart = tk.Canvas(frame, bg="white", height=330, highlightthickness=1, highlightbackground=self.COLORS["border"])
        self.chart.pack(fill="x", pady=12)
        self.metric_tree = ttk.Treeview(frame, columns=("model", "top1", "top3", "mrr", "count"), show="headings", height=4)
        for key, title, width in (("model", "配置", 220), ("top1", "Top-1", 120), ("top3", "Top-3", 120), ("mrr", "MRR", 120), ("count", "测试数", 100)):
            self.metric_tree.heading(key, text=title)
            self.metric_tree.column(key, width=width, anchor="center")
        self.metric_tree.pack(fill="x")
        self._run_evaluation()

    def _build_about_tab(self) -> None:
        card = tk.Frame(self.about_tab, bg=self.COLORS["surface"], padx=40, pady=36)
        card.pack(fill="both", expand=True)
        tk.Label(card, text="关于小沃智能客服", bg="white", fg=self.COLORS["text"], font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w")
        text = (
            "系统基于 GPT-2 中文生成模型构建，采用 [CLS]问题[SEP]回答[SEP] 的课程训练格式。\n\n"
            "界面、模型推理、对话历史和反馈均解耦实现。模型在后台预热，问答在工作线程执行，因此不会阻塞界面。\n\n"
            "涉及资费、优惠、覆盖及办理资格时，请以中国联通 App、10010 或当地营业厅最新信息为准。"
        )
        tk.Label(card, text=text, wraplength=780, justify="left", bg="white", fg=self.COLORS["muted"], font=("Microsoft YaHei UI", 11), pady=20).pack(anchor="w")

    def _start_model_prepare(self) -> None:
        threading.Thread(target=self._prepare_model, daemon=True).start()

    def _prepare_model(self) -> None:
        try:
            self.chat_service.prepare()
        except Exception as error:
            self.events.put(("model_error", error))
        else:
            self.events.put(("model_ready", None))

    def _on_send(self, _event: object = None) -> None:
        question = self.question_entry.get().strip()
        if not question or self.busy:
            return
        self.question_entry.delete(0, "end")
        self._write_user(question)
        self.busy = True
        self.send_button.configure(state="disabled", text="生成中…")
        self.input_hint.configure(text="GPT-2 正在理解并生成回答，请稍候…", fg=self.COLORS["primary"])
        self.status_label.configure(text="  ●  GPT-2 正在回答  ", fg="#BDEBFF")
        threading.Thread(target=self._request_reply, args=(question,), daemon=True).start()

    def _request_reply(self, question: str) -> None:
        try:
            reply = self.chat_service.reply(question)
        except Exception as error:
            self.events.put(("reply_error", error))
            return
        self.events.put(("reply", (question, reply)))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "model_ready":
                    self.model_ready = True
                    if not self.busy:
                        self.status_label.configure(text="  ●  GPT-2 已就绪  ", fg="#A7F3D0")
                        self.input_hint.configure(text="Enter 发送 · GPT-2 模型已就绪", fg=self.COLORS["success"])
                elif kind == "model_error":
                    self.status_label.configure(text="  ●  基础问答模式  ", fg="#FFD690")
                    self.input_hint.configure(text="GPT-2 暂不可用，已自动切换基础问答模式", fg=self.COLORS["warning"])
                elif kind == "reply_error":
                    self._finish_request()
                    messagebox.showerror("问答失败", str(payload), parent=self.root)
                elif kind == "reply":
                    question, reply = payload
                    self.last_question, self.last_reply = question, reply
                    self._write_bot(reply.text, reply)
                    self._show_suggestions(reply.suggestions)
                    self._finish_request()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _finish_request(self) -> None:
        self.busy = False
        self.send_button.configure(state="normal", text="发送问题")
        if self.model_ready:
            self.status_label.configure(text="  ●  GPT-2 已就绪  ", fg="#A7F3D0")
            self.input_hint.configure(text="Enter 发送 · GPT-2 模型已就绪", fg=self.COLORS["success"])
        else:
            self.status_label.configure(text="  ●  服务可用  ", fg="#FFD690")
            self.input_hint.configure(text="Enter 发送", fg=self.COLORS["muted"])
        self.question_entry.focus_set()

    def _write_user(self, text: str) -> None:
        self._append("我\n", "user_name")
        self._append(f"{text}\n\n", "user")

    def _write_bot(self, text: str, reply: ChatReply | None = None) -> None:
        self._append("小沃客服\n", "bot_name")
        self._append(f"{text}\n", "bot")
        if reply:
            self._append(f"业务：{reply.category}  ·  置信度：{reply.confidence:.0%}\n", "meta")
            self._append("这条回答有帮助吗？  ", "meta")
            self._insert_feedback_button("有帮助", True)
            self._insert_feedback_button("需改进", False)
        self._append("\n\n", "meta")

    def _append(self, text: str, tag: str) -> None:
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", text, tag)
        self.chat_text.configure(state="disabled")
        self.chat_text.see("end")

    def _insert_feedback_button(self, label: str, helpful: bool) -> None:
        button = ttk.Button(self.chat_text, text=label, style="Chip.TButton", command=lambda: self._feedback(helpful))
        self.chat_text.configure(state="normal")
        self.chat_text.window_create("end", window=button, padx=3)
        self.chat_text.configure(state="disabled")

    def _feedback(self, helpful: bool) -> None:
        if self.last_reply:
            self.chat_service.add_feedback(self.last_reply.source_id, self.last_question, helpful)
            messagebox.showinfo("感谢反馈", "您的反馈已保存。", parent=self.root)

    def _show_suggestions(self, suggestions: tuple[str, ...]) -> None:
        for widget in self.suggestion_frame.winfo_children():
            widget.destroy()
        self.suggestion_buttons.clear()
        if suggestions:
            tk.Label(self.suggestion_frame, text="相关问题：", bg="white", fg=self.COLORS["muted"], font=("Microsoft YaHei UI", 8)).pack(side="left", padx=(0, 5))
        for question in suggestions[:2]:
            button = ttk.Button(self.suggestion_frame, text=question, style="Chip.TButton", command=lambda value=question: self._ask_suggestion(value))
            button.pack(side="left", padx=(0, 7))
            self.suggestion_buttons.append(button)

    def _ask_suggestion(self, question: str) -> None:
        if self.busy:
            return
        self.question_entry.delete(0, "end")
        self.question_entry.insert(0, question)
        self._on_send()

    def _clear_chat(self) -> None:
        self.chat_text.configure(state="normal")
        self.chat_text.delete("1.0", "end")
        self.chat_text.configure(state="disabled")
        self._write_bot("对话已清空。请问有什么可以帮您？")

    def _run_evaluation(self) -> None:
        try:
            results = self.evaluation_service.evaluate_all(self.items)
        except Exception as error:
            messagebox.showerror("评估失败", str(error), parent=self.root)
            return
        for row in self.metric_tree.get_children():
            self.metric_tree.delete(row)
        for result in results:
            self.metric_tree.insert("", "end", values=(result.model_name, f"{result.top1_accuracy:.1%}", f"{result.top3_accuracy:.1%}", f"{result.mrr:.3f}", result.sample_count))
        self._draw_chart(results)

    def _draw_chart(self, results: list[EvaluationResult]) -> None:
        canvas = self.chart
        canvas.delete("all")
        width = max(canvas.winfo_width(), 820)
        left, top, bottom = 60, 35, 265
        canvas.create_line(left, top, left, bottom, fill="#94A3B8")
        canvas.create_line(left, bottom, width - 25, bottom, fill="#94A3B8")
        for tick in range(0, 101, 20):
            y = bottom - (bottom - top) * tick / 100
            canvas.create_line(left, y, width - 25, y, fill="#E8EDF5")
            canvas.create_text(left - 9, y, text=f"{tick}%", anchor="e", fill=self.COLORS["muted"], font=("Arial", 8))
        group_width = (width - left - 35) / max(1, len(results))
        colors = (self.COLORS["primary"], self.COLORS["accent"])
        for index, result in enumerate(results):
            center = left + group_width * (index + 0.5)
            for offset, (value, color) in enumerate(((result.top1_accuracy, colors[0]), (result.top3_accuracy, colors[1]))):
                x1 = center - 34 + offset * 38
                y1 = bottom - (bottom - top) * value
                canvas.create_rectangle(x1, y1, x1 + 30, bottom, fill=color, outline="")
                canvas.create_text(x1 + 15, y1 - 9, text=f"{value:.0%}", fill=self.COLORS["text"], font=("Arial", 8, "bold"))
            canvas.create_text(center, bottom + 20, text=result.model_name, fill=self.COLORS["text"], font=("Microsoft YaHei UI", 9))
        canvas.create_rectangle(width - 180, 13, width - 168, 25, fill=colors[0], outline="")
        canvas.create_text(width - 162, 19, text="Top-1", anchor="w", fill=self.COLORS["muted"])
        canvas.create_rectangle(width - 105, 13, width - 93, 25, fill=colors[1], outline="")
        canvas.create_text(width - 87, 19, text="Top-3", anchor="w", fill=self.COLORS["muted"])

    def show(self) -> None:
        self.question_entry.focus_set()
        self.root.mainloop()
