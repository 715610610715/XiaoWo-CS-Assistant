# 联通小沃智能客服（GUI）

这是依据 `AI大模型联通小沃客服系统.pdf` 重构的桌面项目。项目不再使用原始的 2018 年乱码语料，而是内置一套重新编写的联通业务教学问答数据，并实现数据校验、文本向量化、模型评估、参数对比图表、GUI 问答和历史反馈。

## 运行

要求 Python 3.10+。Windows 官方 Python 通常已包含 Tkinter；GPT-2 推理还需要安装模型依赖。

```powershell
cd project_2
python -m pip install -r requirements.txt
python main.py
```

如果使用 PyCharm，请确认项目解释器安装了 `torch` 和 `transformers`。当前随项目创建的旧 `.venv` 默认未安装这两个包；也可以直接选择已经安装 `torch 2.8` 和 `transformers 4.57` 的系统 Python 解释器。

命令行查看参数调优结果：

```powershell
python scripts/evaluate.py
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

## 云端大数据训练

内置 41 条人工编写数据用于领域准确性和 GUI 验收，不单独承担大模型语言能力训练。完整流水线默认组合：

- 本地联通知识的标准问法与同义问法；
- 项目原有 `liantongzhidao_filter.csv` 共 203,235 条，筛选、去重后默认抽样 30,000 条真实用户问法，写入 `domain_questions.jsonl`；因回答存在过时政策，绝不默认混入生成式 SFT；
- `DianJin/DianJin-CSC-Data` 约 13.1k 条中文多轮专业客服对话（MIT），用于学习服务流程、语气和策略；
- 数据清单见 `config/data_sources.json`。英文电信 20 万对话和许可需另行确认的数据默认不下载、不混训。

在带 NVIDIA GPU 且可联网的 Linux 云服务器执行：

```bash
pip install -r requirements-cloud.txt
python scripts/prepare_cloud_data.py
python scripts/train_cloud.py
```

如果只想先验证流水线：

```bash
python scripts/prepare_cloud_data.py --max-external 500
python scripts/train_cloud.py --epochs 1 --batch-size 2
```

处理脚本会做手机号、身份证号、长账号脱敏，生成式客服对话按知识点/完整会话分组切成 90%/5%/5%，输出到 `data/processed/`。训练脚本将 `messages` 转换成课程要求的 `[CLS]问题/上下文[SEP]回答[SEP]`，使用上一级 `资料数据和配置文件/config/config.json` 从零构建 `GPT2LMHeadModel`，只对客服回答 token 计算损失，并保存验证损失最小的完整 GPT-2 模型。

如果要从项目根目录现有的 `mymodel` 权重继续训练，而不是随机初始化：

```bash
python scripts/train_cloud.py --init-model ../mymodel
```

可用 `--max-legacy 50000` 增加领域问法上限，或用 `--skip-legacy` 排除。`domain_questions.jsonl` 可用于意图分类、检索器或领域继续预训练，不能当作客服答案 SFT 数据。该语料来源于百度知道公开问答并经原数据集脱敏，适合课程研究；如果模型将用于商业场景，需要另行核对数据许可并优先换成企业自有、获授权且保持更新的业务知识。

## 功能

- 智能问答：GPT-2 生成回答；TF-IDF 负责意图匹配、置信度阈值、关联问题推荐和低置信度安全兜底。
- 全新数据：覆盖套餐、账单、流量、网络、宽带、号码、国际业务和客户服务。
- 简洁客户端：界面只保留智能客服、模型评估和系统说明，不展示内部训练数据或管理入口。
- 模型评估：使用独立测试问法比较 1-gram、2-gram、关键词加权模型，显示 Top-1、Top-3、MRR 和柱状图。
- 本地记录：聊天与反馈追加保存到 `runtime/*.jsonl`，不上传用户内容。

## 解耦结构

```text
app/data       CSV 数据读取与校验
app/nlp        文本标准化、向量化与检索
app/services   GPT-2 生成适配器、对话、评估和历史记录用例
app/gui        Tkinter 展示与交互
app/bootstrap  依赖组装，负责替换数据或模型
```

GPT-2 权重由 `config/settings.json` 的 `model_path` 指定，默认读取项目上一级的 `mymodel`。云端训练结束后，将其改为 `outputs/xiaowo-gpt2/best_model` 即可。模型通过适配器注入，对话服务和 GUI 不直接依赖 Transformers。

## 自定义数据格式

CSV 必须为 UTF-8，字段如下：

| 字段 | 说明 |
|---|---|
| `id` | 唯一知识编号 |
| `category` | 业务分类 |
| `question` | 标准问题 |
| `answer` | 客服回答 |
| `keywords` | `|` 分隔的同义关键词 |
| `test_query` | 仅用于评估的独立问法 |

内置内容为教学演示知识。实际资费、活动、覆盖及办理资格变化较快，请以中国联通官方渠道和当地最新政策为准。
