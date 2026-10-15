"""命令行运行参数对比评估。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.data import CsvKnowledgeRepository  # noqa: E402
from app.services import EvaluationService  # noqa: E402


def main() -> None:
    config = load_config()
    items = CsvKnowledgeRepository(config.data_path).load()
    print(f"测试样本：{len(items)}")
    print(f"{'配置':<20} {'Top-1':>8} {'Top-3':>8} {'MRR':>8}")
    for result in EvaluationService().evaluate_all(items):
        print(f"{result.model_name:<20} {result.top1_accuracy:>8.1%} {result.top3_accuracy:>8.1%} {result.mrr:>8.3f}")


if __name__ == "__main__":
    main()
