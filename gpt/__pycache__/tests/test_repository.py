import tempfile
import unittest
from pathlib import Path

from app.config import PROJECT_ROOT
from app.data import CsvKnowledgeRepository, KnowledgeDataError


class RepositoryTest(unittest.TestCase):
    def test_builtin_knowledge_base_is_valid_and_new(self) -> None:
        items = CsvKnowledgeRepository(PROJECT_ROOT / "data" / "knowledge_base.csv").load()
        self.assertGreaterEqual(len(items), 35)
        self.assertGreaterEqual(len({item.category for item in items}), 7)
        self.assertTrue(all(item.test_query for item in items))

    def test_invalid_schema_has_readable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_text("id,question\n1,测试\n", encoding="utf-8")
            with self.assertRaises(KnowledgeDataError):
                CsvKnowledgeRepository(path).load()


if __name__ == "__main__":
    unittest.main()
