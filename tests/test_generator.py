import json
import unittest

from argus import DatasetGenerationError, generate_dataset


class GenerateDatasetTests(unittest.TestCase):
    def setUp(self):
        self.parent = {
            "parent_id": "parent_000123",
            "source_doc": "robot_manual.pdf",
            "children": [
                {
                    "child_id": "child_000123_01",
                    "text": "机器人启动前需要检查电源连接是否正常。",
                },
                {
                    "child_id": "child_000123_02",
                    "text": "如果控制器无法连接，首先检查网络配置。",
                },
                {
                    "child_id": "child_000123_03",
                    "text": "设备运行期间出现温度告警时，需要查看传感器日志。",
                },
            ],
        }

    def test_generates_valid_dataset_shape(self):
        records = generate_dataset(self.parent)
        encoded = json.dumps(records, ensure_ascii=False)
        decoded = json.loads(encoded)

        self.assertEqual(records, decoded)
        self.assertGreaterEqual(len([r for r in records if r["query_type"] == "single_hop"]), 3)
        self.assertLessEqual(len([r for r in records if r["query_type"] == "single_hop"]), 5)
        self.assertGreaterEqual(
            len([r for r in records if r["query_type"] == "multi_chunk_same_parent"]),
            2,
        )
        self.assertLessEqual(
            len([r for r in records if r["query_type"] == "multi_chunk_same_parent"]),
            3,
        )

    def test_records_only_use_provided_ids(self):
        records = generate_dataset(self.parent)
        allowed_child_ids = {child["child_id"] for child in self.parent["children"]}

        for record in records:
            self.assertEqual(record["gold_parent_id"], self.parent["parent_id"])
            self.assertEqual(record["source_doc"], self.parent["source_doc"])
            self.assertTrue(set(record["gold_child_ids"]).issubset(allowed_child_ids))
            self.assertNotIn(self.parent["parent_id"], record["query"])
            for child_id in allowed_child_ids:
                self.assertNotIn(child_id, record["query"])

    def test_rejects_duplicate_child_ids(self):
        self.parent["children"][1]["child_id"] = self.parent["children"][0]["child_id"]

        with self.assertRaises(DatasetGenerationError):
            generate_dataset(self.parent)

    def test_rejects_single_child_parent(self):
        self.parent["children"] = self.parent["children"][:1]

        with self.assertRaises(DatasetGenerationError):
            generate_dataset(self.parent)


if __name__ == "__main__":
    unittest.main()
