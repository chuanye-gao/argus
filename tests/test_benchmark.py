import unittest

from argus.benchmark import benchmark_document
from argus.chunking import chunk_document
from argus.mapping import map_evidence_to_chunks
from argus.models import ChunkingStrategy, EvaluationSample, EvidenceSpan, PageText, ParsedDocument
from argus.retrieval import evaluate_strategy


def _document() -> ParsedDocument:
    paragraphs = [
        (
            "Payment is due within thirty days after invoice receipt. "
            "If payment is delayed, the vendor may pause service and charge late fees."
        ),
        (
            "Security reviews must verify access controls, audit logs, and encryption settings. "
            "The review owner records evidence before approval."
        ),
        (
            "Support escalation begins with the help desk. Critical outages are routed to the "
            "incident manager and require a post-incident summary."
        ),
    ]
    text = "\n\n".join(paragraphs)
    return ParsedDocument(
        source_doc="contract.txt",
        pages=(
            PageText(
                source_doc="contract.txt",
                page=1,
                text=text,
                start_offset=0,
                end_offset=len(text),
            ),
        ),
    )


class BenchmarkPipelineTests(unittest.TestCase):
    def test_evidence_mapping_uses_overlap_threshold(self):
        document = _document()
        strategy = ChunkingStrategy("fixed_char_120_20", "fixed_char", 120, 20)
        chunks = chunk_document(document, strategy)
        sample = EvaluationSample(
            query="What happens when payment is delayed?",
            evidence_spans=(
                EvidenceSpan(
                    source_doc=document.source_doc,
                    page=1,
                    start_offset=0,
                    end_offset=120,
                ),
            ),
            query_type="single_hop",
            difficulty="easy",
            source_doc=document.source_doc,
        )

        gold = map_evidence_to_chunks(sample, chunks)

        self.assertTrue(gold)
        self.assertEqual(gold[0], chunks[0].chunk_id)

    def test_retrieval_metrics_include_recall_and_mrr(self):
        document = _document()
        strategy = ChunkingStrategy("fixed_char_500_50", "fixed_char", 500, 50)
        chunks = chunk_document(document, strategy)
        sample = EvaluationSample(
            query="What does the document say about late payment service fees?",
            evidence_spans=(
                EvidenceSpan(
                    source_doc=document.source_doc,
                    page=1,
                    start_offset=0,
                    end_offset=130,
                ),
            ),
            query_type="single_hop",
            difficulty="medium",
            source_doc=document.source_doc,
        )

        result = evaluate_strategy(strategy.name, chunks, [sample])
        metrics = result.metrics()

        self.assertEqual(metrics["recall_at_5"], 1.0)
        self.assertGreater(metrics["mrr_at_5"], 0.0)

    def test_benchmark_document_returns_recommendation_report(self):
        document = _document()
        strategies = (
            ChunkingStrategy("fixed_char_160_20", "fixed_char", 160, 20),
            ChunkingStrategy("paragraph_500", "paragraph", 500, 0),
        )

        report = benchmark_document(document, strategies=strategies, max_samples=5)

        self.assertEqual(report["source_doc"], "contract.txt")
        self.assertIn(report["recommended_strategy"]["name"], {strategy.name for strategy in strategies})
        self.assertTrue(report["metrics"])
        self.assertTrue(report["evaluation_samples"])


if __name__ == "__main__":
    unittest.main()
