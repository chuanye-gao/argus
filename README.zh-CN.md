# Argus

语言：[English](README.md)

Argus 是一个面向 PDF 的 RAG 分块基准工具。给定一个 PDF，它会解析文本、
尝试多种分块策略、生成与策略无关的检索评估集、把源文本证据范围映射到
各策略下的 gold chunks，评估检索质量，并推荐一个分块配置。

它不是聊天机器人，也不是 GraphRAG 的替代品。它关注索引前的这一步：

```text
PDF -> parse text -> chunk several ways -> generate evidence-span queries
    -> map spans to chunks -> retrieve -> score -> recommend
```

旧版 parent/child synthetic query generator 仍然可以通过默认 JSON CLI 和
`/generate` API 使用。

## 安装

开发时建议使用仓库内的本地虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

```bash
pip install -e ".[all]"
```

如果只需要 PDF benchmark 模式，需要安装 PyMuPDF：

```bash
pip install -e ".[pdf]"
```

## PDF Benchmark CLI

```bash
python -m argus your-file.pdf --benchmark-pdf --pretty -o report.json
```

常用参数：

```bash
python -m argus your-file.pdf --benchmark-pdf --max-samples 12 --top-k 5 --pretty
```

本地 smoke test 示例：

```powershell
python -m argus samples\pdf\argus_incident_runbook.pdf --benchmark-pdf --max-samples 12 --top-k 5 --pretty -o samples\outputs\argus_incident_runbook_benchmark_report.json
```

如果想在没有 PDF parser 的情况下演示或测试，可以对 UTF-8 文本文件运行同一套
benchmark pipeline：

```bash
python -m argus task.md --benchmark-text --pretty
```

## 输出结构

```json
{
  "source_doc": "example.pdf",
  "recommended_strategy": {
    "name": "recursive_character_800_100",
    "method": "recursive_character",
    "chunk_size": 800,
    "overlap": 100
  },
  "summary": {
    "reason": "The selected strategy achieved the strongest retrieval quality while keeping chunk count moderate.",
    "evaluation_sample_count": 12
  },
  "metrics": [
    {
      "strategy": "fixed_char_500_50",
      "recall_at_1": 0.42,
      "recall_at_3": 0.68,
      "recall_at_5": 0.76,
      "mrr_at_5": 0.55,
      "ndcg_at_5": 0.61,
      "chunk_count": 128,
      "avg_chunk_length": 492.0,
      "avg_retrieved_context_length": 2450.0,
      "estimated_embedding_token_cost": 15744,
      "avg_latency_ms": 12.4
    }
  ],
  "best_by_query_type": {
    "single_hop": "fixed_char_800_100"
  },
  "best_by_difficulty": {
    "medium": "fixed_char_800_100"
  },
  "failure_analysis": [],
  "evaluation_samples": []
}
```

## 阅读报告

建议先看这些字段：

- `recommended_strategy`
- `summary.reason`
- `metrics[].recall_at_5`
- `metrics[].mrr_at_5`
- `metrics[].ndcg_at_5`
- `metrics[].chunk_count`
- `metrics[].avg_chunk_length`
- `metrics[].estimated_embedding_token_cost`
- `metrics[].avg_latency_ms`
- `failure_analysis`
- `evaluation_samples`

对于仓库里的 smoke-test PDF，当前 MVP 会评估 11 个策略，并推荐
`paragraph_1200`。原因是它在生成的 evidence samples 上取得了完整的
Recall@5、MRR@5 和 NDCG@5，同时保持了较低的 chunk 数量。

## 已实现的 MVP

- PDF 解析，保留页码和文档级字符 offset。
- 固定字符数分块：500、800、1200 字符窗口，搭配 50、100、200 字符 overlap。
- 基于段落和句子边界的 recursive character chunking。
- 基于段落的 chunking。
- 与策略无关的 evaluation samples，包含源文本 evidence spans。
- evidence-to-chunk 映射，使用 50% overlap 规则和最大 overlap fallback。
- 无额外依赖的 lexical retrieval index。
- Recall@1、Recall@3、Recall@5、MRR@5、NDCG@5、chunk count、
  average chunk length、estimated token cost、retrieved context length、
  latency diagnostics。
- 推荐报告，包含整体最佳策略、分组最佳策略和失败查询示例。

## 现有 JSON Generator

原有 parent/child generator 仍然是默认 CLI 行为：

```bash
python -m argus samples/robotics.json --pretty
python -m argus --batch samples/batch_input.jsonl --pretty -o output.json
```

HTTP API 仍然可以这样启动：

```bash
python -m argus --serve
```

接口：

- `GET /health`
- `POST /generate`
- `POST /batch`

## 开发

```bash
python -m pytest tests/ -v
```
