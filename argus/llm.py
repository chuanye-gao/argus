"""LLM-backed dataset generation using an OpenAI-compatible API.

Environment variables
---------------------
ARGUS_LLM_API_KEY   : API key (fallback: OPENAI_API_KEY)
ARGUS_LLM_BASE_URL  : Base URL for OpenAI-compatible endpoint (optional)
ARGUS_LLM_MODEL     : Model name (default: gpt-4o-mini)
"""

from __future__ import annotations

import json
import os
from typing import Any

from .generator import DatasetGenerationError, _parse_parent, _validate_records

_SYSTEM_PROMPT = """你是一个检索评测数据生成器，专门为 RAG 系统生成 Recall@TopK 评测数据集。

## 任务

根据输入的 parent chunk（含若干 child chunk），生成用于检索评测的 query 数据。

## 输出格式（严格 JSON 数组，不包含任何其他文字）
每条记录结构：
{
  "query": "...",
  "gold_parent_id": "...",
  "gold_child_ids": ["..."],
  "query_type": "single_hop" | "multi_chunk_same_parent",
  "difficulty": "easy" | "medium" | "hard",
  "source_doc": "..."
}

## 生成规则

数量：每个 parent 生成 3~5 条 single_hop + 2~3 条 multi_chunk_same_parent

single_hop：
- gold_child_ids 只包含 1 个 child_id
- easy：query 可直接从文本中找到答案
- medium：query 需要理解语义才能对应到正确 chunk

multi_chunk_same_parent：
- gold_child_ids 包含 2 个或以上 child_id（只能使用同一个 parent 下的）
- difficulty 必须为 hard
- query 必须确实需要多个 chunk 联合才能回答

严格限制（违反则视为失败）：
- query 必须是自然语言用户问题
- query 不能直接复制原文句子
- gold_child_ids 只能使用输入中已有的 child_id，不得编造
- 不输出答案、解释或 markdown 格式
- 只输出纯 JSON 数组"""


def generate_dataset_llm(parent_chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate a retrieval evaluation dataset using an LLM backend.

    Raises DatasetGenerationError if the LLM output fails validation.
    Requires the ``openai`` package: ``pip install openai``.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise DatasetGenerationError(
            "openai package is required for LLM mode: pip install openai"
        ) from exc

    api_key = os.getenv("ARGUS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("ARGUS_LLM_BASE_URL") or None
    model = os.getenv("ARGUS_LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        raise DatasetGenerationError(
            "Set ARGUS_LLM_API_KEY (or OPENAI_API_KEY) to use LLM mode."
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    user_content = json.dumps(parent_chunk, ensure_ascii=False, indent=2)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
    )

    raw = (response.choices[0].message.content or "").strip()

    # Strip markdown code fences if the model wraps output anyway
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines() if not line.startswith("```")
        ).strip()

    try:
        records = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetGenerationError(f"LLM returned invalid JSON: {exc}") from exc

    if not isinstance(records, list):
        raise DatasetGenerationError("LLM output must be a JSON array")

    parent = _parse_parent(parent_chunk)
    _validate_records(records, parent)
    return records
