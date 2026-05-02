"""Core generation logic for retrieval evaluation datasets.

The implementation is deterministic and dependency-free. It does not try to
answer questions; it only creates query records tied to the provided child IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable


SINGLE_HOP = "single_hop"
MULTI_CHUNK = "multi_chunk_same_parent"

_DOMAIN_TERMS = [
    "机器人",
    "设备",
    "系统",
    "控制器",
    "传感器",
    "电机",
    "电源",
    "网络",
    "IP",
    "接口",
    "端口",
    "配置",
    "参数",
    "启动",
    "运行",
    "停止",
    "重启",
    "连接",
    "通信",
    "校准",
    "安装",
    "升级",
    "日志",
    "告警",
    "异常",
    "故障",
    "权限",
    "数据库",
    "缓存",
    "文件",
    "任务",
    "流程",
    "温度",
    "电压",
    "压力",
    "安全",
]

_FAULT_HINTS = ("异常", "故障", "无法", "失败", "错误", "报错", "中断", "超时", "告警", "不正常")
_CONFIG_HINTS = ("配置", "设置", "参数", "IP", "网络", "端口", "权限", "账号", "密码")
_OPERATION_HINTS = ("启动", "运行", "停止", "重启", "安装", "升级", "执行", "操作", "流程", "步骤")
_CHECK_HINTS = ("检查", "确认", "注意", "禁止", "避免", "确保", "要求", "条件")


class DatasetGenerationError(ValueError):
    """Raised when a parent chunk cannot produce a valid evaluation dataset."""


@dataclass(frozen=True)
class ChildChunk:
    child_id: str
    text: str
    subject: str
    intent: str


@dataclass(frozen=True)
class ParentChunk:
    parent_id: str
    source_doc: str
    children: tuple[ChildChunk, ...]


def generate_dataset(parent_chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate strict retrieval evaluation records for one parent chunk."""

    parent = _parse_parent(parent_chunk)
    records: list[dict[str, Any]] = []

    for child, variant in _single_hop_plan(parent.children):
        records.append(
            _record(
                query=_single_query(child, variant),
                parent=parent,
                child_ids=[child.child_id],
                query_type=SINGLE_HOP,
                difficulty="easy" if variant == 0 else "medium",
            )
        )

    for children in _multi_chunk_plan(parent.children):
        records.append(
            _record(
                query=_multi_query(children),
                parent=parent,
                child_ids=[child.child_id for child in children],
                query_type=MULTI_CHUNK,
                difficulty="hard",
            )
        )

    _validate_records(records, parent)
    return records


def _parse_parent(payload: dict[str, Any]) -> ParentChunk:
    if not isinstance(payload, dict):
        raise DatasetGenerationError("input must be a JSON object")

    parent_id = _required_string(payload, "parent_id")
    source_doc = _required_string(payload, "source_doc")
    raw_children = payload.get("children")

    if not isinstance(raw_children, list):
        raise DatasetGenerationError("children must be a list")
    if len(raw_children) < 2:
        raise DatasetGenerationError("at least two child chunks are required for multi-chunk queries")

    seen: set[str] = set()
    children: list[ChildChunk] = []
    for index, raw_child in enumerate(raw_children):
        if not isinstance(raw_child, dict):
            raise DatasetGenerationError(f"children[{index}] must be an object")

        child_id = _required_string(raw_child, "child_id", f"children[{index}]")
        text = _required_string(raw_child, "text", f"children[{index}]")
        if child_id in seen:
            raise DatasetGenerationError(f"duplicate child_id: {child_id}")
        seen.add(child_id)

        children.append(
            ChildChunk(
                child_id=child_id,
                text=text,
                subject=_subject_from_text(text),
                intent=_intent_from_text(text),
            )
        )

    return ParentChunk(parent_id=parent_id, source_doc=source_doc, children=tuple(children))


def _required_string(payload: dict[str, Any], key: str, prefix: str | None = None) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        field = f"{prefix}.{key}" if prefix else key
        raise DatasetGenerationError(f"{field} must be a non-empty string")
    return value.strip()


def _single_hop_plan(children: tuple[ChildChunk, ...]) -> list[tuple[ChildChunk, int]]:
    target = min(5, max(3, len(children)))
    plan: list[tuple[ChildChunk, int]] = []
    variant_by_child = {child.child_id: 0 for child in children}

    cursor = 0
    while len(plan) < target:
        child = children[cursor % len(children)]
        variant = variant_by_child[child.child_id]
        plan.append((child, variant))
        variant_by_child[child.child_id] = variant + 1
        cursor += 1

    return plan


def _multi_chunk_plan(children: tuple[ChildChunk, ...]) -> list[tuple[ChildChunk, ...]]:
    target = 2 if len(children) == 2 else 3
    pairs: list[tuple[ChildChunk, ...]] = []

    for index in range(len(children) - 1):
        pairs.append((children[index], children[index + 1]))
        if len(pairs) == target:
            return pairs

    pairs.append((children[-1], children[0]))
    return pairs[:target]


def _single_query(child: ChildChunk, variant: int) -> str:
    subject = child.subject

    if child.intent == "fault":
        templates = [
            "{subject}出现问题时应该如何排查？",
            "用户遇到{subject}异常时需要先确认哪些情况？",
            "{subject}无法正常工作可能要查看哪些说明？",
        ]
    elif child.intent == "config":
        templates = [
            "{subject}配置需要关注哪些内容？",
            "用户调整{subject}时应该核对什么？",
            "{subject}设置不符合预期时应从哪里排查？",
        ]
    elif child.intent == "operation":
        templates = [
            "{subject}前需要做哪些准备？",
            "用户执行{subject}相关操作时要注意什么？",
            "如何确认{subject}过程符合要求？",
        ]
    elif child.intent == "check":
        templates = [
            "{subject}需要重点检查什么？",
            "用户确认{subject}状态时应关注哪些要求？",
            "{subject}相关风险应该怎样提前核对？",
        ]
    else:
        templates = [
            "{subject}相关要求是什么？",
            "用户需要了解{subject}时应该查询哪些信息？",
            "{subject}场景下有哪些关键说明？",
        ]

    return templates[variant % len(templates)].format(subject=subject)


def _multi_query(children: Iterable[ChildChunk]) -> str:
    subjects = _unique([child.subject for child in children])
    if len(subjects) == 1:
        joined = subjects[0]
    else:
        joined = "和".join(subjects)
    return f"同时处理{joined}相关问题时，需要综合关注哪些说明？"


def _record(
    *,
    query: str,
    parent: ParentChunk,
    child_ids: list[str],
    query_type: str,
    difficulty: str,
) -> dict[str, Any]:
    return {
        "query": query,
        "gold_parent_id": parent.parent_id,
        "gold_child_ids": child_ids,
        "query_type": query_type,
        "difficulty": difficulty,
        "source_doc": parent.source_doc,
    }


def _subject_from_text(text: str) -> str:
    terms = _terms_from_text(text)
    if not terms:
        return "该内容"

    if len(terms) == 1:
        return terms[0]

    subject = "".join(terms[:2])
    if len(subject) > 12:
        return terms[0]
    return subject


def _terms_from_text(text: str) -> list[str]:
    positioned_terms: list[tuple[int, str]] = []
    for term in _DOMAIN_TERMS:
        position = text.find(term)
        if position >= 0:
            positioned_terms.append((position, term))

    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,20}", text):
        positioned_terms.append((text.find(token), token))

    positioned_terms.sort(key=lambda item: (item[0], len(item[1])))
    return _unique(term for _, term in positioned_terms)


def _intent_from_text(text: str) -> str:
    if _contains_any(text, _FAULT_HINTS):
        return "fault"
    if _contains_any(text, _CONFIG_HINTS):
        return "config"
    if _contains_any(text, _OPERATION_HINTS):
        return "operation"
    if _contains_any(text, _CHECK_HINTS):
        return "check"
    return "generic"


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _validate_records(records: list[dict[str, Any]], parent: ParentChunk) -> None:
    child_ids = {child.child_id for child in parent.children}
    query_values: set[str] = set()

    single_count = 0
    multi_count = 0
    for record in records:
        query = record.get("query")
        gold_child_ids = record.get("gold_child_ids")
        query_type = record.get("query_type")
        difficulty = record.get("difficulty")

        if not isinstance(query, str) or not query.strip():
            raise DatasetGenerationError("generated query must be a non-empty string")
        if query in query_values:
            raise DatasetGenerationError(f"duplicate generated query: {query}")
        query_values.add(query)

        if parent.parent_id in query:
            raise DatasetGenerationError("query must not contain parent_id")
        if any(child_id in query for child_id in child_ids):
            raise DatasetGenerationError("query must not contain child_id")
        if not isinstance(gold_child_ids, list) or not gold_child_ids:
            raise DatasetGenerationError("gold_child_ids must be a non-empty list")
        if any(child_id not in child_ids for child_id in gold_child_ids):
            raise DatasetGenerationError("gold_child_ids contains an unknown child_id")

        if query_type == SINGLE_HOP:
            single_count += 1
            if len(gold_child_ids) != 1:
                raise DatasetGenerationError("single_hop records must reference exactly one child")
        elif query_type == MULTI_CHUNK:
            multi_count += 1
            if len(gold_child_ids) < 2:
                raise DatasetGenerationError("multi_chunk_same_parent records must reference multiple children")
        else:
            raise DatasetGenerationError(f"unsupported query_type: {query_type}")

        if difficulty not in {"easy", "medium", "hard"}:
            raise DatasetGenerationError(f"unsupported difficulty: {difficulty}")

    if not 3 <= single_count <= 5:
        raise DatasetGenerationError("single_hop query count must be between 3 and 5")
    if not 2 <= multi_count <= 3:
        raise DatasetGenerationError("multi_chunk_same_parent query count must be between 2 and 3")
