"""Command line interface for Argus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .generator import DatasetGenerationError, generate_dataset


def _read_payload(path: str | None) -> Any:
    if path is None or path == "-":
        return json.load(sys.stdin)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argus",
        description="Generate strict JSON retrieval evaluation queries from one parent chunk.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to a parent chunk JSON file. Reads stdin when omitted or set to '-'.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output. The output remains a JSON array.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        payload = _read_payload(args.input)
        dataset = generate_dataset(payload)
    except (OSError, json.JSONDecodeError, DatasetGenerationError) as exc:
        print(f"argus: {exc}", file=sys.stderr)
        return 1

    indent = 2 if args.pretty else None
    print(json.dumps(dataset, ensure_ascii=False, indent=indent))
    return 0
