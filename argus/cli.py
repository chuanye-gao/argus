"""Command line interface for Argus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkError, benchmark_pdf, benchmark_text_file
from .generator import DatasetGenerationError, generate_dataset
from .pdf import PdfParsingError


def _read_single(path: str | None) -> Any:
    if path is None or path == "-":
        return json.load(sys.stdin)
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path: str) -> list[Any]:
    records: list[Any] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL line {lineno}: {exc}") from exc
    return records


def _write(data: Any, path: str | None, pretty: bool) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2 if pretty else None)
    if path:
        Path(path).write_text(text, encoding="utf-8")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argus",
        description="Run PDF chunking benchmarks or generate retrieval evaluation queries.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input file. JSON (single) or JSONL (batch). Reads stdin when omitted or '-'.",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Write output to FILE instead of stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: input is JSONL with one parent chunk per line.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM backend (requires ARGUS_LLM_API_KEY or OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the HTTP API server instead of processing a file.",
    )
    parser.add_argument(
        "--benchmark-pdf",
        action="store_true",
        help="Run the PDF chunking benchmark and output a recommendation report.",
    )
    parser.add_argument(
        "--benchmark-text",
        action="store_true",
        help="Run the chunking benchmark on a UTF-8 text file for demos and tests.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=12,
        metavar="N",
        help="Maximum strategy-independent evaluation samples for benchmark mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="K",
        help="Retrieval depth for benchmark mode metrics.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        metavar="HOST",
        help="API server host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        metavar="PORT",
        help="API server port (default: 8000).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- serve mode ---
    if args.serve:
        from .api import serve
        serve(host=args.host, port=args.port)
        return 0

    # --- PDF/text chunking benchmark mode ---
    if args.benchmark_pdf or args.benchmark_text:
        if not args.input:
            print("argus: benchmark mode requires an input file", file=sys.stderr)
            return 1
        try:
            if args.benchmark_pdf:
                report = benchmark_pdf(
                    args.input,
                    max_samples=args.max_samples,
                    top_k=args.top_k,
                )
            else:
                report = benchmark_text_file(
                    args.input,
                    max_samples=args.max_samples,
                    top_k=args.top_k,
                )
            _write(report, args.output, args.pretty)
        except (OSError, PdfParsingError, BenchmarkError, ValueError) as exc:
            print(f"argus: {exc}", file=sys.stderr)
            return 1
        return 0

    # --- choose backend ---
    if args.llm:
        from .llm import generate_dataset_llm as generate_fn
    else:
        generate_fn = generate_dataset

    # --- batch mode ---
    if args.batch:
        try:
            if args.input is None or args.input == "-":
                payloads = [json.loads(l) for l in sys.stdin if l.strip()]
            else:
                payloads = _read_jsonl(args.input)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"argus: {exc}", file=sys.stderr)
            return 1

        all_records: list[Any] = []
        had_error = False
        for index, payload in enumerate(payloads):
            try:
                all_records.extend(generate_fn(payload))
            except DatasetGenerationError as exc:
                print(f"argus: batch[{index}]: {exc}", file=sys.stderr)
                had_error = True

        try:
            _write(all_records, args.output, args.pretty)
        except OSError as exc:
            print(f"argus: {exc}", file=sys.stderr)
            return 1

        return 1 if had_error else 0

    # --- single mode ---
    try:
        payload = _read_single(args.input)
        dataset = generate_fn(payload)
        _write(dataset, args.output, args.pretty)
    except (OSError, json.JSONDecodeError, DatasetGenerationError, ValueError) as exc:
        print(f"argus: {exc}", file=sys.stderr)
        return 1

    return 0
