"""FastAPI web service for Argus.

Start with:
    python -m argus --serve
or:
    uvicorn argus.api:app --host 0.0.0.0 --port 8000

Requires: pip install fastapi uvicorn
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    import uvicorn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "fastapi and uvicorn are required for API mode: pip install fastapi uvicorn"
    ) from exc

from .generator import DatasetGenerationError, generate_dataset

app = FastAPI(
    title="Argus",
    description="Retrieval evaluation tools for parent/child chunks and PDF chunking benchmarks.",
    version="0.2.0",
)


def _get_backend(use_llm: bool):
    if use_llm:
        from .llm import generate_dataset_llm
        return generate_dataset_llm
    return generate_dataset


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/generate")
def generate(payload: dict[str, Any], llm: bool = False) -> list[dict[str, Any]]:
    """Generate a retrieval evaluation dataset for one parent chunk.

    Pass ``?llm=true`` to use the LLM backend (requires ARGUS_LLM_API_KEY).
    """
    try:
        return _get_backend(llm)(payload)
    except DatasetGenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/batch")
def batch(
    payloads: list[dict[str, Any]], llm: bool = False
) -> dict[str, Any]:
    """Generate datasets for multiple parent chunks.

    Returns ``{"results": [...], "errors": [...]}`` where each result item
    is ``{"index": int, "records": [...]}`` and each error item is
    ``{"index": int, "error": str}``.
    """
    fn = _get_backend(llm)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, payload in enumerate(payloads):
        try:
            results.append({"index": index, "records": fn(payload)})
        except DatasetGenerationError as exc:
            errors.append({"index": index, "error": str(exc)})

    return {"results": results, "errors": errors}


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:  # pragma: no cover
    uvicorn.run(app, host=host, port=port)
