# Argus

Argus is a deterministic synthetic retrieval dataset generator for parent/child
RAG chunks. It reads one parent chunk JSON object and outputs a strict JSON array
of retrieval evaluation records:

- `query`
- `gold_child_ids`
- `gold_parent_id`
- `query_type`
- `difficulty`
- `source_doc`

It is not a question answering system and does not emit answers.

## Usage

Run with a JSON file:

```powershell
python -m argus .\sample_parent.json --pretty
```

Or read from stdin:

```powershell
Get-Content .\sample_parent.json -Raw | python -m argus --pretty
```

The default output is valid JSON only. Errors are written to stderr and return a
non-zero exit code.

## Input

```json
{
  "parent_id": "parent_000123",
  "source_doc": "robot_manual.pdf",
  "children": [
    {
      "child_id": "child_000123_01",
      "text": "机器人启动前需要检查电源连接是否正常。"
    },
    {
      "child_id": "child_000123_02",
      "text": "如果控制器无法连接，首先检查网络配置。"
    }
  ]
}
```

Argus requires at least two child chunks because the project specification asks
for multi-chunk queries for every parent chunk.

## Development

Run the test suite:

```powershell
python -m unittest discover -s tests
```
