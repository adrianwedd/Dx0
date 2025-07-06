# Dataset Format

Each clinical case in SDBench is stored as a JSON file containing three keys:

- `id`: unique case identifier, e.g. `"case_001"`.
- `summary`: a short text summary of the case.
- `steps`: list of ordered dialogue steps. Every step object has an integer `id`
  and a `text` field with the paragraph content.

Example:

```json
{
  "id": "case_001",
  "summary": "Example summary",
  "steps": [
    {"id": 1, "text": "First paragraph"},
    {"id": 2, "text": "Next paragraph"}
  ]
}
```

The ingestion pipeline in `sdb.ingest.convert` can convert raw text files into
this JSON structure.
