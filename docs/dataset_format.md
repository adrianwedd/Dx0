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

## FHIR Bundles

DiagnosticReport bundles from EHR systems can be imported with
``scripts/fhir_to_casedb.py``. The script accepts one or more FHIR JSON files
or directories and writes a case database file:

```bash
python scripts/fhir_to_casedb.py fhir/ --output cases.json
```

Pass ``--sqlite`` or use a ``.db`` extension to create a SQLite database ready
for ``CaseDatabase.load_from_sqlite``.

## SQLite Storage

Cases can also be stored in a SQLite database for lazy loading. Use the
``scripts/migrate_to_sqlite.py`` script to convert an existing JSON or CSV
dataset:

```bash
python scripts/migrate_to_sqlite.py cases.json cases.db
```

Then load it with ``--db-sqlite`` when running ``cli.py``.
