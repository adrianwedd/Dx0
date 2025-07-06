# Dx0

This repository contains a skeleton implementation of the **SDBench** framework and the
**MAI-DxO** diagnostic agent. The code outlines core components such as the case
database, cost estimator, gatekeeper and judge agents, as well as the virtual
panel used for the "Chain of Debate" workflow.

The project roadmap is tracked in `tasks.yml`.

## Running the demo

The `cli.py` script runs a short interactive session. Provide a JSON file or a
directory containing case data and specify the desired case identifier:

```bash
python cli.py --db cases.json --case 1
```

## Case database

`CaseDatabase` can load cases from JSON, directories of text files or from
CSV files. Loading from CSV is convenient when case data is stored in a single
table:

```python
from sdb.case_database import CaseDatabase
db = CaseDatabase.load_from_csv("cases.csv")
```

The CSV must contain `id`, `summary` and `full_text` columns.
