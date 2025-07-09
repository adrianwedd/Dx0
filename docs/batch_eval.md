# Asynchronous Batch Evaluation

Large scale experiments can be run concurrently to shorten evaluation time. The
`batch_evaluate` helper executes a user provided case function across multiple
IDs using `asyncio`.

```python
from sdb.evaluation import batch_evaluate

# function that runs one diagnostic session and returns a result dict

def run_case(case_id: str) -> dict[str, str]:
    ...
    return {"id": case_id, "score": "5"}

case_ids = ["1", "2", "3"]
results = batch_evaluate(case_ids, run_case, concurrency=4)
```

You can achieve the same via the CLI:

```bash
python cli.py batch-eval --db cases.json --rubric rubric.json \
    --costs costs.csv --output results.csv --concurrency 4

# or with a SQLite database
python cli.py batch-eval --db-sqlite cases.db --rubric rubric.json \
    --costs costs.csv --output results.csv --concurrency 4
```

## Filtering Cases

Use the `filter-cases` command to create a smaller dataset before running
evaluations. Cases can be selected by keywords found in the summary or full
text, or by matching metadata fields.

```bash
python cli.py filter-cases --db cases.json --keywords fever,cough \
    --output subset.json

python cli.py filter-cases --db cases.json --metadata '{"tag": "respiratory"}' \
    --output subset.csv
```

