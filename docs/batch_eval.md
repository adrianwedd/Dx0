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
```

