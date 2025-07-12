# Evaluation Metrics Database

Session results from batch evaluations can be stored in SQLite for
later analysis. Initialize an empty database:

```bash
python scripts/init_metrics_db.py results.db
```

Pass `--results-db results.db` when running `batch-eval` or the
single-session CLI. Each run inserts a row with the case id, total
cost, judge score, correctness flag and duration.

Query average accuracy and cost across all runs:

```sql
SELECT AVG(score >= 4) AS accuracy, AVG(total_cost) AS avg_cost
FROM results;
```

List the most expensive cases:

```sql
SELECT case_id, total_cost
FROM results
ORDER BY total_cost DESC
LIMIT 5;
```
