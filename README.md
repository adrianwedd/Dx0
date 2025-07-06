# Dx0

This repository contains a skeleton implementation of the **SDBench** framework and the
**MAI-DxO** diagnostic agent. The code outlines core components such as the case
database, cost estimator, gatekeeper and judge agents, as well as the virtual
panel used for the "Chain of Debate" workflow.

The project roadmap is tracked in `tasks.yml`.

## Running the demo

The `cli.py` script runs a short interactive session. Provide a JSON file or a
directory containing case data and specify the desired case identifier. In
addition to the required `--rubric` and `--costs` paths, the script accepts a
number of flags that control how the panel behaves:

- `--panel-engine` – choose the decision engine (`rule` or `llm`)
- `--llm-model` – model for the LLM engine (`gpt-4` or `turbo`)
- `--verbose` / `--quiet` – adjust logging level
- `--budget` – spending limit when running in budgeted mode
- `--mode` – execution mode (`unconstrained`, `budgeted`, `question-only`,
  `instant`, `ensemble`)

Basic usage:

```bash
python cli.py --db cases.json --case 1 --rubric rubric.json --costs costs.csv
```

Run the panel with the LLM engine and debug logging enabled:

```bash
python cli.py --db cases.json --case 1 \
  --rubric rubric.json --costs costs.csv \
  --panel-engine llm --llm-model turbo --verbose
```

Budgeted mode with a $100 limit and minimal output:

```bash
python cli.py --db cases.json --case 1 \
  --rubric rubric.json --costs costs.csv \
  --mode budgeted --budget 100 --quiet
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

## Accuracy vs cost plot

The repository includes a small CSV example `example_results.csv` with
`cost` and `accuracy` columns. Use the helper in `sdb.plotting` to produce
a scatter plot:

```bash
python -c "from sdb.plotting import load_results, plot_accuracy_vs_cost;\
plot_accuracy_vs_cost(load_results('example_results.csv'))"
```

## Development

Install development dependencies and run the test suite:

```bash
pip install -r requirements-dev.txt
pytest -q
```
