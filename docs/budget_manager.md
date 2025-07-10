# BudgetManager

The `BudgetManager` service records the price of each ordered test using a
`CostEstimator` and stops the session once the configured limit is reached. It
also exposes the remaining budget so UIs can display a live cost summary.

```python
from sdb import BudgetManager, CostEstimator, Orchestrator

costs = CostEstimator.load_from_csv("data/sdbench/costs.csv")
bm = BudgetManager(costs, budget=500)
orc = Orchestrator(panel, gatekeeper, budget_manager=bm)
```

The CLI creates a `BudgetManager` automatically when `--mode budgeted` is
selected. Set the spending cap with `--budget`:

```bash
python -m dx0.cli --mode budgeted --budget 1000 --case-file case.json
```

The Physician UI reads `UI_BUDGET_LIMIT` to control the default cap for new
sessions. You can override this for a single connection by passing a `budget`
query parameter when opening the WebSocket:

```bash
export UI_BUDGET_LIMIT=750
uvicorn sdb.ui.app:app --reload
# Connect with ws://localhost:8000/api/v1/ws?token=<TOKEN>&budget=500
```
