# Cost Estimator Plugins

Dx0 can load alternative pricing logic through Python entry points. Each plugin
registers a callable under the `dx0.cost_estimators` group that accepts the path
to a pricing table and returns a `CostEstimator` instance.

## Writing a Plugin

1. Implement a loader function:

```python
from sdb.cost_estimator import CostEstimator, CptCost

def my_estimator(path: str) -> CostEstimator:
    # load prices from `path` or ignore it and call an API
    table = {"cbc": CptCost("100", 1.0)}
    return CostEstimator(table)
```

2. Declare the entry point in your `pyproject.toml`:

```toml
[project.entry-points."dx0.cost_estimators"]
my-estimator = "my_package:my_estimator"
```

3. Install the package so `importlib.metadata` can discover the entry point:

```bash
pip install -e path/to/my_package
```

Select the plugin with `--cost-estimator` or set `SDB_COST_ESTIMATOR` in the
environment. The default `csv` plugin uses `CostEstimator.load_from_csv`.
