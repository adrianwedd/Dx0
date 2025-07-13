# Retrieval Plugins

SDB uses a retrieval index to surface relevant text passages from the case database. Third-party backends can extend this system via Python entry points.

## Writing a Plugin

1. Implement a class providing a `query(text, top_k=1)` method:

```python
from typing import List, Tuple
from sdb.retrieval import BaseRetrievalIndex

class MyIndex(BaseRetrievalIndex):
    def __init__(self, documents: List[str]):
        self.documents = documents

    def query(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        ...  # return a ranked list of passages
```

2. Declare the entry point in your `pyproject.toml`:

```toml
[project.entry-points."sdb.retrieval_plugins"]
my-index = "my_plugin:MyIndex"
```

3. Install the package so `importlib.metadata` can discover the entry point:

```bash
pip install -e path/to/my_plugin
```

## Packaging & Versioning

Create a small Python package with a `pyproject.toml` describing your plugin.
Give the package a unique name (for example `sdb-my-index`) and list `sdb` as a
dependency:

```toml
[project]
name = "sdb-my-index"
version = "0.1.0"
dependencies = ["sdb"]

[project.entry-points."sdb.retrieval_plugins"]
my-index = "my_plugin:MyIndex"
```

Use [semantic versioning](https://semver.org/) when updating your plugin. Install
the package in editable mode while developing:

```bash
pip install -e .
```

When publishing a new release, build a wheel with `python -m build` and upload it
to PyPI using `twine upload dist/*`. Tag releases in Git so version history is
clear.

## Example Repository Layout

```
my_index/
├── pyproject.toml
└── my_plugin.py
```

An example implementation is provided in `examples/retrieval_plugin/`. After
installation, set `retrieval_backend` in `settings.yaml` (or the
`SDB_RETRIEVAL_BACKEND` environment variable) to `my-index` and enable semantic
retrieval to use your custom backend.
