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

Set `retrieval_backend` in `settings.yaml` (or the `SDB_RETRIEVAL_BACKEND` environment variable) to `my-index` and enable semantic retrieval to use your custom backend.
