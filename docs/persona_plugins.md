# Persona Plugins

Dx0 can load alternative persona chains through Python entry points. Each plugin
registers a callable under the `dx0.personas` group that returns a list of
prompt names. The prompts are loaded from the `prompts/` directory just like the
built-in personas.

## Creating a Plugin

1. Create a small Python package with a function returning the chain names:

```python
# my_plugin.py
from typing import List

def my_chain() -> List[str]:
    return ["hypothesis_system", "optimist_system", "checklist_system"]
```

2. In the package's `pyproject.toml`, declare the entry point:

```toml
[project.entry-points."dx0.personas"]
my-chain = "my_plugin:my_chain"
```

3. Install the package so Dx0 can discover it:

```bash
pip install -e path/to/my_plugin
```

After installation, you can instantiate a panel with
`VirtualPanel(persona_chain="my-chain")`.
