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
