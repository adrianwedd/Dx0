# Dx0 & SDBench: Diagnostic Orchestrator and Sequential Diagnosis Benchmark

This repository contains two integrated projects for advancing sequential clinical diagnosis with AI:

1. **Dx0**: A model-agnostic, multi-agent diagnostic orchestrator that simulates a panel of five physician personas—Hypothesis, Test-Chooser, Challenger, Stewardship, and Checklist—to generate diagnoses through iterative debate. Dx0 orders high-value questions or tests, tracks cumulative costs via CPT/CMS mapping, and balances cost-efficiency against diagnostic accuracy using configurable modes (Instant Answer, Question-Only, Budgeted, Unconstrained, Ensemble).

2. **SDBench**: A benchmark suite that ingests 304 NEJM Clinical Pathological Conference (CPC) cases, exposing them as interactive, stepwise diagnostic tasks. SDBench measures agent performance on test-selection strategies, cost-accuracy trade-offs, and clinical reasoning quality, offering an evaluation pipeline with statistical significance testing and Pareto frontier analysis.

---

## Dx0: Diagnostic Decision Orchestrator

### Core Functionality

* **Orchestrator Engine**: Coordinates debate rounds among persona agents, enforcing turn limits and consensus logic.
* **Virtual Panel**: Five specialized agents:

  * *Hypothesis*: Generates initial differential diagnoses.
  * *Test-Chooser*: Selects questions or diagnostic tests to refine hypotheses.
  * *Challenger*: Critiques assumptions and explores alternatives.
  * *Stewardship*: Monitors and enforces cost budgets.
  * *Checklist*: Ensures reasoning completeness before final verdict.
* **Gatekeeper Interface**: Information oracle that reveals only requested findings, synthesizes plausible results for unqueried tests.
* **Cost Estimator**: Maps tests to CPT codes and CMS prices.
* **BudgetManager**: Tracks cumulative spending with the estimator and halts a session when a limit is reached.
* **Evaluation Modes**: Modes for exploring different cost-accuracy scenarios: Instant Answer, Question-Only, Budgeted, Unconstrained Budget, and Ensemble.

### BudgetManager

The ``BudgetManager`` service keeps a running total of ordered test costs and
optionally stops a session once spending exceeds a configurable limit. It replaces the earlier budget tracker and relies on a ``CostEstimator`` to
translate test names into prices. The manager can be passed to
:class:`Orchestrator` or created automatically when the CLI runs in
``budgeted`` mode.

```python
from sdb import BudgetManager, CostEstimator, Orchestrator

costs = CostEstimator.load_from_csv("data/sdbench/costs.csv")
bm = BudgetManager(costs, budget=500)
orc = Orchestrator(panel, gatekeeper, budget_manager=bm)
```

Use ``--budget`` to set the cap when running the CLI in ``budgeted`` mode:

```bash
python -m dx0.cli --mode budgeted --budget 1000 --case-file case.json
```

The Physician UI reads ``UI_BUDGET_LIMIT`` for the default limit and also
accepts a ``budget`` query parameter on the WebSocket connection. See
[`docs/budget_manager.md`](docs/budget_manager.md) for details.

---

## SDBench: Sequential Diagnosis Benchmark

### Core Functionality

* **Case Ingestion**: Parses 304 NEJM CPC cases into JSON vignettes, findings, and ground-truth diagnoses.
* **Interactive Sessions**: Exposes each case as a turn-based Q&A and testing task, mediated by Dx0 or human clinicians via a synchronous chat UI.
* **Metrics & Reporting**: Computes diagnostic accuracy (5-point Likert rubric), cumulative cost, and efficiency metrics; supports permutation testing for significance.
* **Visualization**: Generates Pareto curves, cost-accuracy plots, and summary reports for ensemble and variant evaluations.

---

## Installation

Full setup instructions are provided in [docs/installation.md](docs/installation.md).
The short version is:

```bash
git clone https://github.com/adrianwedd/Dx0.git
cd clinical-ai-suite
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.lock
```

For details on creating a development environment and running the test suite,
see [docs/getting_started.md](docs/getting_started.md).

See [Dependency Management](docs/dependency_updates.md) for details on
updating and auditing requirements. You can build a Docker image using the
multi-stage `Dockerfile`:

```bash
docker build -t clinical-ai-suite .
```

The new build installs development tools only in the first stage and keeps the
runtime image lean (about 800 MB instead of roughly 1.5 GB in the previous
single-stage build).

---

## Quickstart

### Dx0 CLI

```bash
python -m dx0.cli \
  --mode budgeted \
  --case-file data/sdbench/cases/case_001.json \
  --budget 1000 \
  --output results/dx0_case_001.json
```
Budget enforcement is handled by the ``BudgetManager`` service created by the CLI.
Set the limit with ``--budget``.
Add `--cache` to reuse previous LLM responses and reduce API calls:

```bash
python -m dx0.cli --cache ...
```

To retrieve documents semantically and rerank them with a cross encoder, use
`--semantic-retrieval` together with `--cross-encoder-model`:

```bash
python -m dx0.cli \
  --semantic-retrieval \
  --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 ...
```
The cross encoder re-scores the top semantic matches so the most relevant
passages appear first, improving precision.

For faster large-scale retrieval install the optional `faiss-cpu` package and
run `scripts/migrate_to_faiss.py` to convert existing embeddings into a FAISS
index. When `faiss` is available, semantic retrieval will automatically use the
`FaissIndex` backend.


To run models via a local Ollama server, specify `--llm-provider ollama` and
optionally set `--ollama-base-url` if the server is not on the default port:

```bash
python -m dx0.cli \
  --llm-provider ollama \
  --ollama-base-url http://localhost:11434 ...
```

For a locally fine-tuned Hugging Face model, use `--llm-provider hf-local` and
point `--hf-model` to the model directory:

```bash
python -m dx0.cli \
  --llm-provider hf-local \
  --hf-model /models/nejm-lora ...
```

### Configuration

You can load common settings from a YAML file using `--config`:

```yaml
openai_api_key: sk-your-key
openai_model: gpt-4
hf_model: /models/nejm-lora
ollama_base_url: http://localhost:11434
metrics_port: 8000
case_db: data/sdbench/cases
case_db_sqlite: cases.db
```

If ``case_db_sqlite`` is provided, cases are loaded lazily from the SQLite file.

Pass the file to any CLI command:

```bash
python cli.py --config settings.yml --db cases.json --case 1 --rubric r.json --costs c.csv
```

### SDBench CLI

```bash
python -m sdbench.cli \
  --mode ensemble \
  --cases data/sdbench/cases/ \
  --output results/sdbench_summary.csv
```

### Data Ingestion

The repository includes the 304 NEJM CPC cases in `data/sdbench/cases/`.
These files are for research use only and may not be redistributed without
permission from NEJM.

#### Dataset Usage Restrictions

The case text originates from the *Case Records of the Massachusetts General Hospital* series and
is provided solely for non&#8209;commercial research. Do not share or incorporate the
full case files into any commercial product without first obtaining written permission
from NEJM. When in doubt, consult your institution's legal counsel.

To refresh the dataset or fetch updates, run the
ingestion pipeline:

```bash
python -m sdb.ingest.pipeline
```

To append newly released cases without re-downloading the entire corpus, run:

```bash
python scripts/update_cases.py
```

To convert a FHIR ``Bundle`` or ``DiagnosticReport`` into SDBench case JSON,
use the `fhir-import` command:

```bash
python cli.py fhir-import report.json --case-id new_case > case.json
```


### Physician UI

Install the development requirements first (provides `uvicorn` and other packages):

```bash
pip install -r requirements.lock
```

Then start the demo web interface to chat with the Gatekeeper and view running cost estimates:

```bash
uvicorn sdb.ui.app:app --reload
```

Then open `http://localhost:8000` in your browser. Log in with the default
credentials `physician` / `secret`.

The UI makes a request to the `/case` endpoint immediately after login to
retrieve the vignette. Three information panels appear alongside the chat box:

* **Case Summary** – displays the text returned by `/case`.
* **Ordered Tests** – lists labs and imaging that have been completed.
* **Diagnostic Flow** – records each step of the debate among persona agents.

![Screenshot of UI](docs/images/ui.png.b64)
This screenshot is base64-encoded. To view it, save the contents of
`docs/images/ui.png.b64` to a file and decode it with:

```bash
base64 -d docs/images/ui.png.b64 > ui.png
```



### Python API Example

```python
from sdb import (
    CaseDatabase,
    Gatekeeper,
    VirtualPanel,
    Orchestrator,
    CostEstimator,
    BudgetManager,
)

# Dx0 run
db = CaseDatabase.load_from_json("cases.json")
gatekeeper = Gatekeeper(db, "case_001")
panel = VirtualPanel()
costs = CostEstimator.load_from_csv("data/sdbench/costs.csv")
orc = Orchestrator(
    panel,
    gatekeeper,
    budget_manager=BudgetManager(costs, budget=1000),
)
while not orc.finished:
    orc.run_turn("")
print(orc.final_diagnosis, orc.spent)
```

### Advanced Orchestrator Settings

Dx0 exposes several configuration hooks for fine-tuning panel behavior.

**Budgets**

- Pass a `BudgetManager` to :class:`Orchestrator` to enforce a spending limit.
- Use the `--budget` flag when running in budgeted mode or `--budget-limit` for
  other modes. The Physician UI reads `UI_BUDGET_LIMIT` for its default cap.

```python
costs = CostEstimator.load_from_csv("data/sdbench/costs.csv")
bm = BudgetManager(costs, budget=250)
orc = Orchestrator(VirtualPanel(), gatekeeper, budget_manager=bm)
```

**Personas**

- Custom persona chains can be loaded from plugins registered under the
  ``dx0.personas`` entry point group. Instantiate a panel with
  ``VirtualPanel(persona_chain="optimist")`` or your own plugin name.
- See [docs/persona_plugins.md](docs/persona_plugins.md) for plugin creation
  details.

```python
panel = VirtualPanel(persona_chain="my-chain")
orc = Orchestrator(panel, gatekeeper)
```


### Running Tests

The test suite depends on additional packages such as `httpx`,
`starlette`, and `pydantic`. Install the development requirements first:

```bash
pip install -r requirements-dev.txt
pytest -q
```

### Statistical Significance Testing

After running evaluations you can test whether a variant's scores differ
significantly from a baseline using a permutation test:

```bash
python cli.py stats results/baseline.csv results/variant.csv --column accuracy
```

The command prints a p-value. Values below 0.05 indicate the observed
difference in the selected column is unlikely under the null hypothesis of no
effect.

### Asynchronous Batch Evaluation

Use the :func:`batch_evaluate` helper to run case evaluations concurrently:

```python
from sdb.evaluation import batch_evaluate

def run_case(cid: str) -> dict[str, str]:
    ...

case_ids = ["1", "2"]
results = batch_evaluate(case_ids, run_case, concurrency=4)
```

Equivalent functionality is available via the CLI subcommand:

```bash
python cli.py batch-eval --db cases.json --rubric rubric.json \
    --costs costs.csv --output results.csv --concurrency 4
```
If using --llm-provider ollama, set --ollama-base-url to your server URL.

### Notebook Walkthrough

For a short, self-contained demo see
[evaluation_example.ipynb](notebooks/evaluation_example.ipynb). The notebook
illustrates running `batch_evaluate`, using a `WeightedVoter`, and plotting
accuracy versus cost.

### FHIR Session Export

To convert a session transcript and ordered tests into a FHIR Bundle use the
`fhir-export` subcommand:

```bash
python cli.py fhir-export transcript.json tests.json --patient-id 123 > bundle.json
```

The command reads the JSON transcript (list of `[speaker, text]` pairs) and test
list, then writes a combined FHIR ``Bundle`` to ``stdout`` or the optional
``--output`` path.

### Case Annotation

Use the ``annotate-case`` command to add notes or correct test aliases for a
single case. By default the annotated JSON is written to ``annotations/<id>.json``.

```bash
python cli.py annotate-case --db cases.json --case 1 \
    --notes "Review" --test-mapping mapping.json
```

---

## Repository Structure

```
clinical-ai-suite/
├── dx0/                   # Dx0 orchestrator, agents, and interfaces
├── sdbench/               # SDBench ingestion, evaluation pipeline, and UI
├── data/                  # NEJM CPC JSON cases and CMS pricing data
├── prompts/               # Persona and system prompt templates
├── results/               # Output logs, metrics, and plots
├── tests/                 # Unit and integration tests for both modules
├── Dockerfile             # Containerization setup
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── CONTRIBUTING.md        # Contribution guidelines
├── CODE_OF_CONDUCT.md     # Community standards
├── LICENSE                # MIT License file
└── setup.py               # Package metadata
```

---

## Documentation & Contribution

* **README**: Overview and quickstart.
* **CONTRIBUTING.md**: Issue reporting, pull request process, coding standards.
* **PULL_REQUEST_TEMPLATE.md**, **ISSUE_TEMPLATE/**: Guidance for contributors.
* **CODE_OF_CONDUCT.md**: Community norms and reporting procedures.
* **docs/persona_plugins.md**: How to create and install persona plugins.
* **docs/add_users.md**: Adding login accounts for the Physician UI.
* **docs/cost_estimator_plugins.md**: Implement custom pricing logic via entry points.

---

## License

The code in this repository is released under the MIT License. See
[LICENSE](LICENSE) for details. The NEJM CPC case data in `data/sdbench`
are provided solely for non-commercial research. NEJM retains all rights
to the original articles, and redistribution may require permission from
NEJM. We are unable to confirm further distribution rights; consult your
own legal counsel before sharing the dataset.

For a list of changes in each version, see
[the release notes](docs/release_notes.md).

---

## References

* Nori *et al.* (2025). "Sequential Diagnosis with Language Models." *arXiv preprint* arXiv:2506.22405v2. Available at [https://arxiv.org/abs/2506.22405](https://arxiv.org/abs/2506.22405)
* Microsoft AI Blog (2024). "The Path to Medical Superintelligence." Retrieved from [https://microsoft.ai/new/the-path-to-medical-superintelligence/](https://microsoft.ai/new/the-path-to-medical-superintelligence/)
