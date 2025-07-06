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
* **Cost Estimator**: Maps tests to CPT codes and CMS prices, maintaining cumulative cost tracking.
* **Evaluation Modes**: Modes for exploring different cost-accuracy scenarios: Instant Answer, Question-Only, Budgeted, Unconstrained Budget, and Ensemble.

---

## SDBench: Sequential Diagnosis Benchmark

### Core Functionality

* **Case Ingestion**: Parses 304 NEJM CPC cases into JSON vignettes, findings, and ground-truth diagnoses.
* **Interactive Sessions**: Exposes each case as a turn-based Q&A and testing task, mediated by Dx0 or human clinicians via a synchronous chat UI.
* **Metrics & Reporting**: Computes diagnostic accuracy (5-point Likert rubric), cumulative cost, and efficiency metrics; supports permutation testing for significance.
* **Visualization**: Generates Pareto curves, cost-accuracy plots, and summary reports for ensemble and variant evaluations.

---

## Installation

```bash
git clone https://github.com/adrianwedd/Dx0.git
cd clinical-ai-suite
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*(Optional)* Docker build:

```bash
docker build -t clinical-ai-suite .
```

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

### SDBench CLI

```bash
python -m sdbench.cli \
  --mode ensemble \
  --cases data/sdbench/cases/ \
  --output results/sdbench_summary.csv
```

### Data Ingestion

The repository includes the 304 NEJM CPC cases in `data/sdbench/cases/`. To
refresh the dataset or fetch updates, run the ingestion pipeline:

```bash
python -m sdb.ingest.pipeline
```

To append newly released cases without re-downloading the entire corpus, run:

```bash
python scripts/update_cases.py
```


### Physician UI

Start the demo web interface to chat with the Gatekeeper and view
running cost estimates:

```bash
uvicorn sdb.ui.app:app --reload
```

Then open `http://localhost:8000` in your browser.


### Python API Example

```python
from dx0 import DxOrchestrator
from sdbench import Benchmark, Settings

# Dx0 run
dx_settings = Settings(mode="unconstrained", model="gpt-4")
orc = DxOrchestrator(dx_settings)
res = orc.run(case_path="data/sdbench/cases/case_001.json")
print(res.diagnosis, res.total_cost)

# SDBench evaluation
bench = Benchmark(data_dir="data/sdbench/cases/", settings=dx_settings)
metrics = bench.run()
print(metrics)
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

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

* Nori *et al.* (2025). "Sequential Diagnosis with Language Models." *arXiv preprint* arXiv:2506.22405v2. Available at [https://arxiv.org/abs/2506.22405](https://arxiv.org/abs/2506.22405)
* Microsoft AI Blog (2024). "The Path to Medical Superintelligence." Retrieved from [https://microsoft.ai/new/the-path-to-medical-superintelligence/](https://microsoft.ai/new/the-path-to-medical-superintelligence/)
