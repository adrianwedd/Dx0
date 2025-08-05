# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Setup development environment
pip install -r requirements.lock    # Install all dependencies
make install-dev                    # Install development tools
pre-commit install                  # Setup git hooks for code quality

# Alternative setup with Docker
docker build -t clinical-ai-suite .
```

### Testing and Quality
```bash
pytest -q                          # Run test suite
pre-commit run --all-files         # Run code formatting and linting
flake8                             # Manual linting check
black .                            # Format code
isort .                            # Sort imports
mypy --ignore-missing-imports      # Type checking
```

### CLI Operations
```bash
# Dx0 diagnostic orchestrator
python -m dx0.cli --mode budgeted --case-file data/sdbench/cases/case_001.json --budget 1000 --output results/dx0_case_001.json

# SDBench evaluation
python -m sdbench.cli --mode ensemble --cases data/sdbench/cases/ --output results/sdbench_summary.csv

# Main CLI entry point (comprehensive interface)
python cli.py --config settings.yml --db cases.json --case 1 --rubric r.json --costs c.csv

# Web UI for physician interaction
uvicorn sdb.ui.app:app --reload     # Start development server on localhost:8000
```

### Data Operations
```bash
python -m sdb.ingest.pipeline       # Refresh NEJM CPC dataset  
python scripts/update_cases.py      # Append new cases only
python cli.py fhir-import report.json --case-id new_case > case.json  # Import FHIR data
```

## Architecture Overview

### Core Components
This repository contains two integrated systems:

**Dx0**: A multi-agent diagnostic orchestrator that simulates a panel of physician personas (Hypothesis, Test-Chooser, Challenger, Stewardship, Checklist) conducting iterative diagnostic debate with cost tracking via CPT/CMS mapping.

**SDBench**: A benchmark suite that processes 304 NEJM Clinical Pathological Conference cases into interactive diagnostic tasks with evaluation metrics.

### Key Architectural Patterns

**Orchestrator Pattern**: `sdb.orchestrator.Orchestrator` coordinates panel actions through `VirtualPanel` instances, managing turn-based debate rounds with budget enforcement via `BudgetManager`.

**Plugin Architecture**: Decision engines are pluggable via entry points (`dx0.personas`, `sdb.retrieval_plugins`, `dx0.cost_estimators`). The system supports rule-based (`RuleEngine`) and LLM-based (`LLMEngine`) decision making.

**Multi-Provider LLM Support**: Abstracted through `LLMClient` interface with implementations for OpenAI (`OpenAIClient`), Ollama (`OllamaClient`), and Hugging Face local models (`HFLocalClient`).

**Gatekeeper Pattern**: `sdb.gatekeeper.Gatekeeper` acts as an information oracle, revealing case findings incrementally based on agent requests, with cost tracking for each test ordered.

### Data Flow
1. Cases loaded via `CaseDatabase` (JSON or SQLite backend)
2. `Gatekeeper` mediates information access for case findings
3. `VirtualPanel` generates actions through `DecisionEngine`
4. `Orchestrator` coordinates panel debate and enforces budget limits
5. `Evaluator` scores diagnostic accuracy using 5-point Likert rubric
6. Results aggregated for statistical analysis and visualization

### Directory Structure
- `sdb/`: Core framework (orchestrator, panel, gatekeeper, evaluation)
- `sdb/ingest/`: Data pipeline for NEJM case processing
- `sdb/ui/`: Web interface for physician interaction
- `sdb/services/`: Supporting services (budget, metrics, results)
- `cli.py`: Unified command-line interface
- `data/`: NEJM CPC cases and CMS pricing data (DVC-tracked)
- `prompts/`: Persona and system prompt templates
- `tests/`: Comprehensive test suite with pytest

### Configuration and Settings
The system supports YAML configuration files loaded via `--config`:
```yaml
openai_api_key: sk-your-key
openai_model: gpt-4
case_db: data/sdbench/cases
case_db_sqlite: cases.db
metrics_port: 8000
```

### Budget and Cost Management
Budget tracking is centralized through `BudgetManager` which uses `CostEstimator` to map test names to CPT codes and CMS prices. Tests are tracked cumulatively with configurable spending limits.

### Evaluation Modes
- **Instant Answer**: Immediate diagnosis without iterative refinement
- **Question-Only**: Text-based information gathering only
- **Budgeted**: Cost-constrained diagnostic process
- **Unconstrained**: No budget limitations
- **Ensemble**: Multiple model voting with weighted aggregation

### Testing Strategy
Comprehensive test coverage including unit tests for business logic, integration tests for API endpoints, and end-to-end tests using Playwright for UI functionality. Tests use pytest with async support and mock external dependencies.