import json
import csv
import subprocess
import sys
import yaml

import cli
from pathlib import Path
from sdb.llm_client import LLMClient


def test_cli_outputs_final_results(tmp_path):
    cases = [{"id": "1", "summary": "viral infection", "full_text": "cough"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "rubric.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "costs.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
        writer.writeheader()
        writer.writerow(
            {
                "test_name": "complete blood count",
                "cpt_code": "100",
                "price": "10",
            }
        )

    cmd = [
        sys.executable,
        "cli.py",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--panel-engine",
        "rule",
        "--llm-model",
        "gpt-4",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert "Final diagnosis" in result.stdout
    assert "Total cost" in result.stdout
    assert "Session score" in result.stdout
    assert "Total time" in result.stdout


def test_cli_flag_parsing(tmp_path):
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "c.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
        writer.writeheader()
        writer.writerow(
            {
                "test_name": "complete blood count",
                "cpt_code": "1",
                "price": "1",
            }
        )

    cmd = [
        sys.executable,
        "cli.py",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--panel-engine",
        "llm",
        "--llm-provider",
        "openai",
        "--llm-model",
        "turbo",
        "--quiet",
        "--semantic-retrieval",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0


def test_cli_stats_command(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    for path, v1, v2 in [(a, 1, 0), (b, 0, 0)]:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["score"])
            writer.writeheader()
            writer.writerow({"score": str(v1)})
            writer.writerow({"score": str(v2)})

    cmd = [
        sys.executable,
        "cli.py",
        "stats",
        str(a),
        str(b),
        "--column",
        "score",
        "--rounds",
        "100",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "p-value" in result.stdout


def test_cli_with_sqlite(tmp_path):
    from sdb.sqlite_db import save_to_sqlite

    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    db_path = tmp_path / "cases.db"
    save_to_sqlite(str(db_path), cases)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "c.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
        writer.writeheader()
        writer.writerow({"test_name": "x", "cpt_code": "1", "price": "1"})

    cmd = [
        sys.executable,
        "cli.py",
        "--db-sqlite",
        str(db_path),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--panel-engine",
        "rule",
        "--llm-model",
        "gpt-4",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0


def test_batch_eval_command(tmp_path):
    cases = [
        {"id": "1", "summary": "a", "full_text": "x"},
        {"id": "2", "summary": "b", "full_text": "y"},
    ]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "c.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
        writer.writeheader()
        writer.writerow(
            {
                "test_name": "complete blood count",
                "cpt_code": "1",
                "price": "1",
            }
        )

    out_file = tmp_path / "out.csv"
    cmd = [
        sys.executable,
        "cli.py",
        "batch-eval",
        "--db",
        str(case_file),
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--output",
        str(out_file),
        "--panel-engine",
        "rule",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    with open(out_file, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert set(rows[0].keys()) == {
        "id",
        "total_cost",
        "score",
        "correct",
        "duration",
    }


def test_cli_cache_size(monkeypatch, tmp_path):
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "c.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_name", "cpt_code", "price"],
        )
        writer.writeheader()
        writer.writerow({"test_name": "x", "cpt_code": "1", "price": "1"})

    captured: dict[str, int | str | None] = {}

    class DummyClient(LLMClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["path"] = kwargs.get("cache_path")
            captured["size"] = kwargs.get("cache_size")

        def _chat(self, messages, model):
            return None

    monkeypatch.setattr(cli, "OpenAIClient", DummyClient)
    monkeypatch.setattr(cli, "start_metrics_server", lambda *_: None)

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            self.finished = True
            self.total_time = 0.0
            self.final_diagnosis = ""
            self.ordered_tests = []

        def run_turn(self, *_args, **_kwargs):
            return ""

    monkeypatch.setattr(cli, "Orchestrator", DummyOrchestrator)

    argv = [
        "cli.py",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--panel-engine",
        "llm",
        "--llm-provider",
        "openai",
        "--cache",
        "--cache-size",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert captured["path"] == "llm_cache.jsonl"
    assert captured["size"] == 5


def test_cli_cross_encoder_flag(monkeypatch, tmp_path):
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "c.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_name", "cpt_code", "price"],
        )
        writer.writeheader()
        writer.writerow({"test_name": "x", "cpt_code": "1", "price": "1"})

    captured: dict[str, str | None] = {}

    class DummyGatekeeper:
        def __init__(self, *_args, **kwargs):
            captured["ce"] = kwargs.get("cross_encoder_name")

        def register_test_result(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(cli, "Gatekeeper", DummyGatekeeper)
    monkeypatch.setattr(cli, "start_metrics_server", lambda *_: None)

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            self.finished = True
            self.total_time = 0.0
            self.final_diagnosis = ""
            self.ordered_tests = []

        def run_turn(self, *_args, **_kwargs):
            return ""

    monkeypatch.setattr(cli, "Orchestrator", DummyOrchestrator)

    argv = [
        "cli.py",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--cross-encoder-model",
        "ce",
        "--semantic-retrieval",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert captured["ce"] == "ce"


def test_cli_ollama_base_url(monkeypatch, tmp_path):
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "c.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_name", "cpt_code", "price"],
        )
        writer.writeheader()
        writer.writerow({"test_name": "x", "cpt_code": "1", "price": "1"})

    captured: dict[str, str | None] = {}

    class DummyOllamaClient(LLMClient):
        def __init__(self, *args, **kwargs):
            super().__init__(
                cache_path=kwargs.get("cache_path"),
                cache_size=kwargs.get("cache_size"),
            )
            captured["url"] = kwargs.get("base_url")

        def _chat(self, messages, model):
            return None

    monkeypatch.setattr(cli, "OllamaClient", DummyOllamaClient)
    monkeypatch.setattr(cli, "start_metrics_server", lambda *_: None)

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            self.finished = True
            self.total_time = 0.0
            self.final_diagnosis = ""
            self.ordered_tests = []

        def run_turn(self, *_args, **_kwargs):
            return ""

    monkeypatch.setattr(cli, "Orchestrator", DummyOrchestrator)

    argv = [
        "cli.py",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--panel-engine",
        "llm",
        "--llm-provider",
        "ollama",
        "--ollama-base-url",
        "http://127.0.0.1:11434",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert captured["url"] == "http://127.0.0.1:11434"


def test_cli_persona_models(monkeypatch, tmp_path):
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "c.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_name", "cpt_code", "price"],
        )
        writer.writeheader()
        writer.writerow({"test_name": "x", "cpt_code": "1", "price": "1"})

    captured: dict[str, dict] = {}

    class DummyLLMEngine:
        def __init__(self, *args, **kwargs):
            captured["models"] = kwargs.get("persona_models")

    monkeypatch.setattr(cli, "LLMEngine", DummyLLMEngine)
    monkeypatch.setattr(cli, "start_metrics_server", lambda *_: None)

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            self.finished = True
            self.total_time = 0.0
            self.final_diagnosis = ""
            self.ordered_tests = []

        def run_turn(self, *_args, **_kwargs):
            return ""

    monkeypatch.setattr(cli, "Orchestrator", DummyOrchestrator)

    argv = [
        "cli.py",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--panel-engine",
        "llm",
        "--persona-models",
        '{"hypothesis_system": "gpt-4"}',
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert captured["models"] == {"hypothesis_system": "gpt-4"}


def test_cli_budget_limit(monkeypatch, tmp_path):
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "c.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_name", "cpt_code", "price"],
        )
        writer.writeheader()
        writer.writerow({"test_name": "x", "cpt_code": "1", "price": "1"})

    captured: dict[str, float | None] = {}

    class DummyBudgetManager:
        def __init__(self, *args, **kwargs):
            captured["limit"] = kwargs.get("budget")

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            self.finished = True
            self.total_time = 0.0
            self.final_diagnosis = ""
            self.ordered_tests = []

        def run_turn(self, *_args, **_kwargs):
            return ""

    monkeypatch.setattr(cli, "BudgetManager", DummyBudgetManager)
    monkeypatch.setattr(cli, "Orchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "start_metrics_server", lambda *_: None)

    argv = [
        "cli.py",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--budget-limit",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert captured["limit"] == 5.0


def test_cli_budgeted_mode_enforces_budget(monkeypatch, tmp_path):
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "c.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_name", "cpt_code", "price"],
        )
        writer.writeheader()
        writer.writerow({"test_name": "x", "cpt_code": "1", "price": "1"})

    captured: dict[str, float | int] = {"turns": 0}

    class DummyBudgetManager:
        def __init__(self, *args, **kwargs):
            self.budget = float(kwargs.get("budget"))
            self.spent = 0.0
            captured["budget"] = self.budget

        def add_test(self, *_args, **_kwargs):
            self.spent += 5.0

        def over_budget(self) -> bool:
            return self.spent >= self.budget

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            self.budget_manager = kwargs.get("budget_manager")
            self.finished = False
            self.total_time = 0.0
            self.final_diagnosis = ""
            self.ordered_tests = []

        def run_turn(self, *_args, **_kwargs):
            captured["turns"] += 1
            self.budget_manager.add_test("x")
            if self.budget_manager.over_budget():
                self.finished = True
            return ""

    monkeypatch.setattr(cli, "BudgetManager", DummyBudgetManager)
    monkeypatch.setattr(cli, "Orchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "start_metrics_server", lambda *_: None)

    argv = [
        "cli.py",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--mode",
        "budgeted",
        "--budget",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()

    assert captured["budget"] == 5.0
    assert captured["turns"] == 1


def test_cli_cost_table_custom(monkeypatch, tmp_path):
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "table.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
        writer.writeheader()
        writer.writerow({"test_name": "x", "cpt_code": "1", "price": "1"})

    captured: dict[str, object] = {}

    class DummyCostEstimator:
        pass

    def dummy_load(path: str) -> DummyCostEstimator:
        captured["path"] = path
        return DummyCostEstimator()

    class DummyBudgetManager:
        def __init__(self, estimator, *args, **kwargs):
            captured["estimator"] = estimator

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            self.finished = True
            self.total_time = 0.0
            self.final_diagnosis = ""
            self.ordered_tests = []

        def run_turn(self, *_args, **_kwargs):
            return ""

    monkeypatch.setattr(cli.CostEstimator, "load_from_csv", staticmethod(dummy_load))
    monkeypatch.setattr(cli, "BudgetManager", DummyBudgetManager)
    monkeypatch.setattr(cli, "Orchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "start_metrics_server", lambda *_: None)

    argv = [
        "cli.py",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--cost-table",
        str(cost_file),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()

    assert captured["path"] == str(cost_file)
    assert isinstance(captured["estimator"], DummyCostEstimator)


def test_batch_eval_ollama_base_url(monkeypatch, tmp_path):
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "c.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_name", "cpt_code", "price"],
        )
        writer.writeheader()
        writer.writerow({"test_name": "x", "cpt_code": "1", "price": "1"})

    out_file = tmp_path / "out.csv"

    captured: dict[str, str | None] = {}

    class DummyOllamaClient(LLMClient):
        def __init__(self, *args, **kwargs):
            super().__init__(
                cache_path=kwargs.get("cache_path"),
                cache_size=kwargs.get("cache_size"),
            )
            captured["url"] = kwargs.get("base_url")

        def _chat(self, messages, model):
            return None

    def dummy_batch_eval(cases, run_case, concurrency=1):
        return [run_case(cid) for cid in cases]

    monkeypatch.setattr(cli, "OllamaClient", DummyOllamaClient)
    monkeypatch.setattr(cli, "batch_evaluate", dummy_batch_eval)
    monkeypatch.setattr(cli, "start_metrics_server", lambda *_: None)

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            self.finished = True
            self.total_time = 0.0
            self.final_diagnosis = ""
            self.ordered_tests = []

        def run_turn(self, *_args, **_kwargs):
            return ""

    monkeypatch.setattr(cli, "Orchestrator", DummyOrchestrator)

    argv = [
        "--db",
        str(case_file),
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--output",
        str(out_file),
        "--panel-engine",
        "llm",
        "--llm-provider",
        "ollama",
        "--ollama-base-url",
        "http://127.0.0.1:11434",
    ]
    cli.batch_eval_main(argv)
    assert captured["url"] == "http://127.0.0.1:11434"


def test_cli_vote_weights(monkeypatch, tmp_path, capsys):
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "c.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
        writer.writeheader()
        writer.writerow({"test_name": "x", "cpt_code": "1", "price": "1"})

    weights_path = Path(__file__).parent / "vote_weights.json"
    with open(weights_path, "r", encoding="utf-8") as f:
        weights = json.load(f)

    captured: dict[str, object] = {}

    class DummyMetaPanel:
        def __init__(self, *args, **kwargs):
            captured["weights"] = kwargs.get("weights")

        def synthesize(self, results):
            captured["results"] = results
            return "weighted"

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            self.finished = True
            self.total_time = 0.0
            self.final_diagnosis = ""
            self.ordered_tests = []

        def run_turn(self, *_args, **_kwargs):
            return ""

    monkeypatch.setattr(cli, "MetaPanel", DummyMetaPanel)
    monkeypatch.setattr(cli, "Orchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "start_metrics_server", lambda *_: None)

    argv = [
        "cli.py",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--vote-weights",
        str(weights_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()

    output = capsys.readouterr().out
    assert "weighted" in output
    assert captured["weights"] == weights
    assert len(captured["results"]) == 1


def test_cli_weights_file_affects_output(monkeypatch, tmp_path, capsys):
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    rubric_file = tmp_path / "r.json"
    with open(rubric_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    cost_file = tmp_path / "c.csv"
    with open(cost_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_name", "cpt_code", "price"])
        writer.writeheader()
        writer.writerow({"test_name": "x", "cpt_code": "1", "price": "1"})

    weights_a = {"r1": 2.0, "r2": 1.0}
    weights_b = {"r1": 0.1, "r2": 5.0}
    file_a = tmp_path / "w1.yaml"
    file_b = tmp_path / "w2.yaml"
    with open(file_a, "w", encoding="utf-8") as f:
        yaml.safe_dump(weights_a, f)
    with open(file_b, "w", encoding="utf-8") as f:
        yaml.safe_dump(weights_b, f)

    class DummyMetaPanel:
        def __init__(self, *args, **kwargs):
            self.weights = kwargs.get("weights")
            self.voter = cli.WeightedVoter()

        def synthesize(self, _results):
            preds = [
                cli.DiagnosisResult("a", 1.0, run_id="r1"),
                cli.DiagnosisResult("b", 1.0, run_id="r2"),
            ]
            return self.voter.vote(preds, weights=self.weights)

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            self.finished = True
            self.total_time = 0.0
            self.final_diagnosis = ""
            self.ordered_tests = []

        def run_turn(self, *_args, **_kwargs):
            return ""

    monkeypatch.setattr(cli, "MetaPanel", DummyMetaPanel)
    monkeypatch.setattr(cli, "Orchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "start_metrics_server", lambda *_: None)

    argv = [
        "cli.py",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--rubric",
        str(rubric_file),
        "--costs",
        str(cost_file),
        "--weights-file",
        str(file_a),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    out_a = capsys.readouterr().out

    argv[-1] = str(file_b)
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    out_b = capsys.readouterr().out

    assert out_a != out_b


def test_fhir_export_command(tmp_path):
    transcript = [["panel", "hello"], ["gatekeeper", "hi"]]
    tests = ["cbc"]
    t_file = tmp_path / "t.json"
    with open(t_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f)
    tests_file = tmp_path / "tests.json"
    with open(tests_file, "w", encoding="utf-8") as f:
        json.dump(tests, f)

    cmd = [
        sys.executable,
        "cli.py",
        "fhir-export",
        str(t_file),
        str(tests_file),
        "--patient-id",
        "p1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["resourceType"] == "Bundle"
    assert any(
        entry["resource"]["resourceType"] == "ServiceRequest" for entry in data["entry"]
    )


def test_fhir_import_command(tmp_path):
    report = {
        "resourceType": "DiagnosticReport",
        "conclusion": "Summary",
        "result": [{"reference": "#o1"}],
        "contained": [
            {"resourceType": "Observation", "id": "o1", "valueString": "step"}
        ],
    }
    r_file = tmp_path / "report.json"
    with open(r_file, "w", encoding="utf-8") as f:
        json.dump(report, f)

    cmd = [
        sys.executable,
        "cli.py",
        "fhir-import",
        str(r_file),
        "--case-id",
        "c1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["id"] == "c1"
    assert data["summary"] == "Summary"
    assert data["steps"][0]["text"] == "step"


def test_fhir_import_bundle_command(tmp_path):
    report = {
        "resourceType": "DiagnosticReport",
        "conclusion": "sum",
        "result": [{"reference": "Observation/o1"}],
    }
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": report},
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "o1",
                    "valueString": "foo",
                }
            },
        ],
    }
    b_file = tmp_path / "bundle.json"
    with open(b_file, "w", encoding="utf-8") as f:
        json.dump(bundle, f)

    cmd = [sys.executable, "cli.py", "fhir-import", str(b_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["summary"] == "sum"
    assert data["steps"][0]["text"] == "foo"


def test_annotate_case_command(tmp_path):
    cases = [{"id": "1", "summary": "s", "full_text": "t"}]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    mapping = {"cbc": "complete blood count"}
    map_file = tmp_path / "map.json"
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f)

    out_file = tmp_path / "annot" / "1.json"

    cmd = [
        sys.executable,
        "cli.py",
        "annotate-case",
        "--db",
        str(case_file),
        "--case",
        "1",
        "--notes",
        "note",
        "--test-mapping",
        str(map_file),
        "--output",
        str(out_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["id"] == "1"
    assert data["notes"] == "note"
    assert data["test_mappings"] == mapping


def test_filter_cases_command(tmp_path):
    cases = [
        {
            "id": "1",
            "summary": "fever and cough",
            "full_text": "cough",
            "tag": "respiratory",
        },
        {"id": "2", "summary": "abdominal pain", "full_text": "pain", "tag": "gi"},
    ]
    case_file = tmp_path / "cases.json"
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(cases, f)

    out_json = tmp_path / "subset.json"
    cmd = [
        sys.executable,
        "cli.py",
        "filter-cases",
        "--db",
        str(case_file),
        "--keywords",
        "cough",
        "--output",
        str(out_json),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["id"] == "1"

    meta_file = tmp_path / "meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({"tag": "gi"}, f)

    out_csv = tmp_path / "subset.csv"
    cmd = [
        sys.executable,
        "cli.py",
        "filter-cases",
        "--db",
        str(case_file),
        "--metadata",
        str(meta_file),
        "--output",
        str(out_csv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    with open(out_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["id"] == "2"


def test_export_fhir_command(tmp_path):
    transcript = [["panel", "hello"], ["gatekeeper", "hi"]]
    t_file = tmp_path / "t.json"
    with open(t_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f)

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "cli.py",
        "export-fhir",
        str(t_file),
        "--case-id",
        "c1",
        "--output-dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    out_file = out_dir / "c1.json"
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["resourceType"] == "Bundle"
    assert data["entry"][0]["resource"]["sender"]["display"] == "panel"
