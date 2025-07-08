import json
import csv
import subprocess
import sys

import cli
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
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "cpt_code", "price"]
        )
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
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "cpt_code", "price"]
        )
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
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "cpt_code", "price"]
        )
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
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "cpt_code", "price"]
        )
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
        entry["resource"]["resourceType"] == "ServiceRequest"
        for entry in data["entry"]
    )
